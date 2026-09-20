"""Streaming vLLM/OpenAI-compatible transport with observed, sourced measurements."""
import json
import math
import os
from time import perf_counter

import httpx

from contextbench.context.prompt import render_messages
from contextbench.experiments.config import Server
from contextbench.metrics.provenance import finalize_request_metrics, unavailable, utc_now
from contextbench.models import Context, Decision, InferenceResult


def reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        reject_constant(value)
    return result


async def sse_data(response):
    """Handle SSE comments, CRLF, multiline data, and a final unterminated event."""
    lines = []
    async for line in response.aiter_lines():
        if not line:
            if lines:
                yield "\n".join(lines)
                lines = []
        elif line.startswith("data:"):
            lines.append(line[5:].removeprefix(" "))
    if lines:
        yield "\n".join(lines)


def assign_metric(result, field, value, *, integer=False, scale=1.0):
    if value is None:
        return
    valid = type(value) is int if integer else type(value) in (int, float)
    if not valid or not math.isfinite(value) or value < 0:
        setattr(result, field, None)
        result.unsupported[field] = "server returned an invalid or negative numeric value"
        return
    setattr(result, field, value if integer else value * scale)
    result.unsupported.pop(field, None)


class VLLMBackend:
    def __init__(self, config: Server, client: httpx.AsyncClient):
        self.config, self.client = config, client

    @property
    def headers(self) -> dict[str, str]:
        key = os.getenv(self.config.api_key_env)
        return {"Authorization": f"Bearer {key}"} if key else {}

    async def metadata(self) -> dict:
        response = await self.client.get(self.config.base_url.rstrip("/") + "/models",
                                         headers=self.headers)
        response.raise_for_status()
        data = response.json()
        if self.config.model not in [m["id"] for m in data["data"]]:
            raise ValueError("configured model is not served by endpoint")
        return data

    async def version(self) -> dict:
        url = self.config.version_url or self.config.base_url.rstrip("/").removesuffix("/v1") + "/version"
        try:
            response = await self.client.get(url, headers=self.headers, timeout=5)
            response.raise_for_status()
            version = response.json().get("version")
            if not isinstance(version, str) or not version:
                raise ValueError("version endpoint did not return a version string")
            return {"value": version, "source": url, "scope": "server", "reason": None,
                    "status": "available"}
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            return unavailable(url, f"{type(exc).__name__}: {exc}", "server")

    async def generate(self, context: Context, event: dict, seed: int) -> InferenceResult:
        payload = {
            "model": self.config.model,
            "messages": list(context.messages or render_messages(context.text, event)),
            "temperature": 0, "seed": seed, "max_tokens": self.config.max_tokens, "n": 1,
            "stream": True, "stream_options": {"include_usage": True},
        }
        if self.config.structured_output:
            if self.config.structured_output_mode == "vllm":
                payload["structured_outputs"] = {"json": Decision.model_json_schema()}
            else:
                payload["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "expense_decision", "strict": True, "schema": Decision.model_json_schema()}}
        result = InferenceResult(model=self.config.model, started_at=utc_now())
        start = perf_counter()
        last_content, content_chunks, done = None, 0, False
        try:
            async with self.client.stream("POST", self.config.base_url.rstrip("/") +
                                          "/chat/completions", json=payload,
                                          headers=self.headers,
                                          timeout=self.config.timeout_s) as response:
                result.http_status = response.status_code
                response.raise_for_status()
                async for raw in sse_data(response):
                    elapsed = perf_counter() - start
                    result.stream_events.append({"elapsed_s": elapsed, "data": raw})
                    if raw.strip() == "[DONE]":
                        done = True
                        break
                    data = json.loads(raw, parse_constant=reject_constant, parse_float=finite_float)
                    if not isinstance(data, dict):
                        raise ValueError("SSE payload must be an object")
                    if data.get("error"):
                        raise ValueError(f"server stream error: {data['error']}")
                    result.response_id = data.get("id", result.response_id)
                    result.response_model = data.get("model", result.response_model)
                    usage = data.get("usage")
                    if usage is not None:
                        if not isinstance(usage, dict):
                            raise ValueError("usage must be an object")
                        result.raw_usage = usage
                        assign_metric(result, "input_tokens", usage.get("prompt_tokens"), integer=True)
                        assign_metric(result, "output_tokens", usage.get("completion_tokens"), integer=True)
                        details = usage.get("prompt_tokens_details") or {}
                        if not isinstance(details, dict):
                            raise ValueError("prompt_tokens_details must be an object")
                        assign_metric(result, "cached_input_tokens", details.get("cached_tokens"),
                                      integer=True)
                    metrics = data.get("metrics")
                    if metrics is not None:
                        if not isinstance(metrics, dict):
                            raise ValueError("metrics must be an object")
                        result.raw_server_metrics = metrics
                        for source, target in (
                            ("time_to_first_token_ms", "server_ttft_s"),
                            ("generation_time_ms", "decode_time_s"),
                            ("queue_time_ms", "server_queue_s"), ("mean_itl_ms", "server_mean_itl_s"),
                        ):
                            assign_metric(result, target, metrics.get(source), scale=.001)
                        assign_metric(result, "server_output_tokens_per_s", metrics.get("tokens_per_second"))
                    for choice in data.get("choices", []):
                        if choice.get("index", 0) != 0:
                            raise ValueError("unexpected additional completion choice")
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or ""
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if not isinstance(content, str) or not isinstance(reasoning, str):
                            raise ValueError("streamed content/reasoning must be text")
                        if content or reasoning:
                            last_content = elapsed
                            content_chunks += 1
                            if result.ttft_s is None:
                                result.ttft_s = elapsed
                            result.text += content
                            result.reasoning_text += reasoning
                        if choice.get("finish_reason"):
                            result.finish_reason = choice["finish_reason"]
            if not done or result.finish_reason is None:
                raise ValueError("incomplete SSE stream")
            if result.finish_reason != "stop":
                result.error = f"non-success finish reason: {result.finish_reason}"
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        result.latency_s = perf_counter() - start
        result.completed_at = utc_now()
        if content_chunks > 1:
            result.stream_generation_s = last_content - result.ttft_s
        else:
            result.unsupported["stream_generation_s"] = "fewer than two nonempty output deltas"
        if result.error is None and result.output_tokens is not None:
            if result.latency_s > 0:
                result.output_tokens_per_s = result.output_tokens / result.latency_s
            if result.output_tokens > 1 and result.decode_time_s and result.decode_time_s > 0:
                result.decode_tokens_per_s = (result.output_tokens - 1) / result.decode_time_s
        if result.decode_tokens_per_s is None:
            result.unsupported["decode_tokens_per_s"] = (
                "requires a successful response, server decode timing > 0, and output token count > 1; "
                "SSE chunks are not individual tokens")
        if result.cached_input_tokens is not None and result.input_tokens is not None:
            if result.cached_input_tokens > result.input_tokens:
                result.cached_input_tokens = None
                result.unsupported["cached_input_tokens"] = "cached token count exceeds input count"
            elif result.input_tokens > 0:
                result.cached_input_fraction = result.cached_input_tokens / result.input_tokens
        finalize_request_metrics(result)
        return result
