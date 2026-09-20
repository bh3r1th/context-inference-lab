import asyncio
import json
from dataclasses import asdict

import httpx
import pytest

from contextbench.experiments.config import Server
from contextbench.inference.vllm import VLLMBackend
from contextbench.models import Context


def execute(chunks, *, config=None, inspect=None):
    async def run():
        def handler(request):
            if inspect:
                inspect(json.loads(request.content))
            text = "".join("data: " + (c if isinstance(c, str) else json.dumps(c)) + "\n\n"
                           for c in chunks)
            return httpx.Response(200, text=text)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await VLLMBackend(config or Server(model_revision="test", vllm_version="test",
                                                      launch_command="test"), client).generate(
                                                          Context("rules", ()), {}, 17)
    return asyncio.run(run())


def content(text, finish=None):
    return {"id": "response-id", "model": "served-model", "choices": [
        {"index": 0, "delta": {"content": text}, "finish_reason": finish}]}


def test_exact_client_and_server_measurements(monkeypatch):
    # Controlled clock verifies units and boundaries without flaky wall-clock thresholds.
    ticks = iter([10.0, 10.1, 10.4, 10.5, 10.6, 10.7])
    monkeypatch.setattr("contextbench.inference.vllm.perf_counter", lambda: next(ticks))
    result = execute([content("{", None), content("}", "stop"),
                      {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 11,
                       "prompt_tokens_details": {"cached_tokens": 60}},
                       "metrics": {"time_to_first_token_ms": 50, "generation_time_ms": 200,
                                   "queue_time_ms": 20, "mean_itl_ms": 20,
                                   "tokens_per_second": 44}}, "[DONE]"])
    assert result.error is None
    assert result.ttft_s == pytest.approx(.1)
    assert result.latency_s == pytest.approx(.7)
    assert result.stream_generation_s == pytest.approx(.3)
    assert result.server_ttft_s == pytest.approx(.05)
    assert result.decode_time_s == pytest.approx(.2)
    assert result.server_queue_s == pytest.approx(.02)
    assert result.server_mean_itl_s == pytest.approx(.02)
    assert result.server_output_tokens_per_s == 44
    assert result.decode_tokens_per_s == pytest.approx(50)
    assert result.output_tokens_per_s == pytest.approx(11 / .7)
    assert result.cached_input_tokens == 60 and result.cached_input_fraction == .6
    assert result.prefill_s is None
    assert len(result.stream_events) == 4
    assert result.response_model == "served-model" and result.http_status == 200
    assert result.started_at <= result.completed_at
    assert result.raw_usage["completion_tokens"] == 11
    assert result.raw_server_metrics["generation_time_ms"] == 200
    assert "generation_time_ms" in result.metric_sources["decode_time_s"]
    json.dumps(asdict(result), allow_nan=False)


def test_no_decode_estimation_from_multiple_content_chunks():
    result = execute([content("first"), content("more", "stop"),
                      {"usage": {"prompt_tokens": 10, "completion_tokens": 50}}, "[DONE]"])
    assert result.stream_generation_s is not None
    assert result.decode_time_s is None and result.decode_tokens_per_s is None
    assert "SSE chunks" in result.unsupported["decode_tokens_per_s"]
    assert result.server_queue_s is None


def test_missing_metrics_are_null_and_sourced():
    result = execute([content("{}", "stop"), "[DONE]"])
    for field in ("input_tokens", "output_tokens", "cached_input_tokens", "decode_time_s",
                  "server_queue_s", "server_ttft_s", "server_mean_itl_s", "prefill_s",
                  "gpu_memory_used_mib", "gpu_utilization_percent", "kv_cache_utilization"):
        assert getattr(result, field) is None
        assert result.unsupported[field] and result.metric_sources[field]


def test_zero_metrics_are_observations_not_missing():
    result = execute([content("{}", "stop"), {"usage": {"prompt_tokens": 3, "completion_tokens": 1,
                      "prompt_tokens_details": {"cached_tokens": 0}},
                      "metrics": {"queue_time_ms": 0, "generation_time_ms": 0}}, "[DONE]"])
    assert result.server_queue_s == 0 and result.decode_time_s == 0
    assert result.cached_input_fraction == 0
    assert result.decode_tokens_per_s is None
    assert "server_queue_s" not in result.unsupported


@pytest.mark.parametrize("bad", [-1, True, "12"])
def test_invalid_numeric_metrics_are_not_coerced(bad):
    result = execute([content("{}", "stop"),
                      {"usage": {"prompt_tokens": bad}, "metrics": {"queue_time_ms": bad}}, "[DONE]"])
    assert result.input_tokens is None and result.server_queue_s is None
    assert "invalid" in result.unsupported["input_tokens"]


@pytest.mark.parametrize("raw", ['{"usage":{"prompt_tokens":NaN}}',
                                '{"metrics":{"generation_time_ms":1e999}}'])
def test_nonfinite_json_cannot_poison_raw_artifacts(raw):
    result = execute([raw])
    assert result.error and "nonfinite" in result.error
    json.dumps(asdict(result), allow_nan=False)


def test_partial_usage_and_failed_generation_are_retained_without_rates():
    result = execute([content("partial", "length"),
                      {"usage": {"prompt_tokens": 2, "completion_tokens": 3}}, "[DONE]"])
    assert result.error and result.text == "partial" and result.output_tokens == 3
    assert result.output_tokens_per_s is None
    assert result.decode_tokens_per_s is None


def test_reasoning_counts_as_first_observable_generated_output():
    result = execute([{"choices": [{"delta": {"reasoning_content": "thinking"}}]},
                      content("{}", "stop"), "[DONE]"])
    assert result.reasoning_text == "thinking" and result.text == "{}"
    assert result.ttft_s is not None and result.stream_generation_s is not None


@pytest.mark.parametrize("mode,structured,key", [("vllm", True, "structured_outputs"),
                                                 ("json_schema", True, "response_format"),
                                                 ("vllm", False, None)])
def test_request_settings(mode, structured, key):
    config = Server(model="custom", model_revision="abc", vllm_version="test", launch_command="test",
                    max_tokens=37, structured_output=structured, structured_output_mode=mode)
    def inspect(payload):
        assert payload["model"] == "custom" and payload["max_tokens"] == 37
        assert payload["temperature"] == 0 and payload["seed"] == 17 and payload["n"] == 1
        assert payload["stream_options"] == {"include_usage": True}
        if key:
            assert key in payload
        else:
            assert "structured_outputs" not in payload and "response_format" not in payload
    assert execute([content("{}", "stop"), "[DONE]"], config=config, inspect=inspect).error is None


def test_multiline_sse_comments_and_timeout():
    async def run(timeout=False):
        def handler(request):
            if timeout:
                raise httpx.ReadTimeout("test timeout")
            return httpx.Response(200, text=': keepalive\r\n\r\ndata: {"choices":\r\n'
                                  'data: [{"delta":{"content":"{}"},"finish_reason":"stop"}]}\r\n'
                                  '\r\ndata: [DONE]\r\n\r\n')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            config = Server(model_revision="test", vllm_version="test", launch_command="test")
            return await VLLMBackend(config, client).generate(Context("", ()), {}, 1)
    assert asyncio.run(run()).error is None
    timed_out = asyncio.run(run(True))
    assert "ReadTimeout" in timed_out.error and timed_out.latency_s is not None
    assert timed_out.ttft_s is None and timed_out.input_tokens is None


@pytest.mark.parametrize("status,body,expected", [(200, {"version": "test-1"}, "test-1"),
                                                (404, {}, None), (200, {}, None)])
def test_version_discovery_is_optional_and_sourced(status, body, expected):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda r: httpx.Response(status, json=body))) as client:
            config = Server(model_revision="test", vllm_version="declared", launch_command="test")
            return await VLLMBackend(config, client).version()
    result = asyncio.run(run())
    assert result["value"] == expected and result["source"].endswith("/version")
    if expected is None:
        assert result["reason"]
