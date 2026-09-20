"""Metric provenance and explicit unavailable values, shared across artifacts."""
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unavailable(source: str, reason: str, scope: str) -> dict:
    return {"value": None, "status": "unavailable", "source": source,
            "scope": scope, "reason": reason}


REQUEST_SOURCES = {
    "input_tokens": "response.usage.prompt_tokens",
    "output_tokens": "response.usage.completion_tokens",
    "cached_input_tokens": "response.usage.prompt_tokens_details.cached_tokens",
    "cached_input_fraction": "cached_input_tokens / input_tokens",
    "ttft_s": "client perf_counter: HTTP start to first nonempty content/reasoning delta",
    "latency_s": "client perf_counter: HTTP start to stream close/error",
    "stream_generation_s": "client perf_counter: first to last nonempty output delta",
    "server_ttft_s": "response.metrics.time_to_first_token_ms / 1000",
    "decode_time_s": "response.metrics.generation_time_ms / 1000",
    "server_queue_s": "response.metrics.queue_time_ms / 1000",
    "server_mean_itl_s": "response.metrics.mean_itl_ms / 1000",
    "server_output_tokens_per_s": "response.metrics.tokens_per_second",
    "decode_tokens_per_s": "(response.usage.completion_tokens - 1) / server decode_time_s",
    "output_tokens_per_s": "response.usage.completion_tokens / client latency_s",
    "prefill_s": "not exposed by supported per-request response fields",
    "kv_cache_utilization": "server-scoped /metrics samples in telemetry.jsonl",
    "gpu_memory_used_mib": "local-host nvidia-smi samples in telemetry.jsonl",
    "gpu_utilization_percent": "local-host nvidia-smi samples in telemetry.jsonl",
}


def finalize_request_metrics(result):
    result.metric_sources = dict(REQUEST_SOURCES)
    for name in REQUEST_SOURCES:
        if getattr(result, name) is None:
            result.unsupported.setdefault(name, "not returned by this response or not measurable")
    for name in ("kv_cache_utilization", "gpu_memory_used_mib", "gpu_utilization_percent"):
        result.unsupported[name] = "sampled telemetry is not attributable to an individual request"
    result.unsupported["prefill_s"] = "server TTFT is not a measurement of pure prefill time"
