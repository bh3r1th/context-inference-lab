import asyncio
import json

import httpx
import pytest

from contextbench.metrics.gpu import parse_gpu, sample_gpu
from contextbench.metrics.summary import summarize
from contextbench.metrics.telemetry import (
    prefix_intervals,
    prometheus_samples,
    sample_telemetry,
)


def snapshot(hits=1, queries=2, labels='{engine="0"}'):
    text = (f"vllm:prefix_cache_hits_total{labels} {hits}\n"
            f"vllm:prefix_cache_queries_total{labels} {queries}\n")
    return {"prefix_cache": {"samples": prometheus_samples(text)}}


def test_cache_deltas_use_interval_and_preserve_labels():
    intervals = prefix_intervals([snapshot(1000, 2000), snapshot(1030, 2050)])
    hit = next(i for i in intervals if "hits" in i["metric"])
    assert hit["delta"] == 30 and hit["hit_ratio"] == .6
    assert hit["labels"] == '{engine="0"}' and hit["scope"] == "server"


@pytest.mark.parametrize("samples", [[snapshot()], [snapshot(), snapshot(0, 1)],
                                     [snapshot(), snapshot(labels='{engine="1"}')],
                                     [snapshot(), {"prefix_cache": {"samples": None}}]])
def test_unobservable_cache_intervals_stay_null(samples):
    hit = next(i for i in prefix_intervals(samples) if "hits" in i["metric"])
    assert hit["delta"] is None and hit["reason"] and hit["hit_ratio"] is None


def test_zero_queries_cannot_produce_a_hit_ratio():
    hit = next(i for i in prefix_intervals([snapshot(), snapshot()]) if "hits" in i["metric"])
    assert hit["delta"] == 0 and hit["hit_ratio"] is None and hit["hit_ratio_reason"]


def test_invalid_prometheus_values_are_explicit_and_raw_timing_is_server_scoped():
    values = prometheus_samples('vllm:kv_cache_usage_perc NaN\n'
                               'vllm:request_decode_time_seconds_sum{model_name="test"} 3.4\n'
                               'vllm:request_queue_time_seconds_count 9\n')
    assert values[0]["value"] is None and values[0]["reason"]
    assert values[1]["value"] == 3.4 and values[1]["scope"] == "server"
    json.dumps(values, allow_nan=False)


@pytest.mark.parametrize("url,status", [(None, 200), ("http://test/metrics", 503),
                                       ("http://test/metrics", 200)])
def test_unavailable_telemetry_fields_have_reasons(url, status):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda r: httpx.Response(status, text="unrelated metric\n"))) as client:
            return await sample_telemetry(client, url, {}, False)
    result = asyncio.run(run())
    for name in ("kv_cache", "prefix_cache", "server_observations"):
        assert result[name]["samples"] is None and result[name]["reason"] and result[name]["source"]
    assert result["gpu"]["devices"] is None and result["gpu"]["reason"]


def test_gpu_fields_can_be_independently_unavailable():
    gpu = parse_gpu("GPU-1, Example GPU, 570.00, 0000:01:00.0, N/A, 2048, 8192\n")[0]
    assert gpu["name"] == "Example GPU" and gpu["driver_version"] == "570.00"
    assert gpu["utilization_percent"] is None and gpu["memory_used_mib"] == 2048
    assert gpu["unsupported"]["utilization_percent"]


def test_missing_nvidia_smi_is_explicit(monkeypatch):
    async def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi unavailable")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing)
    result = asyncio.run(sample_gpu())
    assert result["devices"] is None and result["scope"] == "local_host"
    assert result["reason"] and "nvidia-smi" in result["source"]


def row(input_tokens, output_tokens):
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "error": None,
            "correct": True, "schema_valid": True, "rule_recall": 1, "missed_critical_rules": []}


def test_summary_does_not_treat_missing_usage_as_zero():
    result = summarize([row(None, None)], 2)
    assert result["requests_per_s"] == .5
    assert result["token_totals"]["output_tokens"]["observed_total"] is None
    assert result["token_totals"]["output_tokens"]["reason"]
    assert result["throughput"]["output_tokens_per_s"]["value"] is None
    assert result["successful_request_metrics"]["decode_time_s"]["p50"] is None
    mixed = summarize([row(10, 5), row(None, None)], 2)
    assert mixed["token_totals"]["output_tokens"]["observed_total"] == 5
    assert mixed["throughput"]["output_tokens_per_s"]["value"] is None
    complete = summarize([row(10, 5), row(20, 7)], 2)
    assert complete["throughput"]["output_tokens_per_s"]["value"] == 6
