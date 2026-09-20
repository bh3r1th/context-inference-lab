"""Measured experiment comparison and Markdown artifact generation."""
import json
from typing import Any


STRATEGIES = (
    "full_context", "prefix_cached_full_context", "retrieval", "compiled_context",
)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _series_value(telemetry: dict, group: str, suffix: str, percentile: str = "p95") -> float | None:
    values = []
    for series in (telemetry.get(group, {}).get("series") or {}).values():
        value = series.get(percentile)
        if value is not None:
            values.append(value)
    return _mean(values)


def _gpu_memory(telemetry: dict, percentile: str = "p95") -> float | None:
    values = []
    for device in (telemetry.get("local_gpu", {}).get("devices") or {}).values():
        value = device.get("memory_used_mib", {}).get(percentile)
        if value is not None:
            values.append(value)
    return _mean(values)


def build_report(summaries: list[dict[str, Any]], metadata: dict[str, Any]) -> dict:
    baselines = {
        summary["concurrency"]: summary.get("average_context_tokens")
        for summary in summaries if summary.get("strategy") == "full_context"
    }
    cells = []
    for summary in summaries:
        average_context_tokens = summary.get("average_context_tokens")
        baseline = baselines.get(summary.get("concurrency"))
        reduction = ((baseline - average_context_tokens) / baseline * 100
                     if baseline is not None and baseline > 0 and average_context_tokens is not None
                     else None)
        telemetry = summary.get("telemetry", {})
        timings = summary.get("successful_request_metrics", {})
        quality = summary.get("quality", {})
        cells.append({
            "strategy": summary.get("strategy"),
            "concurrency": summary.get("concurrency"),
            "status": summary.get("status"),
            "average_context_tokens": average_context_tokens,
            "context_reduction_percent": reduction,
            "correctness": quality.get("correct", summary.get("correctness")),
            "critical_rule_miss_rate": summary.get("quality", {}).get(
                "critical_rule_miss_rate"),
            "critical_rules_missed": summary.get("missed_critical_rules"),
            "ttft_p50_s": timings.get("ttft_s", {}).get("p50"),
            "ttft_p95_s": timings.get("ttft_s", {}).get("p95"),
            "latency_p50_s": timings.get("latency_s", {}).get("p50"),
            "latency_p95_s": timings.get("latency_s", {}).get("p95"),
            "throughput_output_tokens_per_s": summary.get("throughput", {}).get(
                "output_tokens_per_s", {}).get("value"),
            "requests_per_s": summary.get("requests_per_s"),
            "kv_cache_usage_p95": _series_value(telemetry, "kv_cache", "usage", "p95"),
            "gpu_memory_used_mib_p95": _gpu_memory(telemetry, "p95"),
            "batch_duration_s": summary.get("batch_duration_s"),
        })
    return {
        "experiment_id": metadata.get("experiment_id"),
        "run_id": metadata.get("run_id"),
        "strategies": sorted({cell["strategy"] for cell in cells if cell["strategy"]}),
        "concurrency": sorted({cell["concurrency"] for cell in cells
                                if cell["concurrency"] is not None}),
        "metric_scope": {
            "context_tokens": "mean of measured context_token_count values",
            "context_reduction_percent": "relative to full_context at the same concurrency",
            "critical_rule_miss_rate": "requests with one or more missed critical rules",
            "kv_cache_usage_p95": "mean p95 of observed server KV-cache series",
            "gpu_memory_used_mib_p95": "mean p95 of observed local GPU devices",
            "unavailable": "null means the required measurement was not observed",
        },
        "cells": cells,
    }


def _value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict) -> str:
    lines = [
        "# Context reduction experiment",
        "",
        f"Run `{report.get('run_id')}` for experiment `{report.get('experiment_id')}`.",
        "",
        "This report compares measured inference performance and decision quality. It makes no "
        "claims for metrics recorded as `n/a`.",
        "",
        "| Strategy | Concurrency | Context tokens | Reduction % | Correctness | Critical miss rate | "
        "TTFT p50/p95 (s) | Latency p50/p95 (s) | Output tok/s | Requests/s | KV p95 | GPU MiB p95 | Batch s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in report["cells"]:
        lines.append("| " + " | ".join([
            _value(cell["strategy"], 0), _value(cell["concurrency"], 0),
            _value(cell["average_context_tokens"]), _value(cell["context_reduction_percent"]),
            _value(cell["correctness"]), _value(cell["critical_rule_miss_rate"]),
            f"{_value(cell['ttft_p50_s'])}/{_value(cell['ttft_p95_s'])}",
            f"{_value(cell['latency_p50_s'])}/{_value(cell['latency_p95_s'])}",
            _value(cell["throughput_output_tokens_per_s"]), _value(cell["requests_per_s"]),
            _value(cell["kv_cache_usage_p95"]), _value(cell["gpu_memory_used_mib_p95"]),
            _value(cell["batch_duration_s"]),
        ]) + " |")
    lines.extend([
        "",
        "## Metric scope",
        "",
        "- Context reduction uses the measured full-context token average at the same concurrency.",
        "- TTFT and latency percentiles use successful measured requests only.",
        "- KV-cache values are server telemetry; GPU memory values are local `nvidia-smi` telemetry.",
        "- Missing telemetry or incomplete server usage is reported as `n/a`; it is not estimated.",
        "",
    ])
    return "\n".join(lines)


def write_report(output, summaries: list[dict[str, Any]], metadata: dict[str, Any]) -> dict:
    report = build_report(summaries, metadata)
    output.joinpath("report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
    output.joinpath("report.md").write_text(render_markdown(report), encoding="utf-8")
    return report
