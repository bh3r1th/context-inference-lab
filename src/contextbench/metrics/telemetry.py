"""Raw Prometheus/GPU sampling and explicitly server-scoped cache observations."""
import asyncio
import math
import re

import httpx

from contextbench.metrics.gpu import sample_gpu
from contextbench.metrics.provenance import unavailable, utc_now
from contextbench.metrics.summary import percentiles

KV_NAMES = {"vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"}


def prometheus_samples(text: str) -> list[dict]:
    result = []
    for line in text.splitlines():
        match = re.fullmatch(r'(vllm:[a-zA-Z0-9_:]+)(\{.*\})?\s+([^\s]+)(?:\s+\S+)?', line.strip())
        if not match:
            continue
        reason = None
        try:
            value = float(match[3])
            if not math.isfinite(value):
                raise ValueError("nonfinite value")
        except ValueError:
            value, reason = None, f"nonfinite or invalid Prometheus value {match[3]!r}"
        result.append({"metric": match[1], "labels": match[2] or "", "value": value,
                       "reason": reason, "source": "Prometheus /metrics", "scope": "server"})
    return result


def kv_samples(text: str) -> list[dict]:
    return [s for s in prometheus_samples(text) if s["metric"] in KV_NAMES and s["value"] is not None]


def group(samples: list[dict], source: str, reason: str = "metric not exposed") -> dict:
    available = any(s["value"] is not None for s in samples)
    return {"status": "available" if available else "unavailable", "samples": samples or None,
            "source": source, "scope": "server", "reason": None if available else reason}


async def sample_telemetry(client: httpx.AsyncClient, url: str | None,
                           headers: dict, gpu: bool) -> dict:
    sample = {"timestamp": utc_now(), "phase": "periodic"}
    samples, reason = [], "metrics_url is disabled"
    if url:
        try:
            response = await client.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            sample["server_metrics"] = {"status": "available", "scope": "server", "source": url,
                                         "reason": None, "prometheus_text": response.text}
            samples = prometheus_samples(response.text)
            reason = "metric absent or nonfinite in the server scrape"
        except httpx.HTTPError as exc:
            reason = f"{type(exc).__name__}: {exc}"
    if "server_metrics" not in sample:
        sample["server_metrics"] = {**unavailable(url or "Prometheus /metrics", reason, "server"),
                                     "prometheus_text": None}
    sample["kv_cache"] = group([s for s in samples if s["metric"] in KV_NAMES],
                                url or "Prometheus /metrics", reason)
    sample["prefix_cache"] = group([s for s in samples if "prefix_cache" in s["metric"]],
                                    url or "Prometheus /metrics", reason)
    sample["server_observations"] = group(samples, url or "Prometheus /metrics", reason)
    sample["gpu"] = await sample_gpu() if gpu else {
        **unavailable("nvidia-smi", "local_gpu_metrics is disabled; serving GPUs not observed",
                      "local_host"), "devices": None}
    sample["completed_at"] = utc_now()
    return sample


def prefix_intervals(samples: list[dict]) -> list[dict]:
    """Counter deltas only with matched series at both boundaries and no observed reset."""
    if not samples:
        return []
    snapshots = [{s["metric"] + s["labels"]: s for s in
                  (sample.get("prefix_cache", {}).get("samples") or [])} for sample in samples]
    keys = sorted({key for snapshot in snapshots for key in snapshot})
    intervals = []
    for key in keys:
        item = next(snapshot[key] for snapshot in snapshots if key in snapshot)
        name = item["metric"]
        if not re.fullmatch(r"vllm:(?:external_)?prefix_cache_(?:hits|queries)(?:_total)?", name):
            continue
        values = [snapshot.get(key, {}).get("value") for snapshot in snapshots]
        reason, delta = None, None
        if len(values) < 2 or any(v is None for v in values):
            reason = "requires matching finite counters at every sampled boundary"
        elif any(v < 0 for v in values) or any(b < a for a, b in zip(values, values[1:])):
            reason = "counter reset or negative counter observed; interval cannot be recovered"
        else:
            delta = values[-1] - values[0]
        intervals.append({"metric": name, "labels": item["labels"], "delta": delta,
                          "reason": reason, "scope": "server", "source": "Prometheus counter delta"})
    for item in intervals:
        if "_hits" not in item["metric"]:
            continue
        query_name = item["metric"].replace("_hits", "_queries")
        query = next((q for q in intervals if q["metric"] == query_name
                      and q["labels"] == item["labels"]), None)
        item["hit_ratio"] = None
        item["hit_ratio_reason"] = "requires hits and positive queries for the same series"
        if (item["delta"] is not None and query is not None and query["delta"] is not None
                and query["delta"] > 0 and item["delta"] <= query["delta"]):
            item["hit_ratio"] = item["delta"] / query["delta"]
            item["hit_ratio_reason"] = None
    return intervals


def summarize_telemetry(samples: list[dict]) -> dict:
    kv, devices, prefix = {}, {}, {}
    for sample in samples:
        for group_name, target in (("kv_cache", kv), ("prefix_cache", prefix)):
            for item in sample.get(group_name, {}).get("samples") or []:
                if item["value"] is not None:
                    target.setdefault(item["metric"] + item["labels"], []).append(item["value"])
        for device in sample.get("gpu", {}).get("devices") or []:
            identifier = device.get("uuid") or device.get("pci_bus_id")
            if not identifier:
                continue  # Preserve raw data without merging unidentified devices.
            target = devices.setdefault(identifier, {})
            for field in ("utilization_percent", "memory_used_mib", "memory_total_mib"):
                if device.get(field) is not None:
                    target.setdefault(field, []).append(device[field])
    def summary(series, source, scope):
        return {"status": "available" if series else "unavailable", "source": source, "scope": scope,
                "reason": None if series else "no finite measurements collected; see raw telemetry",
                "series": {k: percentiles(v) for k, v in series.items()} if series else None}
    return {"sample_count": len(samples),
            "kv_cache": summary(kv, "Prometheus gauges", "server"),
            "prefix_cache": {**summary(prefix, "Prometheus prefix-cache series", "server"),
                             "counter_intervals": prefix_intervals(samples) or None},
            "local_gpu": {"status": "available" if devices else "unavailable", "scope": "local_host",
                          "source": "nvidia-smi", "reason": None if devices else
                          "no GPU measurements collected; see raw telemetry",
                          "devices": {d: {k: percentiles(v) for k, v in fields.items()}
                                      for d, fields in devices.items()} if devices else None}}


async def collect(client: httpx.AsyncClient, url: str | None, headers: dict,
                  gpu: bool, interval: float, stop: asyncio.Event, emit):
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            emit(await sample_telemetry(client, url, headers, gpu))
