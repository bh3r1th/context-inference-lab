"""Audit completed benchmark artifacts for measurement and run-integrity problems."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def request_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def audit(root: Path, findings: list[str], observations: list[str]):
    metadata = load(root / "metadata.json")
    if metadata.get("status") not in {"completed", "completed_with_errors"}:
        findings.append(f"UNTRUSTED run status {metadata.get('status')}: {root}")
    for cell in sorted(path for path in root.iterdir() if path.is_dir() and "-c" in path.name):
        request_path = cell / "requests.jsonl"
        summary_path = cell / "summary.json"
        if not request_path.exists() or not summary_path.exists():
            findings.append(f"INCOMPLETE missing raw artifacts: {cell}")
            continue
        rows = request_rows(request_path)
        summary = load(summary_path)
        failures = [row for row in rows if row.get("error")]
        if failures:
            findings.append(f"FAILURES {len(failures)} requests in {cell}")
        for row in rows:
            text = str(row.get("error", "")).lower()
            if any(term in text for term in ("out of memory", "oom", "cuda out of memory")):
                findings.append(f"OOM request in {cell}/{row.get('request_index')}")
        for field in ("ttft_s", "latency_s", "input_tokens", "output_tokens"):
            missing = sum(row.get(field) is None for row in rows if not row.get("error"))
            if missing:
                findings.append(f"INCOMPLETE {field}: {missing} successful rows missing in {cell}")
        for field in ("ttft_s", "latency_s"):
            values = [row[field] for row in rows if not row.get("error") and row.get(field) is not None]
            if len(values) >= 4 and len(set(values)) == 1:
                findings.append(f"SUSPICIOUS identical {field} values in {cell}")
            if values and (max(values) > 10 * mean(values)):
                observations.append(f"{cell}: high {field} spread max/mean={max(values) / mean(values):.2f}")
        output_lengths = [row["output_tokens"] for row in rows if row.get("output_tokens") is not None]
        if len(output_lengths) > 1 and len(set(output_lengths)) == 1:
            observations.append(f"{cell}: output length is constant at {output_lengths[0]} tokens")
        expected_rate = len([row for row in rows if not row.get("error")]) / summary["batch_duration_s"] \
            if summary.get("batch_duration_s") else None
        if expected_rate is not None and summary.get("requests_per_s") is not None \
                and not math.isclose(expected_rate, summary["requests_per_s"], rel_tol=1e-9):
            findings.append(f"THROUGHPUT mismatch with wall clock: {cell}")
        telemetry = summary.get("telemetry", {})
        if telemetry.get("local_gpu", {}).get("status") != "available":
            observations.append(f"{cell}: GPU telemetry unavailable")
        if telemetry.get("kv_cache", {}).get("status") != "available":
            observations.append(f"{cell}: KV-cache telemetry unavailable")
        if metadata.get("cache_scenario") == "cold" and summary.get("warmup_requests", 0):
            findings.append(f"CACHE CONTAMINATION cold run has warmups: {cell}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("BENCHMARK_SANITY_REPORT.md"))
    args = parser.parse_args()
    findings, observations = [], []
    runs = []
    for root in args.roots:
        candidates = [root] if (root / "metadata.json").exists() else sorted(root.rglob("metadata.json"))
        runs.extend(path.parent for path in candidates)
    if not runs:
        findings.append("NO COMPLETED RUNS: sanity analysis cannot pass without benchmark artifacts")
    for root in runs:
        audit(root, findings, observations)
    repeated = defaultdict(list)
    for root in runs:
        metadata = load(root / "metadata.json")
        repeated[metadata.get("experiment_id")].append(root)
    for experiment_id, roots in repeated.items():
        if len(roots) < 2:
            observations.append(f"{experiment_id}: no repeated run available for variance comparison")
            continue
        values = []
        for root in roots:
            summaries = load(root / "summary.json")
            values.extend(s.get("successful_request_metrics", {}).get("latency_s", {}).get("p95")
                          for s in summaries if s.get("successful_request_metrics", {}).get("latency_s", {}).get("p95") is not None)
        if len(values) > 1:
            observations.append(f"{experiment_id}: p95 latency CV={pstdev(values) / mean(values):.3f}")
    status = "PASS WITH OBSERVATIONS" if not findings else "ISSUES FOUND"
    lines = ["# Benchmark Sanity Report", "", f"**Status: {status}**", "",
             "This report audits completed raw artifacts without deleting, filtering, or rerunning results.",
             "", "## Findings", ""]
    lines += [f"- {item}" for item in findings] or ["- No integrity failures detected."]
    lines += ["", "## Observations", ""]
    lines += [f"- {item}" for item in observations] or ["- No observations recorded."]
    lines += ["", "GPU throttling cannot be established when clock/throttle telemetry is absent; "
              "missing telemetry is reported rather than inferred.", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    raise SystemExit(0 if not findings else 1)


if __name__ == "__main__":
    main()