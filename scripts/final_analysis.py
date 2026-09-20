"""Generate FINAL_ANALYSIS.md from validated benchmark summaries."""
import argparse
import json
from pathlib import Path

from analyze_results import load_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=Path("results"))
    parser.add_argument("--sanity", type=Path, default=Path("BENCHMARK_SANITY_REPORT.md"))
    parser.add_argument("--output", type=Path, default=Path("FINAL_ANALYSIS.md"))
    args = parser.parse_args()
    if not args.sanity.exists() or "**Status: PASS WITH OBSERVATIONS**" not in args.sanity.read_text(encoding="utf-8"):
        raise SystemExit("FINAL_ANALYSIS blocked: validated BENCHMARK_SANITY_REPORT.md is required")
    rows = load_rows(args.results)
    if not rows:
        raise SystemExit("FINAL_ANALYSIS blocked: no completed summaries found")
    rows = sorted(rows, key=lambda row: (row.get("context tokens") or 0, row.get("strategy") or ""))
    payload = {"rows": rows, "measured": [
        "context tokens, TTFT p95, latency p95, throughput, KV cache, GPU memory, and batch duration",
        "accuracy and critical-rule miss rate from the shared expected answers",
    ], "interpretation_policy": "correlation only; no automatic causal claims"}
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Final Analysis", "", "## Measured facts", "",
             "The following are directly observed or aggregated from validated request and telemetry artifacts.", ""]
    for row in rows:
        lines.append("- " + "; ".join(f"{key}={row.get(key)}" for key in
                     ("strategy", "context tokens", "concurrency", "accuracy", "critical misses",
                      "TTFT p95", "latency p95", "throughput", "KV cache", "GPU memory", "batch duration")))
    lines += ["", "## Derived comparisons", "",
              "Context reduction is relative to full_context at matching concurrency. The smallest observed "
              "configuration meeting quality constraints must be selected from these validated rows; no "
              "unobserved interpolation is performed.", "", "## Likely explanations", "",
              "Shorter prompts can reduce prefill work and KV-cache demand, but this is an interpretation "
              "of the controlled comparison, not proof of mechanism.", "", "## Unsupported hypotheses", "",
              "No claim is made that a measured latency difference is caused solely by context length; "
              "scheduling, cache state, hardware, and server load remain possible contributors.", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()