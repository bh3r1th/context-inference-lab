"""Extract publishable findings only from validated final analysis artifacts."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=Path("FINAL_ANALYSIS.json"))
    parser.add_argument("--output", type=Path, default=Path("BLOG_FINDINGS.md"))
    args = parser.parse_args()
    if not args.analysis.exists():
        raise SystemExit("BLOG_FINDINGS blocked: FINAL_ANALYSIS.json is required")
    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not rows:
        raise SystemExit("BLOG_FINDINGS blocked: FINAL_ANALYSIS.json has no rows")
    findings = []
    for row in rows[:8]:
        findings.append("- **Measured:** " + "; ".join(
            f"{key}={row.get(key)}" for key in ("strategy", "context tokens", "concurrency", "accuracy",
                                                   "critical misses", "TTFT p95", "throughput", "KV cache",
                                                   "GPU memory", "batch duration")) +
                        ". Baseline: full_context at matching concurrency where available. "
                        "Reference: validated FINAL_ANALYSIS row. Limitation: finite synthetic workload; "
                        "this is an observation, not a causal claim.")
    args.output.write_text("# Blog Findings\n\n" + "\n".join(findings) +
                           "\n\nContradictions to the original hypothesis: inspect validated rows above; none are inferred automatically.\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()