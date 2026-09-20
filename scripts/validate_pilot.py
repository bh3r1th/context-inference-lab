"""Validate completed pilot artifacts and write PILOT_VALIDATION.md."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from contextbench.evaluation.scoring import evaluate
from contextbench.metrics.summary import percentiles
from contextbench.models import Decision


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def overlap_count(records):
    intervals = []
    for row in records:
        if row.get("started_at") and row.get("completed_at"):
            intervals.append((datetime.fromisoformat(row["started_at"]),
                              datetime.fromisoformat(row["completed_at"])))
    return max((sum(start <= point < end for start, end in intervals) for point in
                [item for interval in intervals for item in interval]), default=0)


def validate_run(root: Path, issues: list[str], observations: list[str]):
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        issues.append(f"missing metadata: {root}")
        return
    if not root.exists():
        issues.append(f"missing pilot artifact root: {root}")
        return
    metadata = read_json(metadata_path)
    if metadata.get("status") not in {"completed", "completed_with_errors"}:
        issues.append(f"run did not complete: {root} status={metadata.get('status')}")
    dataset = root / "dataset"
    expected = read_json(dataset / "expected.json")
    workload = read_json(root / "workload_manifest.json")
    expected_plan = [(item["event_id"], item["seed"], item["repetition"]) for item in workload]
    cell_paths = sorted(path for path in root.iterdir() if path.is_dir() and "-c" in path.name)
    by_concurrency = {}
    for cell in cell_paths:
        request_path = cell / "requests.jsonl"
        summary_path = cell / "summary.json"
        if not request_path.exists() or not summary_path.exists():
            issues.append(f"missing raw request/summary artifact: {cell}")
            continue
        records = rows(request_path)
        summary = read_json(summary_path)
        by_concurrency.setdefault(summary["concurrency"], []).append(records)
        if summary.get("requests") != len(records):
            issues.append(f"summary/raw request count mismatch: {cell}")
        if len(records) != len(expected_plan):
            issues.append(f"unattempted requests in completed pilot cell: {cell}")
        actual_plan = sorted((row["event_id"], row["seed"], row["repetition"]) for row in records)
        if actual_plan != sorted(expected_plan):
            issues.append(f"event/seed/repetition plan mismatch: {cell}")
        for row in records:
            truth = expected[row["event_id"]]
            if row.get("error") is None:
                try:
                    Decision.model_validate_json(row["text"])
                except Exception as exc:
                    issues.append(f"structured output parse failure {cell}/{row['request_index']}: {exc}")
                scored = evaluate(row["text"], truth["output"], truth["critical_rules"])
                if scored["correct"] != row.get("correct"):
                    issues.append(f"ground-truth evaluation mismatch: {cell}/{row['request_index']}")
            for field in ("ttft_s", "latency_s", "input_tokens", "output_tokens"):
                if row.get(field) is not None and row[field] < 0:
                    issues.append(f"negative {field}: {cell}/{row['request_index']}")
            if row.get("latency_s") is not None and row.get("ttft_s") is not None \
                    and row["ttft_s"] > row["latency_s"]:
                issues.append(f"TTFT exceeds latency: {cell}/{row['request_index']}")
        successful = [row for row in records if row.get("error") is None]
        for field in ("ttft_s", "latency_s"):
            observed = [row[field] for row in successful if row.get(field) is not None]
            actual = summary.get("successful_request_metrics", {}).get(field, {})
            if actual.get("count") != len(observed) or actual.get("p95") != percentiles(observed).get("p95"):
                issues.append(f"aggregate/raw {field} mismatch: {cell}")
        telemetry = summary.get("telemetry", {})
        observations.append(f"{cell}: GPU={telemetry.get('local_gpu', {}).get('status')}, "
                            f"KV={telemetry.get('kv_cache', {}).get('status')}, "
                            f"max-overlap={overlap_count(records)}")
    for concurrency, groups in by_concurrency.items():
        if len(groups) > 1:
            reference = sorted((row["event_id"], row["seed"], row["repetition"]) for row in groups[0])
            for records in groups[1:]:
                if sorted((row["event_id"], row["seed"], row["repetition"]) for row in records) != reference:
                    issues.append(f"strategies received different events at concurrency {concurrency}: {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/pilot/PILOT_VALIDATION.md"))
    args = parser.parse_args()
    issues, observations = [], []
    for root in args.roots:
        if not root.exists():
            issues.append(f"missing pilot artifact root: {root}")
            continue
        for run_root in ([root] if (root / "metadata.json").exists() else
                         sorted(root.glob("*/metadata.json"))):
            validate_run(run_root.parent if run_root.name == "metadata.json" else run_root,
                         issues, observations)
    if not issues and not observations:
        issues.append("no pilot run artifacts found")
    status = "PASS" if not issues else "BLOCKED" if any("missing" in i for i in issues) else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Pilot Validation", "", f"**Status: {status}**", "", "## Checks", "",
             "The validator re-parses structured outputs, re-evaluates against copied ground truth, "
             "compares raw rows with summaries, checks timing/token plausibility, and compares event plans.", "",
             "## Observations", ""]
    lines += [f"- {item}" for item in observations] or ["- No completed pilot observations were available."]
    lines += ["", "## Issues and fixes", ""]
    lines += [f"- {item}" for item in issues] or ["- No issues found."]
    if status == "BLOCKED":
        lines += ["", "The pilot did not run; full benchmark execution is prohibited until this report is PASS."]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output.parent / "pilot_validation.json").write_text(
        json.dumps({"status": status, "issues": issues, "observations": observations}, indent=2) + "\n",
        encoding="utf-8")
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()