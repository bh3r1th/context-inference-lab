"""Reproduce the bounded pilot review from explicitly selected, immutable artifacts.

Writes derived files only. No GPU, inference, cloud writes or raw-artifact edits.
"""
import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from contextbench.evaluation.scoring import evaluate, evaluate_context_coverage
from contextbench.experiments.config import load_config
from contextbench.models import Context
from contextbench.workload import answer
from deploy.gcp.task import verify_results
from scripts.validate_pilot import validate_run

REPO = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_path(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"Input path escapes artifact root: {relative}")
    return path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)


def reviewed_request(record, truth, cell, run_id, execution, event=None):
    score = evaluate(record["text"], truth["output"], truth["critical_rules"])
    require(not record["error"], f"Selected request failed: {cell}/{record['event_id']}")
    require(all(record.get(key) == value for key, value in score.items()),
            f"Archived scoring differs from independent evaluation: {cell}/{record['event_id']}")
    context = Context("", tuple(record["source_ids"]), metadata=record["context_metadata"])
    coverage = evaluate_context_coverage(context, truth)
    return {
        "cell": cell, "run_id": run_id, "execution": execution,
        **{key: record[key] for key in ("request_id", "request_index", "event_id", "seed",
            "repetition", "case", "http_status", "finish_reason", "input_tokens", "output_tokens",
            "latency_s", "ttft_s", "text", "error")},
        "event": event, "expected_output": truth["output"], **score,
        "selected_rule_ids": record["context_metadata"]["selected_rule_ids"],
        "required_rule_ids": truth.get("required_source_ids", truth["output"]["applied_rules"]),
        "original_minimum_safe_context": record.get("minimum_safe_context"),
        "original_missing_required_context_sources": record.get("missing_required_context_sources"),
        **coverage,
    }


def export(artifacts, selection_path, output, verify_manifest=None):
    require(not output.resolve().is_relative_to(artifacts.resolve()),
            "Derived output must be outside the raw artifact root")
    selection = read(selection_path)
    chosen = selection["selected_attempts"]
    require(len(chosen) == selection["expected_cells"], "Selected cell count mismatch")
    require(len({entry["cell"] for entry in chosen}) == len(chosen), "Duplicate selected cells")
    source_ref = selection["compatible_client_reference_commit"]
    source_hashes = {}
    inventory, configs, cells, records, findings, observations = {}, {}, [], [], [], []
    excluded = []
    all_attempts = [*chosen, *selection["retained_failed_attempts"]]
    for entry in all_attempts:
        attempt = input_path(artifacts, entry["attempt"])
        for path in sorted(attempt.rglob("*")):
            if path.is_file():
                require(path.resolve().is_relative_to(artifacts.resolve()), "Symlink escapes inputs")
                inventory[path.relative_to(artifacts).as_posix()] = {
                    "bytes": path.stat().st_size, "sha256": sha256(path)}
    if verify_manifest:
        require(inventory == read(verify_manifest)["input_files"],
                "Input artifacts differ from frozen manifest")
    for entry in selection["retained_failed_attempts"]:
        attempt = input_path(artifacts, entry["attempt"])
        run = next((attempt / "runs").iterdir())
        summary = read(run / "summary.json")[0]
        excluded.append({**entry, "status": read(attempt / "task-status.json"),
                         "successful_requests": summary["successful_requests"],
                         "failed_requests": summary["failed_requests"]})
    for entry in chosen:
        cell = entry["cell"]
        attempt = input_path(artifacts, entry["attempt"])
        run = attempt / "runs" / entry["run_id"]
        metadata, summary = read(run / "metadata.json"), read(run / "summary.json")[0]
        config = load_config(attempt / "config.yaml")
        require(config.model_dump() == metadata["config"], f"Recorded configs differ: {cell}")
        completion = read(attempt / "complete.json")
        task_status = read(attempt / "task-status.json")
        require(completion["cell"] == task_status["cell"] == cell,
                f"Selection does not match artifact cell identity: {cell}")
        config.dataset, config.output_dir = str(run / "dataset"), str(attempt / "runs")
        verify_results(config, require_correct=False)
        require(read(attempt / "complete.json")["status"] == "completed", "Missing completion")
        validate_run(run, findings, observations)
        for relative, digest in metadata["dataset_sha256"].items():
            require(sha256(input_path(run / "dataset", relative)) == digest,
                    f"Dataset checksum mismatch: {cell}/{relative}")
        for relative, digest in metadata["client_source_sha256"].items():
            if relative not in source_hashes:
                blob = subprocess.check_output(
                    ["git", "show", f"{source_ref}:src/contextbench/{relative}"], cwd=REPO)
                source_hashes[relative] = hashlib.sha256(blob).hexdigest()
            require(source_hashes[relative] == digest, f"Client checksum mismatch: {relative}")
        expected = read(run / "dataset/expected.json")
        cities = {row["city"]: row for row in read(run / "dataset/lookup.json")}
        events = read(run / "dataset/events.json")
        event_map = {event["event_id"]: event for event in events}
        for event in events:
            truth = expected[event["event_id"]]["output"]
            require(answer(event, cities) == (
                truth["decision"], truth["limit_usd"], truth["applied_rules"]),
                f"Ground truth differs from independent dataset oracle: {cell}/{event['event_id']}")
        cell_path = run / f"{config.strategies[0]}-c{config.concurrency[0]}"
        raw = [json.loads(line) for line in (cell_path / "requests.jsonl").read_text().splitlines()]
        warmups = [json.loads(line) for line in
                   (cell_path / "warmup.jsonl").read_text().splitlines()]
        require(len(warmups) == summary["warmup_requests"] and
                all(not row["error"] for row in warmups), f"Warmup evidence differs: {cell}")
        require(len(raw) == selection["expected_measured_requests_per_cell"], "Request count mismatch")
        derived = [reviewed_request(row, expected[row["event_id"]], cell, run.name,
                                    Path(entry["attempt"]).parts[0], event_map[row["event_id"]])
                   for row in raw]
        records.extend(derived)
        gpu, kv = [], []
        with (cell_path / "telemetry.jsonl").open() as handle:
            for line in handle:
                sample = json.loads(line)
                gpu.extend(device["memory_used_mib"] for device in
                           sample.get("gpu", {}).get("devices") or []
                           if device.get("memory_used_mib") is not None)
                kv.extend(item["value"] for item in sample.get("kv_cache", {}).get("samples") or []
                          if item.get("value") is not None)
        intervals = summary["telemetry"]["prefix_cache"].get("counter_intervals", [])
        ratios = [item.get("hit_ratio") for item in intervals
                  if item["metric"] == "vllm:prefix_cache_hits_total"]
        row = {
            "cell": cell, "execution": Path(entry["attempt"]).parts[0], "run_id": run.name,
            "strategy": config.strategies[0], "concurrency": config.concurrency[0],
            "request_timeout_s": config.server.timeout_s,
            "requests": len(raw), "successful_requests": len(raw),
            "strict_correct": sum(r["correct"] for r in derived),
            "decision_correct": sum(r["answer_correct"] for r in derived),
            "schema_valid": sum(r["schema_valid"] for r in derived),
            "required_context_covered": sum(r["required_context_covered"] for r in derived),
            "original_coverage_true": sum(bool(r["original_minimum_safe_context"]) for r in derived),
            "dataset_events": len(events), "repetitions": config.repetitions,
            "warmup_requests": summary["warmup_requests"], "warmup_failures": summary["warmup_failures"],
            "task_s": read(attempt / "complete.json")["elapsed_s"],
            "measured_batch_s": summary["batch_duration_s"],
            "mean_input_tokens": summary["average_input_tokens"],
            "mean_output_tokens": summary["average_output_tokens"],
            "latency_p95_s": summary["successful_request_metrics"]["latency_s"]["p95"],
            "ttft_p95_s": summary["successful_request_metrics"]["ttft_s"]["p95"],
            "gpu_memory_peak_mib": max(gpu) if gpu else None,
            "kv_usage_peak_fraction": max(kv) if kv else None,
            "server_prefix_counter_hit_ratio": ratios[0] if ratios else None,
        }
        cells.append(row)
        configs[cell] = metadata["config"]
    require(not findings, f"Integrity failures: {findings}")
    for family in ("pilot-enterprise", "pilot-scaling", "pilot-prefix"):
        groups = {cell["cell"] for cell in cells if cell["cell"].startswith(family)}
        plans = [sorted((r["event_id"], r["seed"], r["repetition"]) for r in records
                        if r["cell"] == cell) for cell in sorted(groups)]
        require(all(plan == plans[0] for plan in plans), f"Cross-cell event plan differs: {family}")
    summary = {
        "study_type": "pilot", "integrity_status": "PASS", "full_benchmark_completed": False,
        "cells": len(cells), "measured_requests": len(records),
        "unique_event_ids": len({row["event_id"] for row in records}),
        "strict_correct": sum(row["correct"] for row in records),
        "decision_correct": sum(row["answer_correct"] for row in records),
        "schema_valid": sum(row["schema_valid"] for row in records),
        "required_context_covered": sum(row["required_context_covered"] for row in records),
        "original_coverage_true": sum(bool(row["original_minimum_safe_context"]) for row in records),
        "selected_task_s": sum(row["task_s"] for row in cells),
        "measured_batch_s": sum(row["measured_batch_s"] for row in cells),
        "decision_wrong": sum(not row["answer_correct"] for row in records),
        "decision_correct_citations_wrong": sum(row["answer_correct"] and not row["correct"]
                                               for row in records),
        "context_missing_required_rules": sum(not row["required_context_covered"] for row in records),
        "case_counts": dict(Counter(row["case"] for row in records)),
        "retained_failed_attempts": excluded,
        "review_note": "Integrity PASS is not a full-run feasibility or answer-quality gate.",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_json(output / "evidence_manifest.json", {
        "schema_version": 1, "selection_sha256": sha256(selection_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "coverage_module_sha256": sha256(REPO / "src/contextbench/evaluation/scoring.py"),
        "selection": selection, "input_files": inventory,
        "recorded_run_configs": configs, "verified_client_source_hashes": source_hashes,
        "validation_observations": observations,
    })
    write_csv(output / "cells.csv", cells)
    with (output / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=REPO / "docs/pilot-study/selection.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.artifacts, args.selection, args.output, args.verify_manifest), indent=2))


if __name__ == "__main__":
    main()
