"""Keep published pilot counts tied to the frozen, independently rescored evidence."""
import json
from pathlib import Path

import pytest

from contextbench.evaluation.scoring import evaluate
from scripts.finalize_cloud_pilot import input_path, reviewed_request

RESULTS = Path(__file__).resolve().parents[1] / "docs/pilot-study/results"


def published_records():
    return [json.loads(line) for line in (RESULTS / "requests.jsonl").read_text().splitlines()]


def test_published_evidence_supports_counts_and_preserves_original_coverage():
    rows = published_records()
    summary = json.loads((RESULTS / "summary.json").read_text())
    scores = [evaluate(row["text"], row["expected_output"], []) for row in rows]
    assert len(rows) == summary["measured_requests"] == 180
    assert len({row["cell"] for row in rows}) == summary["cells"] == 18
    assert len({row["event_id"] for row in rows}) == summary["unique_event_ids"] == 10
    assert sum(score["correct"] for score in scores) == summary["strict_correct"] == 36
    assert sum(score["answer_correct"] for score in scores) == summary["decision_correct"] == 63
    covered = sum(set(row["required_rule_ids"]) <= set(row["selected_rule_ids"]) for row in rows)
    assert covered == summary["required_context_covered"] == 162
    assert sum(row["original_minimum_safe_context"] for row in rows) == 20
    assert summary["full_benchmark_completed"] is False


def test_retroactive_export_rejects_changed_answer_scores_and_only_corrects_coverage():
    row = next(row for row in published_records() if row["correct"] and
               not row["original_minimum_safe_context"])
    truth = {"output": row["expected_output"], "critical_rules": row["required_rule_ids"]}
    raw = {**row, "source_ids": ["policy.md"],
           "context_metadata": {"selected_rule_ids": row["selected_rule_ids"]},
           "minimum_safe_context": False,
           "missing_required_context_sources": row["original_missing_required_context_sources"]}
    result = reviewed_request(raw, truth, row["cell"], row["run_id"], row["execution"])
    assert result["original_minimum_safe_context"] is False
    assert result["required_context_covered"] is True
    assert result["correct"] == raw["correct"]
    with pytest.raises(ValueError, match="scoring differs"):
        reviewed_request({**raw, "correct": False}, truth, row["cell"], row["run_id"], row["execution"])


def test_manifest_paths_cannot_escape_artifact_root(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        input_path(tmp_path, "../outside")
