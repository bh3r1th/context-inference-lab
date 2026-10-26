import asyncio
import json

import pytest

from contextbench.context import make_strategy
from contextbench.evaluation.scoring import evaluate_context_coverage
from contextbench.inference.runner import run_batch
from contextbench.metrics.summary import summarize
from contextbench.models import Context, InferenceResult


@pytest.mark.parametrize("kind,ids", [
    ("document", ("policy.md",)), ("chunk", ("policy.md:0-100",)),
    ("rule", ("R_BASE", "L_AUSTIN")),
])
def test_rule_presence_is_independent_of_source_id_namespace(kind, ids):
    context = Context("rules", ids, metadata={
        "source_id_kind": kind, "selected_rule_ids": ["R_BASE", "L_AUSTIN"]})
    result = evaluate_context_coverage(context, {"applied_rules": ["R_BASE", "L_AUSTIN"]})
    assert result["required_context_covered"] is True
    assert result["missing_required_context_sources"] == []


def test_explicit_required_rules_and_partial_retrieval_are_respected():
    context = Context("partial", ("policy.md:0-100",), metadata={
        "source_id_kind": "chunk", "selected_rule_ids": ["R_BASE"]})
    result = evaluate_context_coverage(context, {
        "output": {"applied_rules": ["R_BASE"]},
        "required_source_ids": ["R_BASE", "R_EXCEPTION"]})
    assert result["required_context_covered"] is False
    assert result["missing_required_context_sources"] == ["R_EXCEPTION"]


def test_unknown_source_ids_are_not_treated_as_rule_evidence():
    truth = {"applied_rules": ["R_BASE"]}
    unknown = evaluate_context_coverage(Context("rules", ("R_BASE",)), truth)
    assert unknown["required_context_covered"] is None
    assert unknown["missing_required_context_sources"] is None
    explicit = evaluate_context_coverage(Context(
        "rules", ("R_BASE",), metadata={"source_id_kind": "rule"}), truth)
    assert explicit["required_context_covered"] is True
    assert evaluate_context_coverage(None, truth)["required_context_covered"] is False


def test_full_context_runner_reports_presence_even_when_answer_is_wrong(dataset):
    event = json.loads((dataset / "events.json").read_text())[0]
    expected = json.loads((dataset / "expected.json").read_text())

    class Backend:
        async def generate(self, context, event, seed):
            output = expected[event["event_id"]]["output"]
            return InferenceResult(text=json.dumps({**output, "decision": "reject"}))

    records, duration = asyncio.run(run_batch(
        Backend(), make_strategy("full_context", dataset), [event], expected, 1, 42,
        lambda _: None))
    assert records[0]["required_context_covered"] is True
    assert records[0]["minimum_safe_context"] is True
    assert records[0]["correct"] is False
    assert summarize(records, duration)["quality"]["required_context_covered"] == 1
