import json

import pytest
from pydantic import ValidationError

from contextbench.context import make_strategy
from contextbench.evaluation.scoring import evaluate
from contextbench.models import Decision


def test_full_prefix_stability(dataset):
    full = make_strategy("full_context", dataset)
    cached = make_strategy("prefix_cached_full_context", dataset)
    assert full.build({"city": "Austin"}).text == full.build({"city": "Boston"}).text
    assert full.build({"city": "Austin"}).prompt != full.build({"city": "Boston"}).prompt
    assert full.build({}) == cached.build({})
    assert len(full.build({}).source_ids) == 4


def test_compilation_is_source_only_and_selects_exception(dataset):
    (dataset / "expected.json").unlink()
    strategy = make_strategy("compiled_context", dataset)
    ordinary = strategy.build({"city": "Austin"})
    emergency = strategy.build({"city": "Austin", "medical_emergency": True})
    assert "R_MEDICAL" not in ordinary.source_ids
    assert "R_MEDICAL" in emergency.source_ids
    assert "L_AUSTIN" in ordinary.source_ids
    assert "L_BOSTON" not in ordinary.source_ids
    assert len(ordinary.text) < len(make_strategy("full_context", dataset).build({}).text)


def test_retrieval_deterministic_and_bounded(dataset):
    strategy = make_strategy("retrieval", dataset, 2)
    event = {"city": "Seattle", "category": "travel"}
    assert strategy.build(event) == strategy.build(event)
    assert len(strategy.build(event).source_ids) == 2
    assert "L_SEATTLE" in strategy.build(event).text


def test_evaluation_catches_rule_miss_and_wrong_decision(dataset):
    truth = json.loads((dataset / "expected.json").read_text())["EV-002"]
    output = truth["output"]
    assert evaluate(json.dumps(output), output, truth["critical_rules"])["correct"]
    wrong = {**output, "applied_rules": ["R_BASE", "L_AUSTIN"]}
    score = evaluate(json.dumps(wrong), output, truth["critical_rules"])
    assert score["schema_valid"] and not score["correct"]
    assert score["missed_critical_rules"] == ["R_MEDICAL"]
    assert not evaluate("not json", output, truth["critical_rules"])["schema_valid"]
    assert not evaluate(json.dumps({**output, "decision": "reject"}), output, [])["correct"]

    def test_quality_evaluation_distinguishes_required_and_incorrect_rules(dataset):
        output = json.loads((dataset / "expected.json").read_text())["EV-002"]["output"]
        truth = {"output": output, "required_source_ids": output["applied_rules"]}
        extra = {**output, "applied_rules": [*output["applied_rules"], "R_BASE"]}
        score = evaluate(json.dumps(extra), truth, ["R_MEDICAL"])
        assert score["schema_valid"] and score["answer_correct"]
        assert score["required_rule_coverage"] == 1.0
        assert score["incorrect_rule_inclusion"] == ["R_BASE"]
        assert not score["correct"]

        abstain = {**output, "decision": "abstain", "limit_usd": None}
        abstain_score = evaluate(json.dumps(abstain),
                                 {"output": abstain, "required_source_ids": ["R_UNKNOWN"]}, [])
        assert abstain_score["abstention_correct"]


@pytest.mark.parametrize("patch", [{"limit_usd": "200"}, {"limit_usd": -1},
                                  {"decision": "maybe"}, {"extra": 1}, {"reason": ""}])
def test_schema_strict(dataset, patch):
    output = json.loads((dataset / "expected.json").read_text())["EV-000"]["output"]
    with pytest.raises(ValidationError):
        Decision.model_validate({**output, **patch})
