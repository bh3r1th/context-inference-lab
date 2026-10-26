from pydantic import ValidationError

from contextbench.models import Context, Decision


def evaluate_context_coverage(context: Context | None, expected: dict) -> dict:
    """Compare rule IDs across strategies; source IDs may name documents or chunks.

    This measures required-rule presence, not answer correctness or minimum context
    size. Keep minimum_safe_context as a deprecated compatibility alias only.
    """
    output = expected.get("output", expected)
    required = set(expected.get("required_source_ids", output["applied_rules"]))
    selected = [] if context is None else context.metadata.get("selected_rule_ids")
    source = "no context" if context is None else "context_metadata.selected_rule_ids"
    if selected is None and context.metadata.get("source_id_kind") == "rule":
        selected, source = context.source_ids, "source_ids (explicit rule IDs)"
    available = isinstance(selected, (list, tuple)) and all(
        isinstance(rule_id, str) for rule_id in selected)
    missing = sorted(required - set(selected)) if available else None
    covered = not missing if available else None
    return {
        "required_context_covered": covered,
        "missing_required_context_sources": missing,
        "context_coverage_source": source if available else None,
        "context_coverage_reason": None if available else "rule-ID coverage evidence unavailable",
        "minimum_safe_context": covered,
    }


def evaluate(text: str, expected: dict, critical_rules: list[str]) -> dict:
    expected_output = expected.get("output", expected)
    required_rules = set(expected.get("required_source_ids", expected_output["applied_rules"]))
    critical = set(critical_rules)
    expected_decision = expected_output["decision"]
    try:
        answer = Decision.model_validate_json(text)
    except ValidationError as exc:
        return _score_failure(required_rules, critical, expected_decision, str(exc))
    cited = set(answer.applied_rules)
    missed = required_rules - cited
    incorrect = cited - required_rules
    answer_correct = all(getattr(answer, key) == expected_output[key]
                         for key in ("event_id", "decision", "limit_usd"))
    abstention_correct = (answer.decision == expected_decision == "abstain"
                          and answer.limit_usd is None)
    return {
        "schema_valid": True,
        "correct": answer_correct and not missed and not incorrect,
        "answer_correct": answer_correct,
        "required_rule_coverage": len(cited & required_rules) / len(required_rules)
        if required_rules else 1.0,
        "rule_recall": len(cited & required_rules) / len(required_rules)
        if required_rules else 1.0,
        "missed_critical_rules": sorted(critical - cited),
        "missed_required_rules": sorted(missed),
        "incorrect_rule_inclusion": sorted(incorrect),
        "abstention_correct": abstention_correct,
        "validation_error": None,
    }


def _score_failure(required_rules: set[str], critical: set[str], expected_decision: str,
                   validation_error: str) -> dict:
    return {
        "schema_valid": False,
        "correct": False,
        "answer_correct": False,
        "required_rule_coverage": 0.0,
        "rule_recall": 0.0,
        "missed_critical_rules": sorted(critical),
        "missed_required_rules": sorted(required_rules),
        "incorrect_rule_inclusion": [],
        "abstention_correct": False if expected_decision == "abstain" else None,
        "validation_error": validation_error,
    }
