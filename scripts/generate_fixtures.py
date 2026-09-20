"""Generate public-safe, deterministic sources and independently specified answer keys."""
import argparse
import json
from pathlib import Path


def generate(root: Path, paragraphs: int = 80):
    docs = root / "documents"
    docs.mkdir(parents=True, exist_ok=True)
    rules = [
        ("policy.md", "R_BASE", {}, "For travel expenses, approve amounts at or below the city "
         "limit and reject amounts above it. Cite R_BASE and the matching city lookup ID."),
        ("policy.md", "R_UNKNOWN", {}, "If city has no lookup, or amount_usd or category is "
         "missing, abstain with null limit_usd. Cite only R_UNKNOWN in that case."),
        ("exceptions.md", "R_MEDICAL", {"medical_emergency": True}, "A medical emergency "
         "overrides the city travel limit to 1000 USD. Cite R_MEDICAL, R_BASE, and city ID."),
        ("controls.md", "R_RESTRICTED", {"restricted_vendor": True}, "Restricted vendors "
         "must be rejected regardless of amount or emergency. Retain the applicable limit "
         "and cite R_RESTRICTED along with other applicable rules."),
        ("lookup.md", "L_AUSTIN", {"city": "Austin"}, "Austin travel limit is 200 USD."),
        ("lookup.md", "L_BOSTON", {"city": "Boston"}, "Boston travel limit is 300 USD."),
        ("lookup.md", "L_SEATTLE", {"city": "Seattle"}, "Seattle travel limit is 250 USD."),
    ]
    grouped = {}
    for filename, rule_id, when, text in rules:
        rule = {"id": rule_id, "when": when, "fact": text}
        grouped.setdefault(filename, []).append("RULE " + json.dumps(rule, sort_keys=True))
    for filename, source_rules in grouped.items():
        filler = [f"Reference note {i:04d} for {filename}: Archive records use department "
                  f"code D{i % 37:02d}. Historical inventory labels describe office supplies "
                  "and document retention examples. These reference notes impose no expense "
                  "approval conditions and do not override policy rules."
                  for i in range(paragraphs)]
        midpoint = len(filler) // 2
        content = [f"# Synthetic enterprise reference: {filename}", *filler[:midpoint],
                   *source_rules, *filler[midpoint:]]
        (docs / filename).write_text("\n\n".join(content) + "\n", encoding="utf-8")
    cases = [
        ("normal", {"city": "Austin", "amount_usd": 150}, "approve", 200,
         ["R_BASE", "L_AUSTIN"], ["R_BASE"]),
        ("cross-document", {"city": "Boston", "amount_usd": 100, "restricted_vendor": True},
         "reject", 300, ["R_BASE", "L_BOSTON", "R_RESTRICTED"], ["R_RESTRICTED"]),
        ("buried-exception", {"city": "Austin", "amount_usd": 800, "medical_emergency": True},
         "approve", 1000, ["R_BASE", "L_AUSTIN", "R_MEDICAL"], ["R_MEDICAL"]),
        ("lookup-dependent", {"city": "Seattle", "amount_usd": 275}, "reject", 250,
         ["R_BASE", "L_SEATTLE"], ["L_SEATTLE"]),
        ("ambiguous-abstain", {"city": "Unknown", "amount_usd": 75}, "abstain", None,
         ["R_UNKNOWN"], ["R_UNKNOWN"]),
    ]
    events, expected = [], {}
    for i, (case, details, decision, limit, applied, critical) in enumerate(cases):
        event_id = f"EV-{i:03d}"
        events.append({"event_id": event_id, "category": "travel", **details})
        expected[event_id] = {"case": case, "critical_rules": critical, "output": {
            "event_id": event_id, "decision": decision, "limit_usd": limit,
            "applied_rules": applied, "reason": "Synthetic reference answer."}}
    (root / "events.json").write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")
    (root / "expected.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--paragraphs", type=int, default=80)
    args = parser.parse_args()
    if args.paragraphs < 0:
        parser.error("paragraphs must be nonnegative")
    generate(args.output, args.paragraphs)
