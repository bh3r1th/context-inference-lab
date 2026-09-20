"""Synthetic expense-control benchmark authoring and validation; no inference calls."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from pydantic import Field

from contextbench.models import Decision, StrictModel


class DatasetConfig(StrictModel):
    seed: int = 1729
    events: int = Field(default=60, ge=12)
    reference_sections: int = Field(default=80, ge=8)
    lookup_rows: int = Field(default=24, ge=4)


CASES = ("normal", "cross-section", "cross-document", "buried-exception",
         "lookup-dependent", "ambiguous-abstain")
RULES = [
    ("R_BASE", "policy.md", "1. Expense eligibility", {},
     "Travel expenses in USD are approved when amount_usd is at or below the applicable "
     "limit, otherwise rejected. All amounts are whole dollars. Cite R_BASE and the city row."),
    ("R_DEPARTMENT", "policy.md", "9. Department budgets", {"department": "FIELD"},
     "FIELD travel uses the lower of the city limit and 150 USD. Cite R_DEPARTMENT as well."),
    ("R_VENDOR", "controls.md", "3. Vendor screening", {"vendor_status": "restricted"},
     "A restricted vendor must be rejected even during an emergency. Retain the applicable "
     "limit and cite R_VENDOR and all rules used to establish that limit."),
    ("R_EMERGENCY", "exceptions.md", "Appendix Q. Incident travel", {"emergency": True},
     "An emergency with an incident_id replaces the city and department limits with 1000 USD. "
     "Cite R_EMERGENCY, R_BASE, the city row, and R_DEPARTMENT if FIELD. Vendor controls still apply."),
    ("R_ABSTAIN", "policy.md", "0. Evidence completeness", {},
     "Before applying any other rule, abstain with null limit and cite only R_ABSTAIN if "
     "city has no lookup row, category is not travel, currency is not USD, amount_usd is missing "
     "or is not a nonnegative integer, department is not OFFICE or FIELD, vendor_status is "
     "not clear or restricted, emergency is missing or not boolean, emergency is true without "
     "incident_id, or receipt_amount_usd is present and differs from amount_usd. "
     "Missing city, category, currency, department, or vendor_status also requires abstention."),
]


def dump(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def answer(event, cities):
    """Dataset-only consistency oracle, independent of scenario answer construction."""
    amount = event.get("amount_usd")
    city = cities.get(event.get("city"))
    invalid = (city is None or event.get("category") != "travel"
               or event.get("currency") != "USD" or type(amount) is not int
               or (type(amount) is int and amount < 0)
               or event.get("department") not in ("OFFICE", "FIELD")
               or event.get("vendor_status") not in ("clear", "restricted")
               or type(event.get("emergency")) is not bool
               or (event.get("emergency") and not event.get("incident_id"))
               or event.get("receipt_amount_usd", amount) != amount)
    if invalid:
        return "abstain", None, ["R_ABSTAIN"]
    limit, ids = city["limit_usd"], ["R_BASE", city["id"]]
    if event["department"] == "FIELD":
        limit = min(limit, 150)
        ids.append("R_DEPARTMENT")
    if event["emergency"]:
        limit = 1000
        ids.append("R_EMERGENCY")
    if event["vendor_status"] == "restricted":
        ids.append("R_VENDOR")
    decision = "reject" if event["vendor_status"] == "restricted" or amount > limit else "approve"
    return decision, limit, ids


def generate(root: Path, config: DatasetConfig):
    rng = random.Random(config.seed)
    root.mkdir(parents=True, exist_ok=True)
    docs = root / "documents"
    docs.mkdir(exist_ok=True)
    cities = [{"id": f"L_CITY_{i:04d}", "city": f"Harbor-{i:04d}",
               "limit_usd": rng.choice([200, 250, 300, 350, 400])}
              for i in range(config.lookup_rows)]
    sources = [{"id": rid, "document": doc, "section": section, "when": when, "fact": fact}
               for rid, doc, section, when, fact in RULES]
    sources += [{"id": row["id"], "document": "lookup.md", "section": row["city"],
                 "when": {"city": row["city"]},
                 "fact": f"{row['city']} travel limit is {row['limit_usd']} USD."}
                for row in cities]
    for filename in ("policy.md", "controls.md", "exceptions.md", "lookup.md"):
        blocks = [f"# Fictional Meridian Services — {filename}",
                  "Synthetic benchmark only. Effective 2026-01-01. All organizations, cities, "
                  "people and records are invented. RULE records are normative; archive notes "
                  "are nonbinding. Evidence completeness precedes all other controls."]
        positions = {}
        for source in (s for s in sources if s["document"] == filename):
            position = (config.reference_sections * 9 // 10 if filename == "exceptions.md"
                        else 1 if source["id"] in ("R_BASE", "R_ABSTAIN")
                        else config.reference_sections * 3 // 4)
            positions.setdefault(position, []).append(source)
        for i in range(config.reference_sections):
            blocks.append(
                f"## Archive reference {i:04d}\n\n"
                f"Department D{i % 37:02d} reconciles batch B{i:05d} using the intake register. "
                "The archived workflow records submission date, cost center, invoice identifier, "
                "and reviewer assignment. A reconciliation packet includes receipt images and "
                "duplicate detection notes. Teams route unresolved coding questions to the service "
                "desk and retain the correction history with the original packet. This historical "
                "description adds no eligibility requirement or monetary limit to the current policy.")
            for source in positions.get(i, []):
                blocks.extend([f"## {source['section']}", "RULE " + json.dumps(source, sort_keys=True)])
        (docs / filename).write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    records, events, expected = [], [], {}
    for i in range(config.events):
        case, variant = CASES[i % len(CASES)], i // len(CASES)
        city = rng.choice(cities)
        limit = city["limit_usd"]
        event = {"event_id": f"EV-{i:06d}", "category": "travel", "currency": "USD",
                 "city": city["city"], "amount_usd": limit, "department": "OFFICE",
                 "vendor_status": "clear", "emergency": False,
                 "employee_id": f"EMP-{rng.randrange(10000):05d}",
                 "invoice_id": f"INV-{i:07d}"}
        ids, decision, reason = ["R_BASE", city["id"]], "approve", "At the city limit."
        if case == "normal":
            event["amount_usd"] = rng.randint(0, limit)
            reason = "Within the city limit."
        elif case == "cross-section":
            event.update(department="FIELD", amount_usd=151 if variant % 2 else 150)
            limit = 150
            ids.append("R_DEPARTMENT")
            decision = "reject" if variant % 2 else "approve"
            reason = "Apply the department cap from section 9 to section 1 eligibility."
        elif case == "cross-document":
            event["vendor_status"] = "restricted"
            ids.append("R_VENDOR")
            decision, reason = "reject", "Vendor control overrides monetary eligibility."
        elif case == "buried-exception":
            event.update(emergency=True, incident_id=f"INC-{i:06d}",
                         amount_usd=1001 if variant % 2 else 1000)
            limit = 1000
            ids.append("R_EMERGENCY")
            decision = "reject" if variant % 2 else "approve"
            reason = "Appendix Q replaces the city limit with the incident travel limit."
        elif case == "lookup-dependent":
            event["amount_usd"] = limit + (variant % 2)
            decision = "reject" if variant % 2 else "approve"
            reason = "Compare the claim with the exact city lookup row."
        else:
            variant %= 4
            if variant == 0:
                event["city"] = "Unlisted Harbor"
            elif variant == 1:
                del event["amount_usd"]
            elif variant == 2:
                event["receipt_amount_usd"] = event["amount_usd"] + 1
            else:
                event["emergency"] = True
            ids, decision, limit = ["R_ABSTAIN"], "abstain", None
            reason = "Evidence is missing or conflicting; do not guess."
        output = {"event_id": event["event_id"], "decision": decision, "limit_usd": limit,
                  "applied_rules": ids, "reason": reason}
        truth = {"case": case, "required_source_ids": ids, "critical_rules": ids,
                 "output": output}
        events.append(event)
        records.append({"event": event, "ground_truth": truth})
        expected[event["event_id"]] = truth
    dump(root / "lookup.json", cities)
    dump(root / "sources.json", sources)
    dump(root / "records.json", records)
    dump(root / "events.json", events)
    dump(root / "expected.json", expected)
    dump(root / "output.schema.json", Decision.model_json_schema())
    paths = sorted([*docs.glob("*.md"), *root.glob("*.json")])
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths if p.name != "manifest.json"}
    dump(root / "manifest.json", {"version": 1, "config": config.model_dump(), "sha256": hashes})
    validate(root)


def validate(root: Path):
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    sources, events, expected = read("sources.json"), read("events.json"), read("expected.json")
    source_map = {s["id"]: s for s in sources}
    require(len(source_map) == len(sources), "duplicate source IDs")
    actual = []
    for path in sorted((root / "documents").glob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("RULE "):
                rule = json.loads(line[5:])
                require(rule["document"] == path.name, "incorrect source location")
                actual.append(rule)
    require(sorted(actual, key=lambda s: s["id"]) == sorted(sources, key=lambda s: s["id"]),
            "source catalog differs from documents")
    rows = read("lookup.json")
    cities = {r["city"]: r for r in rows}
    require(len(cities) == len(rows), "duplicate city rows")
    for row in rows:
        source = source_map.get(row["id"], {})
        require(source.get("fact") == f"{row['city']} travel limit is {row['limit_usd']} USD."
                and source.get("when") == {"city": row["city"]}, "lookup source mismatch")
    event_ids = [e["event_id"] for e in events]
    require(len(set(event_ids)) == len(events), "duplicate event IDs")
    require(set(event_ids) == set(expected), "event/answer coverage mismatch")
    require({v["case"] for v in expected.values()} == set(CASES), "missing case coverage")
    for event in events:
        truth = expected[event["event_id"]]
        output = Decision.model_validate(truth["output"])
        required = truth["required_source_ids"]
        require(bool(required) and len(set(required)) == len(required), "invalid required sources")
        require(set(required) <= source_map.keys(), "unknown required source")
        require(required == output.applied_rules == truth["critical_rules"], "rule coverage mismatch")
        require(output.event_id == event["event_id"], "output event ID mismatch")
        require((output.decision, output.limit_usd, output.applied_rules) == answer(event, cities),
                "incorrect ground truth")
    require(read("records.json") == [{"event": e, "ground_truth": expected[e["event_id"]]}
                                    for e in events], "annotated records mismatch")
    manifest = read("manifest.json")
    config = DatasetConfig.model_validate(manifest["config"])
    require(len(events) == config.events and len(rows) == config.lookup_rows, "config count mismatch")
    paths = {p.relative_to(root).as_posix(): p for p in
             [*(root / "documents").glob("*.md"), *root.glob("*.json")]
             if p.name != "manifest.json"}
    require(paths.keys() == manifest["sha256"].keys(), "manifest file coverage mismatch")
    for name, path in paths.items():
        require(hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"][name],
                f"checksum mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/datasets/small.json"))
    parser.add_argument("--output", type=Path, default=Path("data/enterprise"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate(args.output)
    else:
        config = DatasetConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        if args.seed is not None:
            config.seed = args.seed
        generate(args.output, config)
    print(f"Validated dataset: {args.output}")


if __name__ == "__main__":
    main()
