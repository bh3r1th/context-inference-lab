import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from contextbench.workload import CASES, DatasetConfig, answer, generate, validate


def read(root, name):
    return json.loads((root / name).read_text(encoding="utf-8"))


@pytest.fixture
def workload(tmp_path):
    generate(tmp_path, DatasetConfig(events=48, reference_sections=8, lookup_rows=4))
    return tmp_path


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_presets(tmp_path, size):
    config = DatasetConfig.model_validate_json(
        (Path(__file__).parents[1] / f"configs/datasets/{size}.json").read_text())
    generate(tmp_path, config)
    validate(tmp_path)
    assert len(read(tmp_path, "events.json")) == config.events
    assert len(read(tmp_path, "lookup.json")) == config.lookup_rows
    assert {t["case"] for t in read(tmp_path, "expected.json").values()} == set(CASES)
    assert all(p.stat().st_size > config.reference_sections * 400
               for p in (tmp_path / "documents").glob("*.md"))


def test_reproducible_and_seed_sensitive(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    config = DatasetConfig(events=24, reference_sections=8, lookup_rows=4)
    generate(first, config)
    generate(second, config)
    before = {p.relative_to(first): p.read_bytes() for p in first.rglob("*") if p.is_file()}
    assert before == {p.relative_to(second): p.read_bytes()
                      for p in second.rglob("*") if p.is_file()}
    generate(first, config)  # Existing manifests must not affect regeneration.
    assert before == {p.relative_to(first): p.read_bytes() for p in first.rglob("*") if p.is_file()}
    generate(second, config.model_copy(update={"seed": 42}))
    assert read(first, "events.json") != read(second, "events.json")


def test_annotations_and_topology(workload):
    records = read(workload, "records.json")
    sources = {s["id"]: s for s in read(workload, "sources.json")}
    for record in records:
        truth = record["ground_truth"]
        assert "required_source_ids" not in record["event"]
        assert all(r in sources for r in truth["required_source_ids"])
        if truth["case"] == "cross-section":
            assert sources["R_BASE"]["document"] == sources["R_DEPARTMENT"]["document"]
            assert sources["R_BASE"]["section"] != sources["R_DEPARTMENT"]["section"]
        if truth["case"] == "cross-document":
            assert sources["R_BASE"]["document"] != sources["R_VENDOR"]["document"]
    text = (workload / "documents/exceptions.md").read_text(encoding="utf-8")
    assert text.index('"id": "R_EMERGENCY"') > len(text) * .75
    for case in ("cross-section", "buried-exception", "lookup-dependent"):
        assert {r["ground_truth"]["output"]["decision"] for r in records
                if r["ground_truth"]["case"] == case} == {"approve", "reject"}
    abstentions = [r["event"] for r in records
                  if r["ground_truth"]["case"] == "ambiguous-abstain"]
    assert any("amount_usd" not in e for e in abstentions)
    assert any("receipt_amount_usd" in e for e in abstentions)


@pytest.mark.parametrize("corruption", ["unknown_rule", "decision", "missing_answer",
                                       "duplicate_event", "lookup", "document", "records"])
def test_rejects_corrupt_ground_truth(workload, corruption):
    filename = "expected.json"
    value = read(workload, filename)
    key = next(iter(value))
    if corruption == "unknown_rule":
        value[key]["required_source_ids"].append("DOES_NOT_EXIST")
    elif corruption == "decision":
        value[key]["output"]["decision"] = "reject"
    elif corruption == "missing_answer":
        del value[key]
    elif corruption == "duplicate_event":
        filename = "events.json"
        value = read(workload, filename)
        value.append(value[0])
    elif corruption == "lookup":
        filename = "lookup.json"
        value = read(workload, filename)
        value[0]["limit_usd"] += 1
    elif corruption == "records":
        filename, value = "records.json", []
    else:
        path = workload / "documents/policy.md"
        path.write_text(path.read_text(encoding="utf-8").replace("R_BASE", "R_MISSING"),
                        encoding="utf-8")
    if corruption != "document":
        (workload / filename).write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        validate(workload)


def test_precedence_and_boundary_oracle():
    cities = {"Test": {"id": "L_TEST", "limit_usd": 300}}
    event = {"city": "Test", "category": "travel", "currency": "USD", "amount_usd": 1000,
             "department": "FIELD", "vendor_status": "restricted", "emergency": True,
             "incident_id": "INC-1"}
    assert answer(event, cities) == (
        "reject", 1000, ["R_BASE", "L_TEST", "R_DEPARTMENT", "R_EMERGENCY", "R_VENDOR"])
    assert answer({**event, "vendor_status": "clear"}, cities)[0] == "approve"
    assert answer({**event, "receipt_amount_usd": 999}, cities) == (
        "abstain", None, ["R_ABSTAIN"])


@pytest.mark.parametrize("patch", [{"events": 5}, {"seed": "1"}, {"lookup_rows": 0},
                                  {"reference_sections": -1}, {"unexpected": 1}])
def test_invalid_config(patch):
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate(patch)
