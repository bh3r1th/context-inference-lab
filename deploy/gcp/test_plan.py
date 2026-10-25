"""Check that cloud tasks partition the documented pilot without changing its work."""

import json
import re
from pathlib import Path

import pytest

from contextbench.experiments.config import load_config
from deploy.gcp.make_plan import ROOT, make_plan


def signature(entry):
    return (entry["config"], entry["strategy"], entry["concurrency"],
            entry.get("scaling_size"))


def test_pilot_covers_each_documented_cell_once():
    plan = make_plan("pilot")
    expected = {
        ("configs/pilot.yaml", strategy, concurrency, None)
        for strategy in ("full_context", "prefix_cached_full_context", "retrieval",
                         "compiled_context") for concurrency in (1, 8)
    }
    expected.update({
        ("configs/pilot-scaling.yaml", "full_context", concurrency, size)
        for concurrency in (1, 8) for size in ("2k", "8k", "32k")
    })
    expected.update({
        (f"configs/pilot-prefix-{scenario}.yaml", "prefix_cached_full_context", concurrency, None)
        for concurrency in (1, 8) for scenario in ("cold", "warm")
    })
    assert len(plan) == len(expected) == 18
    assert {signature(entry) for entry in plan} == expected
    assert len({entry["id"] for entry in plan}) == len(plan)
    assert all(re.fullmatch(r"[a-z0-9_-]+", entry["id"]) for entry in plan)


def test_pilot_preserves_workload_and_selected_cache_contract():
    protected = {p: p.read_bytes() for p in (ROOT / "configs").rglob("*") if p.is_file()}
    plan = make_plan("pilot")
    measured_requests = 0
    for entry in plan:
        config = load_config(ROOT / entry["config"])
        assert config.repetitions == 1
        assert len(config.event_ids) == 10
        assert config.seed == (1729 if entry.get("scaling_size") else 42)
        server = config.server_for(entry["strategy"])
        assert server.prefix_caching == (entry["strategy"] == "prefix_cached_full_context")
        measured_requests += len(config.event_ids) * config.repetitions
    assert measured_requests == 180
    assert all(path.read_bytes() == content for path, content in protected.items())


def test_checked_in_smoke_plan_is_one_real_request():
    plan = json.loads((Path(__file__).parent / "smoke-plan.json").read_text())
    assert plan == make_plan("smoke")
    config = load_config(ROOT / plan[0]["config"])
    assert len(plan) == 1
    assert config.event_ids == ["EV-000"]
    assert config.repetitions == 1 and config.warmup_requests == 0
    assert plan[0]["strategy"] == "full_context" and plan[0]["concurrency"] == 1


def test_full_plan_requires_pilot_measurements():
    with pytest.raises(ValueError, match="full runs require GPU pilot results"):
        make_plan("full")
