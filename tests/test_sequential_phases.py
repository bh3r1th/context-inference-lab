import sys
import importlib.util
import hashlib
from pathlib import Path

import pytest

from contextbench.experiments.config import load_config
from contextbench.experiments.phases import phase_config, scaling_paths

_VALIDATE_SPEC = importlib.util.spec_from_file_location(
    "validate_pilot", Path(__file__).parents[1] / "scripts/validate_pilot.py")
validate_pilot = importlib.util.module_from_spec(_VALIDATE_SPEC)
_VALIDATE_SPEC.loader.exec_module(validate_pilot)


ROOT = Path(__file__).parents[1]


def test_sequential_phases_use_only_port_8000_and_preserve_strategy_union():
    uncached = phase_config(ROOT / "configs/experiment.yaml", "uncached")
    cached = phase_config(ROOT / "configs/experiment.yaml", "cached")
    assert uncached.server.base_url == "http://localhost:8000/v1"
    assert cached.server.base_url == "http://localhost:8000/v1"
    assert uncached.server.metrics_url == "http://localhost:8000/metrics"
    assert cached.server.metrics_url == "http://localhost:8000/metrics"
    assert uncached.server.prefix_caching is False
    assert cached.server.prefix_caching is True
    assert uncached.strategies == ["full_context", "retrieval", "compiled_context"]
    assert cached.strategies == ["prefix_cached_full_context"]
    assert set(uncached.strategies) | set(cached.strategies) == set(
        load_config(ROOT / "configs/experiment.yaml").strategies)


def test_pilot_validation_cannot_pass_with_only_one_phase(tmp_path):
    output = tmp_path / "pilot.md"
    sys.argv = ["validate_pilot", str(tmp_path / "uncached"), "--output", str(output)]
    with pytest.raises(SystemExit) as result:
        validate_pilot.main()
    assert result.value.code == 1
    assert "BLOCKED" in output.read_text(encoding="utf-8")
    assert "missing required completed phases" in output.read_text(encoding="utf-8")


def test_scaling_phase_uses_shared_fixture_root(tmp_path):
    fixture_root = tmp_path / "context-scaling"
    dataset, run_output = scaling_paths(fixture_root, fixture_root / "uncached" / "runs", "2k")
    assert dataset == fixture_root / "2k"
    assert dataset != fixture_root / "uncached" / "2k"
    assert run_output == fixture_root / "uncached" / "runs"


def test_cached_and_uncached_scaling_fixture_hashes_are_identical(tmp_path):
    fixture_root = tmp_path / "context-scaling"
    uncached_dataset, _ = scaling_paths(fixture_root, fixture_root / "uncached" / "runs", "8k")
    cached_dataset, _ = scaling_paths(fixture_root, fixture_root / "cached" / "runs", "8k")
    assert uncached_dataset == cached_dataset
    fixture = uncached_dataset / "events.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("shared fixture", encoding="utf-8")
    assert hashlib.sha256(fixture.read_bytes()).digest() == hashlib.sha256(
        cached_dataset.joinpath("events.json").read_bytes()).digest()