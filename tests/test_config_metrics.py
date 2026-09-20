import pytest
from pydantic import ValidationError

from contextbench.experiments.config import ExperimentConfig, load_config
from contextbench.metrics.summary import percentiles
from contextbench.metrics.telemetry import kv_samples, summarize_telemetry

SERVER = {"model_revision": "abc", "vllm_version": "test", "launch_command": "test"}


@pytest.mark.parametrize("patch", [
    {"concurrency": [0]}, {"concurrency": [1, 1]}, {"concurrency": []},
    {"strategies": ["typo"]}, {"repetitions": 0}, {"unknown": True},
    {"event_ids": []}, {"event_ids": ["EV-000", "EV-000"]},
    {"telemetry_interval_s": float("inf")},
    {"server": {**SERVER, "timeout_s": float("nan")}},
    {"strategy_servers": {"typo": SERVER}},
])
def test_config_rejects_invalid(patch):
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"server": SERVER, **patch})


def test_relative_paths_resolved(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("server:\n  model_revision: abc\n  vllm_version: test\n  launch_command: test\n")
    assert load_config(path).dataset == str((tmp_path / "../data/synthetic").resolve())


def test_percentiles():
    assert percentiles([1, 2, 3, 4])["p50"] == 2.5
    assert percentiles([1, 2, 3, 4])["p95"] == pytest.approx(3.85)
    assert percentiles([1])["p99"] == 1
    assert percentiles([])["p50"] is None


def test_kv_metrics_keep_labels_and_skip_nonfinite():
    values = kv_samples('# HELP unrelated\nvllm:kv_cache_usage_perc{engine="0"} 0.25\n'
                        'vllm:kv_cache_usage_perc{engine="1"} 0.75\n'
                        'vllm:gpu_cache_usage_perc NaN\n')
    assert [v["value"] for v in values] == [0.25, 0.75]
    summary = summarize_telemetry([{"kv_cache": {"samples": values}, "gpu": {}}])
    assert len(summary["kv_cache"]["series"]) == 2
    assert summarize_telemetry([])["kv_cache"]["status"] == "unavailable"
