"""Matrix tests use deterministic fixture responses, never GPU measurements."""
import asyncio
import csv
import json
from collections import Counter

import pytest

from contextbench.context import make_strategy
from contextbench.experiments import orchestrator
from contextbench.experiments.config import ExperimentConfig
from contextbench.metrics.summary import summarize
from contextbench.models import InferenceResult

SERVER = {"model": "fixture", "model_revision": "fixture", "vllm_version": "fixture",
          "launch_command": "fixture", "metrics_url": None}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def fake_backend(dataset, monkeypatch):
    truth = read(dataset / "expected.json")

    class Backend:
        instances = []
        fail_event = None
        fail_discovery = False

        def __init__(self, config, client):
            self.config = config
            self.headers = {}
            self.active = self.peak = self.calls = 0
            self.instances.append(self)

        async def metadata(self):
            if self.fail_discovery:
                raise RuntimeError("discovery unavailable")
            return {"data": [{"id": "fixture"}]}

        async def version(self):
            return {"value": "fixture", "source": "test", "reason": None}

        async def generate(self, context, event, seed):
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0)
                if event["event_id"] == self.fail_event:
                    raise RuntimeError("injected inference failure")
                return InferenceResult(text=json.dumps(truth[event["event_id"]]["output"]),
                                       input_tokens=100, output_tokens=20, ttft_s=.1, latency_s=.2)
            finally:
                self.active -= 1

    monkeypatch.setattr(orchestrator, "VLLMBackend", Backend)
    return Backend


def configuration(dataset, output, **overrides):
    return ExperimentConfig.model_validate({
        "dataset": str(dataset), "output_dir": str(output), "repetitions": 2, "warmup_requests": 2,
        "server": SERVER, "strategy_servers": {
            "prefix_cached_full_context": {**SERVER, "base_url": "http://cached.test/v1",
                                           "prefix_caching": True}}, **overrides})


def test_default_full_matrix_same_workload_ids_warmups_and_summaries(dataset, tmp_path, fake_backend):
    config = configuration(dataset, tmp_path / "runs", repetitions=8)
    output = asyncio.run(orchestrator.run(config))
    summaries = read(output / "summary.json")
    metadata = read(output / "metadata.json")
    report = read(output / "report.json")
    assert metadata["status"] == "completed"
    assert metadata["planned_cells"] == metadata["completed_cells"] == len(summaries) == 12
    assert len(report["cells"]) == 12
    assert (output / "report.md").exists()
    assert config.concurrency == [1, 8, 32]
    plan = read(output / "workload_manifest.json")
    assert len(plan) == 40
    counts = Counter((e["event_id"], e["repetition"]) for e in plan)
    assert len(counts) == 40 and set(counts.values()) == {1}
    reference, ids = None, set()
    for summary in summaries:
        cell = output / f"{summary['strategy']}-c{summary['concurrency']}"
        rows = sorted(jsonl(cell / "requests.jsonl"), key=lambda r: r["request_index"])
        warmups = jsonl(cell / "warmup.jsonl")
        assert len(rows) == summary["requests"] == 40
        assert len(warmups) == summary["warmup_requests"] == 2
        assert all(r["phase"] == "warmup" for r in warmups)
        assert all(r["phase"] == "measurement" for r in rows)
        order = [(r["event_id"], r["seed"], r["repetition"]) for r in rows]
        if reference is None:
            reference = order
        assert order == reference
        assert rows[0]["experiment_id"] == metadata["experiment_id"]
        assert summary["failure_rate"] == 0 and summary["batch_duration_s"] > 0
        assert summary["average_input_tokens"] == 100 and summary["average_output_tokens"] == 20
        assert summary["successful_request_metrics"]["ttft_s"]["p99"] == .1
        for row in rows + warmups:
            assert row["request_id"] not in ids
            ids.add(row["request_id"])
        if summary["strategy"] == "prefix_cached_full_context":
            assert summary["server_url"] == "http://cached.test/v1" and summary["prefix_caching"]
        else:
            assert not summary["prefix_caching"]
    assert all(backend.peak == 32 and backend.active == 0 for backend in fake_backend.instances)
    with (output / "summary.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 12
    assert csv_rows[0]["latency_s_p95"] == "0.2" and csv_rows[0]["failure_rate"] == "0.0"


def test_reruns_unique_and_order_reproducible(dataset, tmp_path, fake_backend):
    config = configuration(dataset, tmp_path / "runs", strategies=["full_context"], concurrency=[1],
                           experiment_id="shared-experiment", warmup_requests=0)
    first = asyncio.run(orchestrator.run(config))
    before = (first / "full_context-c1/requests.jsonl").read_bytes()
    second = asyncio.run(orchestrator.run(config))
    assert first != second
    assert read(first / "metadata.json")["experiment_id"] == "shared-experiment"
    assert read(first / "workload_manifest.json") == read(second / "workload_manifest.json")
    assert (first / "full_context-c1/requests.jsonl").read_bytes() == before
    changed = configuration(dataset, tmp_path / "runs", seed=43, strategies=["full_context"], concurrency=[1])
    third = asyncio.run(orchestrator.run(changed))
    assert read(first / "workload.json") != read(third / "workload.json")


def test_preparation_context_and_inference_failures_do_not_stop_matrix(
        dataset, tmp_path, fake_backend, monkeypatch):
    fake_backend.fail_event = "EV-004"

    class FailingContext:
        def __init__(self):
            self.strategy = make_strategy("compiled_context", dataset)

        def build(self, event):
            if event["event_id"] == "EV-001":
                raise RuntimeError("injected context failure")
            return self.strategy.build(event)

    def factory(name, root, **kwargs):
        if name == "retrieval":
            raise RuntimeError("injected preparation failure")
        return FailingContext() if name == "compiled_context" else make_strategy(name, root, **kwargs)

    monkeypatch.setattr(orchestrator, "make_strategy", factory)
    output = asyncio.run(orchestrator.run(configuration(dataset, tmp_path / "runs", concurrency=[1, 8])))
    summaries = read(output / "summary.json")
    assert len(summaries) == 8 and read(output / "metadata.json")["status"] == "completed_with_errors"
    assert jsonl(output / "errors.jsonl")
    for summary in summaries:
        cell = output / f"{summary['strategy']}-c{summary['concurrency']}"
        rows = jsonl(cell / "requests.jsonl")
        if summary["strategy"] == "retrieval":
            assert summary["status"] == "failed" and rows == []
            assert summary["batch_duration_s"] is None and summary["unattempted_requests"] == 10
            assert summary["errors"][0]["phase"] == "setup"
        else:
            assert len(rows) == 10 and summary["failure_rate"] > 0
            assert any(r["error_stage"] == "inference" for r in rows)
            if summary["strategy"] == "compiled_context":
                failed_context = next(r for r in rows if r["error_stage"] == "context")
                assert not failed_context["inference_attempted"]
                assert failed_context["input_tokens"] is None
                assert failed_context["context_sha256"] is None


def test_warmup_failure_retained_and_measurement_still_runs(dataset, tmp_path, fake_backend):
    config = configuration(dataset, tmp_path / "runs", strategies=["full_context"], concurrency=[1],
                           event_ids=["EV-000"], repetitions=1)
    original = fake_backend.generate

    async def fail_first(self, context, event, seed):
        if self.calls == 0:
            self.calls += 1
            raise RuntimeError("first warmup failed")
        return await original(self, context, event, seed)

    fake_backend.generate = fail_first
    output = asyncio.run(orchestrator.run(config))
    summary = read(output / "summary.json")[0]
    assert summary["requests"] == 1 and summary["failure_rate"] == 0
    assert summary["warmup_requests"] == 2 and summary["warmup_failures"] == 1
    assert jsonl(output / "full_context-c1/warmup.jsonl")[0]["error"]
    assert summary["status"] == "completed_with_errors"


def test_cached_without_enabled_endpoint_fails_only_cached_cells(dataset, tmp_path, fake_backend):
    output = asyncio.run(orchestrator.run(configuration(
        dataset, tmp_path / "runs", strategy_servers={}, concurrency=[1])))
    summaries = read(output / "summary.json")
    assert len(summaries) == 4
    assert [s["strategy"] for s in summaries if s["status"] == "failed"] == ["prefix_cached_full_context"]


def test_model_discovery_failure_still_attempts_requests(dataset, tmp_path, fake_backend):
    fake_backend.fail_discovery = True
    output = asyncio.run(orchestrator.run(configuration(
        dataset, tmp_path / "runs", strategies=["full_context"], concurrency=[1])))
    assert read(output / "summary.json")[0]["requests"] == 10
    assert jsonl(output / "errors.jsonl")[0]["phase"] == "server_discovery"


def test_summary_averages_percentiles_and_failure_rate():
    rows = [{"error": None if i < 3 else "failure", "input_tokens": count,
             "output_tokens": count, "latency_s": i + 1, "ttft_s": (i + 1) / 10,
             "correct": i < 3, "schema_valid": i < 3, "rule_recall": 1,
             "missed_critical_rules": []} for i, count in enumerate([10, 20, None, 30])]
    result = summarize(rows, 2)
    assert result["failure_rate"] == .25 and result["requests_per_s"] == 1.5
    assert result["average_input_tokens"] == result["average_output_tokens"] == 20
    assert result["token_totals"]["input_tokens"]["observed_requests"] == 3
    assert result["successful_request_metrics"]["latency_s"]["p50"] == 2
    assert result["successful_request_metrics"]["latency_s"]["p95"] == pytest.approx(2.9)
    assert result["successful_request_metrics"]["latency_s"]["p99"] == pytest.approx(2.98)
    assert result["throughput"]["output_tokens_per_s"]["value"] is None
    empty = summarize([], None)
    assert empty["failure_rate"] is None and empty["batch_duration_s"] is None

    def test_summary_groups_quality_by_case_context_and_concurrency():
        rows = [{"error": None, "input_tokens": None, "output_tokens": None,
                 "latency_s": None, "ttft_s": None, "correct": correct,
                 "schema_valid": True, "answer_correct": correct,
                 "required_rule_coverage": 1.0, "abstention_correct": None,
                 "cross_document_reasoning_success": correct if case == "cross-document" else None,
                 "buried_exception_success": None, "minimum_safe_context": safe,
                 "rule_recall": 1.0, "missed_critical_rules": [],
                 "incorrect_rule_inclusion": [], "strategy": "retrieval", "case": case,
                 "context_chars": context_chars, "concurrency": 4}
                for correct, safe, case, context_chars in
                [(True, True, "normal", 100), (False, False, "cross-document", 200)]]
        quality = summarize(rows, 1)["quality_by_strategy_case_context_size_concurrency"]
        assert {group["case"] for group in quality} == {"normal", "cross-document"}
        cross_document = next(group for group in quality if group["case"] == "cross-document")
        assert cross_document["context_size"] == 200
        assert cross_document["concurrency"] == 4
        assert cross_document["cross_document_reasoning_success"] == 0
        assert cross_document["minimum_safe_context"] == 0
