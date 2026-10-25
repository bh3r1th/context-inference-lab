"""Cloud-free checks of task partitioning, failure handling, and artifact outcomes."""
import json
import signal
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from deploy.gcp import task

ROOT = Path(__file__).resolve().parents[2]
SMOKE = {"id": "smoke", "config": "configs/smoke.yaml", "strategy": "full_context",
         "concurrency": 1}


@pytest.fixture
def config(tmp_path):
    return task.cell_config(SMOKE, ROOT, tmp_path, "0.29.0")


def artifacts(config, *, status="completed", correct=True, requests=1):
    root = Path(config.output_dir) / "current-run"
    cell = root / f"{config.strategies[0]}-c{config.concurrency[0]}"
    cell.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({"status": status}))
    (root / "summary.json").write_text(json.dumps([{
        "status": status, "requests": requests, "successful_requests": requests,
    }]))
    (cell / "requests.jsonl").write_text("".join(
        json.dumps({"correct": correct, "error": None}) + "\n" for _ in range(requests)))


def test_plan_requires_exact_task_count_and_unique_ids(tmp_path):
    path = tmp_path / "plan.json"
    second = {**SMOKE, "id": "second"}
    path.write_text(json.dumps([SMOKE, second]))
    assert task.read_plan(path, 1, 2) == second
    with pytest.raises(ValueError, match="count must equal"):
        task.read_plan(path, 0, 1)
    with pytest.raises(ValueError, match="index must be in range"):
        task.read_plan(path, 2, 2)
    path.write_text(json.dumps([SMOKE, SMOKE]))
    with pytest.raises(ValueError, match="unique"):
        task.read_plan(path, 0, 2)


def test_cell_normalizes_cached_server_and_records_actual_command(tmp_path):
    entry = {"id": "cached", "config": "configs/pilot-prefix-cold.yaml",
             "strategy": "prefix_cached_full_context", "concurrency": 8}
    config = task.cell_config(entry, ROOT, tmp_path, "0.29.0")
    assert config.server.prefix_caching and config.phase == "cached"
    assert config.cache_scenario == "cold" and config.warmup_requests == 0
    assert config.server.base_url == "http://127.0.0.1:8000/v1"
    assert config.strategy_servers == {} and config.concurrency == [8]
    assert config.strategies == ["prefix_cached_full_context"]
    command = task.shlex.split(config.server.launch_command)
    assert command.count("--enable-prefix-caching") == 1
    assert "--no-enable-prefix-caching" not in command
    assert command[command.index("--port") + 1] == "8000"
    assert command[command.index("--max-model-len") + 1] == "72K"
    assert config.server.max_tokens == 256


def test_cell_preserves_smoke_workload_and_checks_version(config, tmp_path):
    assert config.event_ids == ["EV-000"] and config.repetitions == 1
    assert config.server.vllm_version == "0.29.0"
    assert not config.server.prefix_caching and config.phase == "uncached"
    with pytest.raises(ValueError, match="differs"):
        task.cell_config(SMOKE, ROOT, tmp_path, "0.8.5.post1")
    with pytest.raises(ValueError, match="Only smoke/pilot"):
        task.cell_config({**SMOKE, "config": "configs/experiment.yaml"}, ROOT, tmp_path, "0.29.0")
    with pytest.raises(ValueError, match="outside"):
        task.cell_config({**SMOKE, "concurrency": 32}, ROOT, tmp_path, "0.29.0")


@pytest.mark.parametrize("original", ["--gpu-memory-utilization 0.90",
                                       "--gpu-memory-utilization=0.90"])
def test_l4_budget_replaces_source_budget_without_changing_model_settings(tmp_path, original):
    source = task.load_config(ROOT / SMOKE["config"])
    command = source.server.launch_command.replace("--gpu-memory-utilization 0.90", original)
    args = task.server_command(command, model=source.server.model,
                               revision=source.server.model_revision, cached=False)
    assert args.count("--gpu-memory-utilization") == 1
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.92"
    assert args[args.index("--max-model-len") + 1] == "72K"
    assert args[args.index("--revision") + 1] == source.server.model_revision
    assert "--enforce-eager" not in args


def test_budget_override_is_recorded_in_generated_config(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTEXTBENCH_GPU_MEMORY_UTILIZATION", "0.93")
    config = task.cell_config(SMOKE, ROOT, tmp_path, "0.29.0+cu129")
    args = task.shlex.split(config.server.launch_command)
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.93"


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "1", "-0.1", "1.1", "bad"])
def test_invalid_memory_budget_fails_before_model_loading(monkeypatch, tmp_path, value):
    monkeypatch.setenv("CONTEXTBENCH_GPU_MEMORY_UTILIZATION", value)
    with pytest.raises(ValueError):
        task.cell_config(SMOKE, ROOT, tmp_path, "0.29.0+cu129")


def test_official_cuda_build_preserves_workload_and_records_full_version(config, tmp_path):
    from deploy.gcp.check_dependencies import EXPECTED_VERSIONS

    actual_version = EXPECTED_VERSIONS["vllm"]
    assert actual_version == "0.29.0+cu129"
    cuda_config = task.cell_config(SMOKE, ROOT, tmp_path, actual_version)
    assert cuda_config.server.vllm_version == actual_version
    # Only provenance changes; model, request settings and generated command stay fixed.
    assert cuda_config.model_dump() == {
        **config.model_dump(),
        "server": {**config.server.model_dump(), "vllm_version": actual_version},
    }


@pytest.mark.parametrize("version", [
    "0.29.0+cu130", "0.29.0+unknown", "0.29.0+cu129.extra", "0.30.0+cu129",
    "0.29.0.dev1+cu129", "0.29.0.post1+cu129",
])
def test_unapproved_builds_cannot_satisfy_configured_release(tmp_path, version):
    with pytest.raises(ValueError, match="differs"):
        task.cell_config(SMOKE, ROOT, tmp_path, version)


@pytest.mark.parametrize("status,correct,requests", [
    ("completed_with_errors", True, 1), ("completed", True, 0), ("completed", False, 1),
])
def test_zero_exit_or_empty_outputs_do_not_establish_success(config, status, correct, requests):
    artifacts(config, status=status, correct=correct, requests=requests)
    with pytest.raises(task.TaskFailure):
        task.verify_results(config, require_correct=True)


def test_completed_nonempty_correct_smoke_is_accepted(config):
    artifacts(config)
    task.verify_results(config, require_correct=True)


def test_ready_probe_checks_served_model_over_http(config):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            assert self.path == "/v1/models"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": config.server.model}]}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config.server.base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        task.wait_ready(FakeProcess(None), config, task.Control(2), lambda: None, 1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class FakeProcess:
    pid = 12345

    def __init__(self, returncode):
        self.returncode = returncode

    def poll(self):
        return self.returncode


class Store:
    def __init__(self, *, deny_probe=False, fail_sync_at=0):
        self.deny_probe, self.fail_sync_at = deny_probe, fail_sync_at
        self.synced = []
        self.markers = []
        self.status_updates = []

    def probe(self):
        if self.deny_probe:
            raise PermissionError("injected bucket denial")

    def sync(self, root):
        self.synced.append(json.loads((root / "task-status.json").read_text()))
        if len(self.synced) == self.fail_sync_at:
            raise OSError("injected upload failure")

    def put_json(self, name, result):
        if name == "task-status.json":
            self.status_updates.append(dict(result))
        else:
            self.markers.append((name, dict(result)))


def fake_children(monkeypatch, config, *, benchmark_exit=0, ready_error=None, probe_exit=0):
    children, terminated = [], []

    def popen(command, **kwargs):
        assert kwargs["start_new_session"] is True
        if command[1:] == ["-m", "deploy.gcp.gpu_probe"]:
            child = FakeProcess(probe_exit)
        elif command[0] == "vllm":
            child = FakeProcess(None)
        else:
            assert "scripts/run_benchmark.py" in command
            child = FakeProcess(benchmark_exit)
            artifacts(config)
        child.command = command
        children.append(child)
        return child

    def ready(*_):
        if ready_error:
            raise ready_error

    monkeypatch.setattr(task.subprocess, "Popen", popen)
    monkeypatch.setattr(task, "wait_ready", ready)
    monkeypatch.setattr(task, "terminate", lambda process: terminated.append(process))
    return children, terminated


def test_success_marks_complete_only_after_final_artifacts_upload(monkeypatch, config, tmp_path):
    children, terminated = fake_children(monkeypatch, config)
    store = Store()
    assert task.execute(SMOKE, config, tmp_path, ROOT, store, task.Control(5)) == 0
    assert len(children) == 3 and terminated == [children[2], children[1], children[0]]
    assert store.synced[-1]["status"] == "completed"
    assert store.markers[0][0] == "complete.json"


@pytest.mark.parametrize("failure", ["benchmark", "readiness", "final_upload", "bucket", "gpu"])
def test_failure_preserves_status_and_never_marks_complete(monkeypatch, config, tmp_path, failure):
    children, terminated = fake_children(
        monkeypatch, config, benchmark_exit=7 if failure == "benchmark" else 0,
        ready_error=task.TaskFailure("startup failed") if failure == "readiness" else None,
        probe_exit=1 if failure == "gpu" else 0)
    store = Store(deny_probe=failure == "bucket", fail_sync_at=2 if failure == "final_upload" else 0)
    assert task.execute(SMOKE, config, tmp_path, ROOT, store, task.Control(5)) == 1
    assert json.loads((tmp_path / "task-status.json").read_text())["status"] == "failed"
    assert store.markers == []
    if failure == "bucket":
        assert children == []
    else:
        assert children[0] in terminated
    if failure == "gpu":
        assert len(children) == 1 and "deploy.gcp.gpu_probe" in children[0].command
        assert (tmp_path / "gpu-preflight.log").exists()
        assert not (tmp_path / "vllm.log").exists()
    if failure == "final_upload":
        assert store.status_updates[-1]["status"] == "failed"


@pytest.mark.parametrize("interruption", ["deadline", "signal"])
@pytest.mark.parametrize("stage", ["probe", "readiness"])
def test_interruption_uploads_partial_artifacts_and_terminates_children(
        monkeypatch, config, tmp_path, interruption, stage):
    children, terminated = fake_children(monkeypatch, config)
    control = task.Control(5)

    def interrupt(*_):
        if interruption == "signal":
            control.stopped.set()
        else:
            control.deadline = 0
        control.check()

    monkeypatch.setattr(task, "wait_probe" if stage == "probe" else "wait_ready", interrupt)
    store = Store()
    assert task.execute(SMOKE, config, tmp_path, ROOT, store, control) == 1
    assert children[0] in terminated and len(children) == (1 if stage == "probe" else 2)
    assert store.synced[-1]["status"] == "failed"
    assert store.markers == []


def test_gpu_probe_timeout_stops_before_model_start(monkeypatch, config, tmp_path):
    children, terminated = fake_children(monkeypatch, config, probe_exit=None)
    store = Store()
    assert task.execute(SMOKE, config, tmp_path, ROOT, store, task.Control(5), probe_timeout=0) == 1
    assert len(children) == 1 and children[0] in terminated
    assert store.markers == []
    assert "GPU preflight exceeded" in store.synced[-1]["error"]


def test_terminate_targets_only_child_process_group_and_escalates(monkeypatch):
    process = FakeProcess(None)
    calls = []

    def wait(timeout):
        calls.append(("wait", timeout))
        if len(calls) == 2:
            raise subprocess.TimeoutExpired("child", timeout)
        process.returncode = -signal.SIGKILL

    process.wait = wait
    monkeypatch.setattr(task.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    task.terminate(process)
    assert calls[0] == (process.pid, signal.SIGTERM)
    assert calls[2] == (process.pid, signal.SIGKILL)
