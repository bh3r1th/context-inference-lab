"""Run one smoke/pilot cell per Cloud Run task; persist every attempt separately.

This module deliberately does not bypass the repository's full-benchmark pilot gate.
Cloud tests validate orchestration only; a successful GPU pilot is still required.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import httpx
import yaml

from contextbench.experiments.config import ExperimentConfig, load_config


class TaskFailure(RuntimeError):
    pass


def read_plan(path: Path, index: int, count: int, *, cell_id: str | None = None) -> dict:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("TASK_PLAN must contain a nonempty JSON array")
    ids = [entry.get("id") for entry in entries if isinstance(entry, dict)]
    if len(ids) != len(entries) or any(
        not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
        for name in ids
    ) or len(set(ids)) != len(ids):
        raise ValueError("Plan entries need unique, safe string IDs")
    if cell_id is not None:
        entries = [entry for entry in entries if entry["id"] == cell_id]
        if not entries:
            raise ValueError(f"TASK_CELL_ID is absent from the task plan: {cell_id}")
    if count != len(entries) or not 0 <= index < count:
        raise ValueError("Cloud Run task count must equal selected plan length; "
                         "index must be in range")
    return entries[index]


def server_command(command: str, *, model: str, revision: str, cached: bool,
                   gpu_memory_utilization: float = 0.92) -> list[str]:
    if not math.isfinite(gpu_memory_utilization) or not 0 < gpu_memory_utilization < 1:
        raise ValueError("GPU memory utilization must be finite and between 0 and 1")
    args = shlex.split(command)
    if args[:3] != ["vllm", "serve", model]:
        raise ValueError("Launch command must start with vllm serve and the configured model")
    revision_index = args.index("--revision") if "--revision" in args else len(args)
    if revision_index + 1 >= len(args) or args[revision_index + 1] != revision:
        raise ValueError("Launch command must use the configured immutable model revision")
    if "--api-key" in args or any(arg.startswith("--api-key=") for arg in args):
        raise ValueError("Do not embed credentials in launch commands")
    normalized, index = [], 0
    while index < len(args):
        arg = args[index]
        if arg in {"--host", "--port", "--gpu-memory-utilization"}:
            if index + 1 >= len(args):
                raise ValueError(f"Missing value for {arg}")
            index += 2
            continue
        if arg.startswith(("--host=", "--port=", "--gpu-memory-utilization=")) or arg in {
            "--enable-prefix-caching", "--no-enable-prefix-caching"
        }:
            index += 1
            continue
        normalized.append(arg)
        index += 1
    return [*normalized, "--host", "127.0.0.1", "--port", "8000",
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--enable-prefix-caching" if cached else "--no-enable-prefix-caching"]


def cell_config(entry: dict, root: Path, output: Path, version: str) -> ExperimentConfig:
    config_path = (root / entry["config"]).resolve()
    if not config_path.is_relative_to((root / "configs").resolve()):
        raise ValueError("Config must be within the repository configs directory")
    if config_path.name != "smoke.yaml" and not config_path.name.startswith("pilot"):
        raise ValueError("Only smoke/pilot configs are supported until the GPU pilot passes")
    config = load_config(config_path)
    strategy, concurrency = entry["strategy"], entry["concurrency"]
    if strategy not in config.strategies or concurrency not in config.concurrency:
        raise ValueError("Plan selects a strategy/concurrency outside the source configuration")
    server = config.server_for(strategy).model_dump()
    # The official CUDA 12.9 build of release 0.29.0 is versioned 0.29.0+cu129.
    # Accept that specific variant; retain its full version in the task config.
    if server["vllm_version"] != version and (server["vllm_version"], version) != (
            "0.29.0", "0.29.0+cu129"):
        raise ValueError(f"Installed vLLM {version} differs from configured {server['vllm_version']}")
    cached = strategy == "prefix_cached_full_context"
    command = server_command(server["launch_command"], model=server["model"],
                             revision=server["model_revision"], cached=cached,
                             gpu_memory_utilization=float(os.environ.get(
                                 "CONTEXTBENCH_GPU_MEMORY_UTILIZATION", "0.92")))
    server.update(base_url="http://127.0.0.1:8000/v1", metrics_url="http://127.0.0.1:8000/metrics",
                  version_url="http://127.0.0.1:8000/version", prefix_caching=cached,
                  timeout_s=positive_seconds("CONTEXTBENCH_REQUEST_TIMEOUT_SECONDS",
                                             server["timeout_s"]),
                  vllm_version=version, launch_command=shlex.join(command))
    values = config.model_dump()
    values.update(server=server, strategy_servers={}, strategies=[strategy],
                  concurrency=[concurrency], phase="cached" if cached else "uncached",
                  cache_scenario=(config.cache_scenario or "warm") if cached else None,
                  output_dir=str((output / "runs").resolve()), local_gpu_metrics=True)
    if "scaling_size" in entry:
        from scripts.generate_scaling_workloads import TARGETS, generate

        if entry["scaling_size"] not in TARGETS:
            raise ValueError("Unknown scaling size")
        fixtures = output / "scaling"
        generate(root / "data/enterprise", fixtures)
        values["dataset"] = str((fixtures / entry["scaling_size"]).resolve())
        values["experiment_id"] = f"pilot-scaling-{entry['scaling_size']}"
    return ExperimentConfig.model_validate(values)


class GCSArtifacts:
    def __init__(self, bucket: str, prefix: str, *, deadline: float | None = None):
        from google.cloud import storage

        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]+[a-z0-9]", bucket):
            raise ValueError("RESULTS_BUCKET must be a bucket name without gs://")
        self.bucket = storage.Client().bucket(bucket)
        self.prefix = prefix.strip("/")
        self.previous = {}
        self.deadline = deadline or time.monotonic() + 3540

    def request_options(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TaskFailure("Artifact upload deadline exceeded")
        # No unbounded SDK retries: Cloud Run can retry the isolated task attempt.
        return {"timeout": (min(5, remaining / 2), min(20, remaining / 2)), "retry": None}

    def put_json(self, name: str, value: dict):
        self.bucket.blob(f"{self.prefix}/{name}").upload_from_string(
            json.dumps(value, indent=2), content_type="application/json", **self.request_options())

    def probe(self):
        self.put_json("access-check.json", {"status": "storage-write-verified"})

    def sync(self, root: Path):
        # Snapshot changing files so upload checksums apply to stable contents.
        with tempfile.TemporaryDirectory(prefix="contextbench-upload-") as temporary:
            for source in sorted(root.rglob("*")):
                if not source.is_file() or source.suffix == ".tmp":
                    continue
                relative = source.relative_to(root).as_posix()
                stat = source.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
                if self.previous.get(relative) == signature:
                    continue
                snapshot = Path(temporary) / "snapshot"
                shutil.copyfile(source, snapshot)
                self.bucket.blob(f"{self.prefix}/{relative}").upload_from_filename(
                    str(snapshot), **self.request_options())
                self.previous[relative] = signature


class Control:
    def __init__(self, budget: float, stopped: threading.Event | None = None):
        self.started = time.monotonic()
        self.deadline = self.started + budget
        self.stopped = stopped or threading.Event()

    def check(self):
        if self.stopped.is_set():
            raise TaskFailure("Task interrupted by shutdown signal")
        if time.monotonic() >= self.deadline:
            raise TaskFailure("Internal task deadline exceeded; retaining partial artifacts")


def terminate(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    except ProcessLookupError:
        pass


def wait_ready(process, config: ExperimentConfig, control: Control, tick, timeout: float):
    deadline = time.monotonic() + timeout
    key = os.environ.get(config.server.api_key_env)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    with httpx.Client(timeout=2, trust_env=False) as client:
        while True:
            control.check()
            if process.poll() is not None:
                raise TaskFailure(f"vLLM exited before readiness (exit {process.returncode})")
            try:
                response = client.get(config.server.base_url + "/models", headers=headers)
                response.raise_for_status()
                if config.server.model in [model["id"] for model in response.json()["data"]]:
                    return
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                pass
            if time.monotonic() >= deadline:
                raise TaskFailure("vLLM did not become ready before the model startup deadline")
            tick()
            control.stopped.wait(1)


def wait_workload(process, server, control: Control, tick) -> int:
    while process.poll() is None:
        control.check()
        if server.poll() is not None:
            raise TaskFailure(f"vLLM exited during the workload (exit {server.returncode})")
        tick()
        control.stopped.wait(1)
    if process.returncode:
        raise TaskFailure(f"Benchmark process exited with status {process.returncode}")
    return process.returncode


def wait_probe(process, control: Control, tick, timeout: float):
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        control.check()
        if time.monotonic() >= deadline:
            raise TaskFailure("GPU preflight exceeded its startup deadline")
        tick()
        control.stopped.wait(1)
    control.check()
    if process.returncode:
        raise TaskFailure(f"GPU preflight failed (exit {process.returncode}); see gpu-preflight.log")


def verify_results(config: ExperimentConfig, *, require_correct: bool):
    roots = list(Path(config.output_dir).glob("*/metadata.json"))
    if len(roots) != 1:
        raise TaskFailure("Expected exactly one current benchmark run")
    root = roots[0].parent
    metadata = json.loads(roots[0].read_text())
    summaries = json.loads((root / "summary.json").read_text())
    events = json.loads((Path(config.dataset) / "events.json").read_text())
    count = sum(config.event_ids is None or event["event_id"] in config.event_ids
                for event in events) * config.repetitions
    if count <= 0 or len(summaries) != 1:
        raise TaskFailure("Benchmark did not complete exactly one nonempty cell")
    summary = summaries[0]
    if metadata.get("status") != "completed" or summary.get("status") != "completed":
        raise TaskFailure(
            f"Benchmark status {metadata.get('status')!r}; cell status "
            f"{summary.get('status')!r}; successful={summary.get('successful_requests')}, "
            f"failed={summary.get('failed_requests')}, "
            f"warmup_failures={summary.get('warmup_failures')}. "
            "See requests.jsonl, warmup.jsonl and errors.jsonl for details")
    if summary.get("requests") != count or summary.get("successful_requests") != count:
        raise TaskFailure("Benchmark has failures, missing requests, or incomplete summary")
    records = [json.loads(line) for line in
               (root / f"{config.strategies[0]}-c{config.concurrency[0]}" / "requests.jsonl")
               .read_text().splitlines()]
    if len(records) != count or any(row.get("error") for row in records):
        raise TaskFailure("Raw request outcomes do not match the completed cell")
    if require_correct and not all(row.get("correct") is True for row in records):
        raise TaskFailure("Strict smoke/pilot quality check failed")


def execute(entry: dict, config: ExperimentConfig, output: Path, root: Path,
            store, control: Control, *, ready_timeout=900, upload_interval=30,
            probe_timeout=120) -> int:
    probe = server = workload = None
    logs = []
    result = {"cell": entry["id"], "status": "running"}
    last_upload = -math.inf

    def write_status():
        (output / "task-status.json").write_text(json.dumps(result, indent=2) + "\n")

    def tick():
        nonlocal last_upload
        if time.monotonic() - last_upload >= upload_interval:
            store.sync(output)
            last_upload = time.monotonic()

    output.mkdir(parents=True, exist_ok=True)
    store.deadline = control.deadline
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
    write_status()
    try:
        control.check()
        store.probe()  # Fail before GPU/model initialization if bucket access is unavailable.
        tick()
        probe_log = (output / "gpu-preflight.log").open("wb")
        logs.append(probe_log)
        probe = subprocess.Popen([sys.executable, "-m", "deploy.gcp.gpu_probe"], cwd=root,
                                 stdout=probe_log, stderr=subprocess.STDOUT, start_new_session=True)
        wait_probe(probe, control, tick, probe_timeout)
        print("GPU preflight passed; starting vLLM.", flush=True)
        server_log = (output / "vllm.log").open("wb")
        logs.append(server_log)
        server = subprocess.Popen(shlex.split(config.server.launch_command), cwd=root,
                                  stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        wait_ready(server, config, control, tick, ready_timeout)
        print("vLLM is ready; running the selected benchmark cell.", flush=True)
        control.check()
        benchmark_log = (output / "benchmark.log").open("wb")
        logs.append(benchmark_log)
        workload = subprocess.Popen([sys.executable, "scripts/run_benchmark.py", str(config_path)],
                                    cwd=root, stdout=benchmark_log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        wait_workload(workload, server, control, tick)
        control.check()
        verify_results(config, require_correct=entry.get("require_correct",
                       Path(entry["config"]).name == "smoke.yaml"))
        result["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 - all task failures must retain cleanup and artifacts.
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(result["error"], file=sys.stderr, flush=True)
    finally:
        terminate(workload)
        terminate(server)
        terminate(probe)
        for log in logs:
            log.close()
        if control.stopped.is_set():
            result.update(status="failed", error="Task interrupted by shutdown signal")
        if result["status"] != "completed":
            for name in ("gpu-preflight.log", "vllm.log", "benchmark.log"):
                path = output / name
                if path.exists():
                    # Keep failure diagnostics in Cloud Logging as well as GCS.
                    with path.open("rb") as handle:
                        handle.seek(max(0, path.stat().st_size - 16000))
                        tail = handle.read().decode("utf-8", errors="replace")
                    print(f"Last output from {name}:\n{tail}", file=sys.stderr, flush=True)
        result["elapsed_s"] = time.monotonic() - control.started
        write_status()
        try:
            store.deadline = min(control.started + 3540, time.monotonic() + 120)
            store.sync(output)
            if control.stopped.is_set():
                raise TaskFailure("Task interrupted during final artifact upload")
            if result["status"] == "completed":
                store.put_json("complete.json", result)
        except Exception as exc:  # noqa: BLE001 - an SDK failure must make the task fail.
            result.update(status="failed", error=f"Artifact persistence failed: {type(exc).__name__}")
            write_status()
            print(result["error"], file=sys.stderr, flush=True)
            # A previous sync may already have uploaded a completed status. Correct
            # it when the connection still works; complete.json remains authoritative.
            try:
                store.put_json("task-status.json", result)
            except Exception:  # noqa: BLE001 - preserve the original failure and deadline.
                print("Could not update remote task status after the upload failure.",
                      file=sys.stderr, flush=True)
    if result["status"] == "completed":
        print("Task completed; verified artifacts uploaded.", flush=True)
    return 0 if result["status"] == "completed" else 1


def positive_seconds(name: str, default: float, *, maximum=3300) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f"{name} must be positive and no greater than {maximum}")
    return value


def main() -> int:
    # Start the task clock before planning, fixture generation, or authentication.
    control = Control(positive_seconds("TASK_DEADLINE_SECONDS", 3300))
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, lambda *_: control.stopped.set())
    try:
        root = Path.cwd()
        index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
        count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1"))
        entry = read_plan(Path(os.environ.get("TASK_PLAN", "/app/deploy/gcp/smoke-plan.json")),
                          index, count, cell_id=os.environ.get("TASK_CELL_ID") or None)
        execution = os.environ.get("CLOUD_RUN_EXECUTION", "local")
        attempt = os.environ.get("CLOUD_RUN_TASK_ATTEMPT", "0")
        for value in (execution, attempt):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
                raise ValueError("Invalid execution/attempt identifier")
        unique = f"{execution}/task-{index}-{entry['id']}/attempt-{attempt}-{uuid.uuid4().hex}"
        output = Path(os.environ.get("RESULT_ROOT", "/tmp/contextbench-task")) / unique
        prefix = f"{os.environ.get('RESULTS_PREFIX', 'cloud-run').strip('/')}/{unique}"
        bucket = os.environ.get("RESULTS_BUCKET", "context-inference-lab")
        print(f"Task {index + 1}/{count}: {entry['id']}; artifacts: gs://{bucket}/{prefix}/",
              flush=True)
        store = GCSArtifacts(bucket, prefix,
                             deadline=control.deadline)
        version = importlib.metadata.version("vllm")
        # Fixture generation may take time; the same task deadline remains in force.
        config = cell_config(entry, root, output, version)
        return execute(entry, config, output, root, store, control,
                       ready_timeout=positive_seconds("MODEL_READY_TIMEOUT_SECONDS", 900),
                       upload_interval=positive_seconds("UPLOAD_INTERVAL_SECONDS", 30))
    except Exception as exc:  # noqa: BLE001 - process entrypoint reports setup failures without success.
        print(f"Task setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
