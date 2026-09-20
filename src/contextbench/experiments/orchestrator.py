"""Reproducible matrix execution with append-only request logs and isolated cell failures."""
import asyncio
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import httpx

from contextbench.context import make_strategy
from contextbench.experiments.config import ExperimentConfig
from contextbench.inference.runner import run_batch
from contextbench.inference.vllm import VLLMBackend
from contextbench.metrics.gpu import sample_gpu
from contextbench.metrics.provenance import unavailable, utc_now
from contextbench.metrics.report import write_report
from contextbench.metrics.summary import summarize
from contextbench.metrics.telemetry import collect, sample_telemetry, summarize_telemetry
from contextbench.models import Decision


def write_json(path: Path, value):
    # Replace summaries atomically; raw logs are never rewritten.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(row.get(k), allow_nan=False)
                          if row.get(k) is None or isinstance(row.get(k), (list, dict)) else row[k]
                          for k in fields} for row in rows)
    temporary.replace(path)


def summary_csv_row(summary: dict) -> dict:
    fields = ("experiment_id", "run_id", "cell_id", "strategy", "concurrency", "status",
              "server_url", "prefix_caching", "started_at", "completed_at", "planned_requests",
              "requests", "unattempted_requests", "successful_requests", "failed_requests",
              "failure_rate", "warmup_requests", "warmup_failures", "batch_duration_s",
              "preparation_s", "requests_per_s", "attempts_per_s", "average_input_tokens",
              "average_output_tokens", "correctness", "schema_validity", "quality",
              "minimum_safe_context", "errors")
    row = {field: summary.get(field) for field in fields}
    row["quality"] = summary.get("quality", {}).get("correct")
    row["minimum_safe_context"] = summary.get("quality", {}).get("minimum_safe_context")
    for name in ("ttft_s", "latency_s"):
        for percentile in ("p50", "p95", "p99", "count"):
            row[f"{name}_{percentile}"] = summary["successful_request_metrics"][name][percentile]
    for name in ("input_tokens", "output_tokens"):
        row[f"{name}_per_s"] = summary["throughput"][f"{name}_per_s"]["value"]
        row[f"{name}_observed_requests"] = summary["token_totals"][name]["observed_requests"]
    return row


def build_workload(events: list[dict], repetitions: int, seed: int) -> tuple[list[dict], list[dict]]:
    plan = [{"event_index": i, "event_id": event["event_id"], "repetition": repetition}
            for repetition in range(repetitions) for i, event in enumerate(events)]
    random.Random(seed).shuffle(plan)
    for index, entry in enumerate(plan):
        entry.update(request_index=index, seed=seed + index)
    return [events[entry["event_index"]] for entry in plan], plan


async def run(config: ExperimentConfig) -> Path:
    dataset = Path(config.dataset)
    events = json.loads((dataset / "events.json").read_text(encoding="utf-8"))
    expected = json.loads((dataset / "expected.json").read_text(encoding="utf-8"))
    if not events or len({e["event_id"] for e in events}) != len(events):
        raise ValueError("events must be nonempty and have unique IDs")
    if config.event_ids is not None:
        missing = set(config.event_ids) - {e["event_id"] for e in events}
        if missing:
            raise ValueError(f"unknown event_ids: {sorted(missing)}")
        events = [e for e in events if e["event_id"] in config.event_ids]
    for event in events:
        truth = expected[event["event_id"]]
        answer = Decision.model_validate(truth["output"])
        if answer.event_id != event["event_id"]:
            raise ValueError("answer key event_id mismatch")
        if not set(truth["critical_rules"]) <= set(answer.applied_rules):
            raise ValueError("critical rules must be a subset of expected applied rules")

    experiment_id = config.experiment_id or "experiment-" + uuid.uuid4().hex
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    output = Path(config.output_dir) / run_id
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(dataset, output / "dataset")
    hashes = {str(p.relative_to(dataset)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(dataset.rglob("*")) if p.is_file()}
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(source_root.rglob("*.py"))}
    metadata = {"experiment_id": experiment_id, "run_id": run_id, "config": config.model_dump(),
                "dataset_sha256": hashes, "client_source_sha256": source_hashes,
                "started_at": utc_now(), "completed_at": None, "status": "running",
                "model": config.server.model, "model_revision": config.server.model_revision,
                "prefix_caching": config.server.prefix_caching,
                "launch_command": config.server.launch_command,
                "vllm_version_declared": config.server.vllm_version,
                "vllm_version_source": "operator-declared configuration",
                "vllm_version_observed": unavailable("server /version", "not yet discovered", "server"),
                "gpu_metadata": {**unavailable("nvidia-smi", "not yet discovered", "local_host"),
                                 "devices": None},
                "benchmark_request_parameters": {"temperature": 0, "n": 1,
                                                 "max_tokens": config.server.max_tokens,
                                                 "structured_output": config.server.structured_output,
                                                 "structured_output_mode": config.server.structured_output_mode},
                "phase": config.phase,
                "cache_scenario": config.cache_scenario,
                "prefix_cache_contract": {
                    "static_prefix": "system message: instructions plus documents",
                    "dynamic_suffix": "user message: canonical event JSON",
                    "identity_evidence": "per-request static_prefix_sha256 and token count",
                    "cache_hit_evidence": "server usage cached_tokens or Prometheus counters only",
                },
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {p: importlib.metadata.version(p) for p in ("httpx", "pydantic", "PyYAML")},
                "server_configuration_source": "operator-declared; verify against server launch logs",
                "servers": {}, "error_count": 0,
                "planned_cells": len(config.strategies) * len(config.concurrency)}
    try:
        metadata["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=source_root.parent.parent).strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["git_sha"] = None
    write_json(output / "metadata.json", metadata)
    write_json(output / "output.schema.json", Decision.model_json_schema())
    workload, plan = build_workload(events, config.repetitions, config.seed)
    write_json(output / "workload.json", workload)
    write_json(output / "workload_manifest.json", plan)
    summaries = []

    def checkpoint():
        write_json(output / "summary.json", summaries)
        write_csv(output / "summary.csv", [summary_csv_row(s) for s in summaries])
        write_json(output / "metadata.json", metadata)

    try:
        limits = httpx.Limits(max_connections=max(config.concurrency) + 4,
                              max_keepalive_connections=max(config.concurrency) + 4)
        with (output / "errors.jsonl").open("x", encoding="utf-8") as error_log:
            def record_error(phase, error, cell_id=None):
                record = {"experiment_id": experiment_id, "run_id": run_id, "cell_id": cell_id,
                          "timestamp": utc_now(), "phase": phase, "error": str(error)}
                error_log.write(json.dumps(record, allow_nan=False) + "\n")
                error_log.flush()
                metadata["error_count"] += 1
                return record

            async with httpx.AsyncClient(timeout=config.server.timeout_s, limits=limits) as client:
                try:
                    metadata["gpu_metadata"] = await sample_gpu() if config.local_gpu_metrics else {
                        **unavailable("nvidia-smi", "local_gpu_metrics disabled; serving GPUs not observed",
                                      "local_host"), "devices": None}
                except Exception as exc:
                    record_error("gpu_discovery", exc)
                    metadata["gpu_metadata"] = {**unavailable("nvidia-smi", str(exc), "local_host"),
                                                "devices": None}
                for name in config.strategies:
                    server = config.server_for(name)
                    backend = VLLMBackend(server, client)
                    strategy, preparation_error = None, None
                    prepare_start = perf_counter()
                    try:
                        if any("REPLACE_WITH" in v for v in (server.model_revision, server.vllm_version,
                                                              server.launch_command)):
                            raise ValueError("record actual model revision, vLLM version and launch command")
                        if name == "prefix_cached_full_context" and not server.prefix_caching:
                            raise ValueError("cached strategy requires a declared cache-enabled server; "
                                             "configure strategy_servers.prefix_cached_full_context")
                        strategy = make_strategy(name, dataset, config=config.context)
                    except Exception as exc:
                        preparation_error = f"{type(exc).__name__}: {exc}"
                    preparation_s = perf_counter() - prepare_start
                    server_info = {"config": server.model_dump(), "models": None,
                                   "version": unavailable("server /version", "setup did not complete", "server")}
                    if strategy is not None:
                        try:
                            server_info["models"] = await backend.metadata()
                        except Exception as exc:
                            server_info["models_error"] = str(exc)
                            record_error("server_discovery", exc)
                        try:
                            server_info["version"] = await backend.version()
                        except Exception as exc:
                            record_error("version_discovery", exc)
                            server_info["version"] = unavailable("server /version", str(exc), "server")
                    metadata["servers"][name] = server_info
                    if server == config.server:
                        metadata["served_models"] = server_info["models"]
                        metadata["vllm_version_observed"] = server_info["version"]
                    for concurrency in config.concurrency:
                        cell_id = f"{run_id}/{name}-c{concurrency}"
                        cell = output / f"{name}-c{concurrency}"
                        cell.mkdir()
                        rows, warmups, samples, errors = [], [], [], []
                        started_at, duration, batch_start = utc_now(), None, None
                        interrupted = False
                        with (cell / "requests.jsonl").open("x", encoding="utf-8") as raw, \
                             (cell / "warmup.jsonl").open("x", encoding="utf-8") as warm_raw, \
                             (cell / "telemetry.jsonl").open("x", encoding="utf-8") as telemetry:
                            def emit(row, *, warmup=False):
                                phase = "warmup" if warmup else "measurement"
                                row.update(experiment_id=experiment_id, run_id=run_id, cell_id=cell_id,
                                           request_id=f"{cell_id}/{phase}/{row['request_index']}",
                                           phase=phase, strategy=name, concurrency=concurrency,
                                           server_url=server.base_url, prefix_caching=server.prefix_caching)
                                row["repetition"] = None if warmup else plan[row["request_index"]]["repetition"]
                                handle = warm_raw if warmup else raw
                                handle.write(json.dumps(row, allow_nan=False) + "\n")
                                handle.flush()
                                (warmups if warmup else rows).append(row)

                            def emit_telemetry(row):
                                telemetry.write(json.dumps(row, allow_nan=False) + "\n")
                                telemetry.flush()
                                samples.append(row)

                            async def snapshot(phase):
                                try:
                                    value = await sample_telemetry(client, server.metrics_url,
                                                                   backend.headers, config.local_gpu_metrics)
                                except Exception as exc:
                                    value = {"timestamp": utc_now(), "error": str(exc)}
                                    errors.append(record_error("telemetry", exc, cell_id))
                                emit_telemetry({**value, "phase": phase})

                            try:
                                if preparation_error:
                                    raise ValueError(preparation_error)
                                warm_events = [workload[i % len(workload)] for i in range(config.warmup_requests)]
                                await run_batch(backend, strategy, warm_events, expected, concurrency,
                                                config.seed, lambda row: emit(row, warmup=True))
                                await snapshot("before_workload")
                                stop = asyncio.Event()
                                task = asyncio.create_task(collect(
                                    client, server.metrics_url, backend.headers, config.local_gpu_metrics,
                                    config.telemetry_interval_s, stop, emit_telemetry))
                                try:
                                    batch_start = perf_counter()
                                    _, duration = await run_batch(backend, strategy, workload, expected,
                                                                  concurrency, config.seed, emit)
                                finally:
                                    if duration is None and batch_start is not None:
                                        duration = perf_counter() - batch_start
                                    stop.set()
                                    try:
                                        await task
                                    except OSError:
                                        raise
                                    except Exception as exc:
                                        errors.append(record_error("telemetry", exc, cell_id))
                                    await snapshot("after_workload")
                            except OSError:
                                # Persistence failures cannot be isolated safely: retain prior logs and stop.
                                raise
                            except Exception as exc:
                                errors.append(record_error("setup" if batch_start is None else "batch", exc, cell_id))
                            except BaseException:
                                interrupted = True
                                raise
                            finally:
                                warmup_failures = sum(r["error"] is not None for r in warmups)
                                failed_requests = sum(r["error"] is not None for r in rows)
                                status = ("interrupted" if interrupted else "failed" if duration is None
                                          else "completed_with_errors" if errors or warmup_failures or failed_requests
                                          or len(rows) != len(workload)
                                          else "completed")
                                summary = {"experiment_id": experiment_id, "run_id": run_id, "cell_id": cell_id,
                                           "strategy": name, "concurrency": concurrency, "status": status,
                                           "server_url": server.base_url, "prefix_caching": server.prefix_caching,
                                           "started_at": started_at, "completed_at": utc_now(),
                                           "planned_requests": len(workload), "unattempted_requests": len(workload) - len(rows),
                                           "preparation_s": preparation_s, "warmup_requests": len(warmups),
                                           "warmup_failures": warmup_failures, "errors": errors,
                                           "telemetry": summarize_telemetry(samples), **summarize(rows, duration)}
                                summaries.append(summary)
                                write_json(cell / "warmup.json", warmups)
                                write_csv(cell / "requests.csv", sorted(rows, key=lambda r: r["request_index"]))
                                write_json(cell / "summary.json", summary)
                                checkpoint()
                metadata["status"] = ("completed" if metadata["error_count"] == 0 and
                                      all(s["status"] == "completed" for s in summaries)
                                      else "completed_with_errors")
                write_report(output, summaries, metadata)
    except BaseException as exc:
        metadata["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["completed_at"] = utc_now()
        metadata["completed_cells"] = len(summaries)
        checkpoint()
    return output
