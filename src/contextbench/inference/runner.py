"""Bounded request workers; pipeline failures retain all available raw measurements."""
import asyncio
import hashlib
from dataclasses import asdict
from time import perf_counter

from contextbench.evaluation.scoring import evaluate
from contextbench.metrics.provenance import finalize_request_metrics, utc_now
from contextbench.models import ContextStrategy, InferenceBackend, InferenceResult


async def run_batch(backend: InferenceBackend, strategy: ContextStrategy, events: list[dict],
                    expected: dict, concurrency: int, seed: int, emit) -> tuple[list[dict], float]:
    """Fixed admission order and request seeds; completion order can vary with concurrency."""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    enqueued_at = utc_now()
    start = perf_counter()
    queue = asyncio.Queue()
    for i, event in enumerate(events):
        queue.put_nowait((i, event))
    records = []

    async def worker():
        while True:
            try:
                index, event = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            queue_s = perf_counter() - start
            context_start = perf_counter()
            context, stage, error_stage = None, "context", None
            attempted_inference = False
            result = InferenceResult()
            try:
                context = strategy.build(event)
            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                error_stage = stage
            context_s = perf_counter() - context_start
            if context is not None:
                stage = "inference"
                attempted_inference = True
                try:
                    result = await backend.generate(context, event, seed + index)
                except Exception as exc:
                    result.error = f"{type(exc).__name__}: {exc}"
                if result.error:
                    error_stage = stage
            if not result.metric_sources:
                finalize_request_metrics(result)
            truth = expected[event["event_id"]]
            try:
                evaluation = evaluate(result.text, truth["output"], truth["critical_rules"])
            except Exception as exc:
                error_stage = error_stage or "evaluation"
                error = f"evaluation {type(exc).__name__}: {exc}"
                result.error = f"{result.error}; {error}" if result.error else error
                evaluation = {"correct": False, "schema_valid": False, "rule_recall": 0.0,
                              "missed_critical_rules": truth["critical_rules"], "validation_error": error}
            required_context_sources = set(truth.get("required_source_ids",
                                                    truth["output"]["applied_rules"]))
            selected_context_sources = set(context.source_ids) if context else set()
            evaluation["missing_required_context_sources"] = sorted(
                required_context_sources - selected_context_sources)
            evaluation["minimum_safe_context"] = not evaluation["missing_required_context_sources"]
            case = truth["case"]
            evaluation["cross_document_reasoning_success"] = (
                evaluation["correct"] if case == "cross-document" else None)
            evaluation["buried_exception_success"] = (
                evaluation["correct"] if case == "buried-exception" else None)
            if result.error:
                evaluation["correct"] = False
            row = {"request_index": index, "event_id": event["event_id"],
                   "seed": seed + index, "enqueued_at": enqueued_at,
                   "case": truth["case"], "client_queue_s": queue_s,
                   "client_queue_source": "client perf_counter: batch admission to worker dispatch",
                   "error_stage": error_stage, "inference_attempted": attempted_inference,
                   "context_build_s": context_s,
                   "context_chars": len(context.text) if context else None,
                   "context_bytes": len(context.text.encode()) if context else None,
                   "context_sha256": hashlib.sha256(context.text.encode()).hexdigest() if context else None,
                   "prompt_sha256": hashlib.sha256(context.prompt.encode()).hexdigest() if context else None,
                   "source_ids": list(context.source_ids) if context else [],
                   "context_token_count": context.token_count if context else None,
                   "context_metadata": context.metadata if context else None,
                   "worker_started_s": queue_s,
                   "batch_response_s": perf_counter() - start,
                   **asdict(result), **evaluation}
            # Persistence errors are fatal: never keep sending requests after the raw log fails.
            emit(row)
            records.append(row)
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*workers)
    finally:
        for task in workers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    return records, perf_counter() - start
