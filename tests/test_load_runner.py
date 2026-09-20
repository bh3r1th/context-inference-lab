import asyncio
import json

from contextbench.context import make_strategy
from contextbench.inference.runner import run_batch
from contextbench.models import InferenceResult


def test_concurrency_is_bounded_and_all_requests_recorded(dataset):
    events = json.loads((dataset / "events.json").read_text())
    truth = json.loads((dataset / "expected.json").read_text())
    class Backend:
        active = 0
        peak = 0

        async def generate(self, context, event, seed):
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return InferenceResult(text=json.dumps(truth[event["event_id"]]["output"]))
    backend = Backend()
    emitted = []
    rows, duration = asyncio.run(run_batch(backend, make_strategy("full_context", dataset),
                                           events * 3, truth, 3, 42, emitted.append))
    assert backend.peak == 3
    assert len(rows) == len(emitted) == 15
    assert sorted(r["request_index"] for r in rows) == list(range(15))
    assert all(r["client_queue_s"] >= 0 for r in rows)
    assert all(r["context_token_count"] > 0 for r in rows)
    assert all(r["context_metadata"]["tokenizer"]["is_estimate"] for r in rows)
    assert duration > 0


def test_backend_exception_keeps_failed_request_record(dataset):
    events = json.loads((dataset / "events.json").read_text())[:1]
    truth = json.loads((dataset / "expected.json").read_text())

    class Backend:
        async def generate(self, context, event, seed):
            raise RuntimeError("transport adapter failed")

    emitted = []
    rows, _ = asyncio.run(run_batch(Backend(), make_strategy("full_context", dataset),
                                   events, truth, 1, 42, emitted.append))
    assert len(rows) == len(emitted) == 1
    assert "RuntimeError" in rows[0]["error"] and not rows[0]["correct"]
    assert rows[0]["input_tokens"] is None and rows[0]["unsupported"]["input_tokens"]
