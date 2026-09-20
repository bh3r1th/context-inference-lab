"""Real local HTTP/SSE integration; fixture replies are not model inference evidence."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from contextbench.experiments.config import ExperimentConfig
from contextbench.experiments.orchestrator import run


def test_http_end_to_end(dataset, tmp_path):
    expected = json.loads((dataset / "expected.json").read_text())
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            if self.path == "/v1/models":
                self.wfile.write(b'{"data":[{"id":"fixture-model"}]}')
            else:
                self.wfile.write(b"vllm:kv_cache_usage_perc 0.25\n")

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            event = json.loads(payload["messages"][-1]["content"])
            answer = json.dumps(expected[event["event_id"]]["output"])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in [
                {"id": "fixture", "choices": [{"delta": {"content": answer}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 30}},
            ]:
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        config = ExperimentConfig.model_validate({
            "dataset": str(dataset), "output_dir": str(tmp_path / "results"),
            "strategies": ["full_context", "retrieval", "compiled_context"],
            "concurrency": [1, 3], "repetitions": 1, "warmup_requests": 0,
            "server": {"base_url": f"http://127.0.0.1:{port}/v1",
                       "metrics_url": f"http://127.0.0.1:{port}/metrics",
                       "model": "fixture-model", "model_revision": "test-only",
                       "vllm_version": "fixture", "launch_command": "fixture HTTP server"}})
        output = asyncio.run(run(config))
        summaries = json.loads((output / "summary.json").read_text())
        assert len(summaries) == 6
        assert all(s["correctness"] == 1 for s in summaries)
        assert all(s["requests"] == 5 for s in summaries)
        assert len(list(output.glob("*/requests.csv"))) == 6
        records = (output / "full_context-c1/requests.jsonl").read_text().splitlines()
        assert len(records) == 5
        assert json.loads(records[0])["prefill_s"] is None
        assert json.loads((output / "metadata.json").read_text())["status"] == "completed"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
