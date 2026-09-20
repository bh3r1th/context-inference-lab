"""One synthetic event over real local HTTP; fixture responses are not GPU evidence."""
import asyncio
import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from contextbench.experiments.config import ExperimentConfig
from contextbench.experiments.orchestrator import run


def test_one_event_vllm_smoke(dataset, tmp_path):
    answer = json.loads((dataset / "expected.json").read_text())["EV-000"]["output"]

    class Handler(BaseHTTPRequestHandler):
        requests_seen = 0

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            if self.path == "/v1/models":
                body = '{"data":[{"id":"smoke-fixture"}]}'
            elif self.path == "/version":
                body = '{"version":"fixture-not-real-vllm"}'
            else:
                body = ('vllm:kv_cache_usage_perc{engine="0"} 0.25\n'
                        f'vllm:prefix_cache_hits_total{{engine="0"}} {self.requests_seen * 10}\n'
                        f'vllm:prefix_cache_queries_total{{engine="0"}} {self.requests_seen * 20}\n')
            self.wfile.write(body.encode())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert payload["model"] == "smoke-fixture" and payload["temperature"] == 0
            assert payload["max_tokens"] == 128 and "structured_outputs" in payload
            assert json.loads(payload["messages"][-1]["content"])["event_id"] == "EV-000"
            type(self).requests_seen += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in [
                {"id": "smoke-1", "model": "smoke-fixture", "choices": [
                    {"delta": {"content": json.dumps(answer)}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 11,
                 "prompt_tokens_details": {"cached_tokens": 10}},
                 "metrics": {"queue_time_ms": 1, "generation_time_ms": 100,
                             "time_to_first_token_ms": 10, "tokens_per_second": 100}},
            ]:
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        config = ExperimentConfig.model_validate({
            "dataset": str(dataset), "output_dir": str(tmp_path / "results"), "event_ids": ["EV-000"],
            "strategies": ["full_context"], "concurrency": [1], "repetitions": 1, "warmup_requests": 0,
            "server": {"base_url": f"http://127.0.0.1:{port}/v1",
                       "metrics_url": f"http://127.0.0.1:{port}/metrics", "model": "smoke-fixture",
                       "model_revision": "fixture", "vllm_version": "fixture-not-real-vllm",
                       "launch_command": "test fixture HTTP server", "max_tokens": 128}})
        output = asyncio.run(run(config))
        rows = (output / "full_context-c1/requests.jsonl").read_text().splitlines()
        assert len(rows) == Handler.requests_seen == 1
        request = json.loads(rows[0])
        assert request["correct"] and request["raw_usage"]["completion_tokens"] == 11
        assert request["decode_time_s"] == .1 and request["server_queue_s"] == .001
        assert request["gpu_memory_used_mib"] is None and request["unsupported"]["gpu_memory_used_mib"]
        assert len(request["stream_events"]) == 3
        with (output / "full_context-c1/requests.csv").open(newline="") as handle:
            assert next(csv.DictReader(handle))["prefill_s"] == "null"
        telemetry = [json.loads(line) for line in
                     (output / "full_context-c1/telemetry.jsonl").read_text().splitlines()]
        assert telemetry[0]["phase"] == "before_workload"
        assert telemetry[-1]["phase"] == "after_workload"
        metadata = json.loads((output / "metadata.json").read_text())
        assert metadata["status"] == "completed"
        assert metadata["vllm_version_observed"]["value"] == "fixture-not-real-vllm"
        assert metadata["started_at"] <= metadata["completed_at"]
        assert metadata["gpu_metadata"]["devices"] is None
        summary = json.loads((output / "summary.json").read_text())[0]
        assert summary["requests_per_s"] > 0 and summary["throughput"]["output_tokens_per_s"]["value"] > 0
        intervals = summary["telemetry"]["prefix_cache"]["counter_intervals"]
        assert next(i for i in intervals if "hits" in i["metric"])["hit_ratio"] == .5
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
