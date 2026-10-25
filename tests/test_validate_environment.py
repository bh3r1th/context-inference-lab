"""Exercise validator authentication over HTTP, including the keyless GPU job."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from contextbench.experiments.config import load_config
from scripts import validate_environment as validator

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def validate_endpoint(monkeypatch, tmp_path):
    config = load_config(ROOT / "configs/smoke.yaml")
    config.output_dir = str(tmp_path / "results")
    config.server.api_key_env = "CONTEXTBENCH_TEST_API_KEY"
    observed = []
    reply = {"status": 200, "model": config.server.model}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            observed.append((self.path, self.headers.get("Authorization")))
            self.send_response(reply["status"])
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": reply["model"]}]}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config.server.base_url = f"http://127.0.0.1:{server.server_port}/v1"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump()))
    monkeypatch.setattr(sys, "argv", ["validate_environment.py", str(path)])
    monkeypatch.setattr(validator.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def query_gpu(command, *, text):
        assert command == ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                           "--format=csv,noheader"]
        assert text is True
        return "NVIDIA L4, 580.126.20, 23034 MiB\n"

    monkeypatch.setattr(validator.subprocess, "check_output", query_gpu)
    monkeypatch.delenv(config.server.api_key_env, raising=False)
    try:
        yield observed, reply, config.server.api_key_env
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("key", [None, "", "configured-test-key"])
def test_validator_omits_empty_auth_and_preserves_configured_auth(
        validate_endpoint, monkeypatch, capsys, key):
    observed, _, variable = validate_endpoint
    if key is not None:
        monkeypatch.setenv(variable, key)
    validator.main()
    assert observed == [("/v1/models", f"Bearer {key}" if key else None)]
    assert json.loads(capsys.readouterr().out)["status"] == "ready"


@pytest.mark.parametrize("status,model,error", [
    (401, "unused", "health/model check failed"),
    (503, "unused", "health/model check failed"),
    (200, "wrong-model", "is not served"),
])
def test_keyless_validation_still_rejects_bad_endpoints(
        validate_endpoint, capsys, status, model, error):
    observed, reply, _ = validate_endpoint
    reply.update(status=status, model=model)
    with pytest.raises(SystemExit, match=error):
        validator.main()
    assert observed == [("/v1/models", None)]
    assert "ready" not in capsys.readouterr().out
