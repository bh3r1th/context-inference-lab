"""Fail-fast validation for a GPU-backed benchmark run."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import httpx

from contextbench.experiments.config import load_config


def fail(message):
    raise SystemExit(f"Environment validation failed: {message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    if shutil.which("nvidia-smi") is None:
        fail("nvidia-smi is not available; use a CUDA GPU host")
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                       "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"GPU query failed: {exc}")
    try:
        config = load_config(args.config)
    except Exception as exc:
        fail(f"invalid configuration: {exc}")
    dataset = Path(config.dataset)
    required = ("events.json", "expected.json", "documents")
    missing = [name for name in required if not (dataset / name).exists()]
    if missing:
        fail(f"dataset {dataset} is missing {missing}")
    output = Path(config.output_dir)
    try:
        output.mkdir(parents=True, exist_ok=True)
        probe = output / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        fail(f"output directory is not writable: {exc}")
    for strategy in config.strategies:
        server = config.server_for(strategy)
        if strategy == "prefix_cached_full_context" and not server.prefix_caching:
            fail(f"{strategy}: endpoint must declare prefix_caching: true")
        try:
            response = httpx.get(server.base_url.rstrip("/") + "/models", timeout=5,
                                 headers={"Authorization": "Bearer " + __import__("os").environ.get(
                                     server.api_key_env, "")})
            response.raise_for_status()
            models = response.json().get("data", [])
            if server.model not in [item.get("id") for item in models]:
                fail(f"{strategy}: configured model {server.model!r} is not served")
        except Exception as exc:
            fail(f"{strategy}: vLLM endpoint health/model check failed: {exc}")
    print(json.dumps({"gpu": gpu.splitlines(), "model": config.server.model,
                      "dataset": str(dataset), "output_dir": str(output), "status": "ready"},
                     indent=2))


if __name__ == "__main__":
    main()