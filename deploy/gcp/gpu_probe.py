"""Check the assigned GPU and CUDA runtime before any model download or load."""

import importlib
import importlib.metadata
import json
import os
import subprocess
import sys

REEXEC_MARKER = "CONTEXTBENCH_GPU_PROBE_REEXEC"


def check_gpu():
    versions = {name: importlib.metadata.version(name) for name in ("torch", "vllm")}
    print(json.dumps({"packages": versions, "cuda_compatibility": {
        name: os.environ.get(name) for name in
        ("VLLM_ENABLE_CUDA_COMPATIBILITY", "VLLM_CUDA_COMPATIBILITY_PATH")
    }}), flush=True)
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30, check=False)
    print("nvidia-smi GPU, driver, memory:\n" + result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError(f"nvidia-smi exited with status {result.returncode}: {result.stderr}")

    # EngineCore starts a fresh interpreter with vLLM's prepared library path.
    # Re-exec once when that path changes so this probe uses the same dynamic
    # linker environment. Never import torch in the parent task.
    previous_library_path = os.environ.get("LD_LIBRARY_PATH")
    importlib.import_module("vllm.env_override")
    if (os.environ.get("LD_LIBRARY_PATH") != previous_library_path
            and os.environ.get(REEXEC_MARKER) != "1"):
        environment = os.environ.copy()
        environment[REEXEC_MARKER] = "1"
        print("Restarting GPU preflight with prepared CUDA compatibility libraries.", flush=True)
        os.execve(sys.executable, [sys.executable, "-m", "deploy.gcp.gpu_probe"], environment)
    torch = importlib.import_module("torch")
    print(json.dumps({"torch_cuda_build": torch.version.cuda}), flush=True)
    torch.cuda.init()
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("No CUDA GPU is available to the task")
    properties = torch.cuda.get_device_properties(0)
    value = torch.ones(1, device="cuda:0") + 1
    torch.cuda.synchronize()
    if value.item() != 2:
        raise RuntimeError("CUDA tensor operation returned an unexpected result")
    print(json.dumps({"status": "passed", "device_count": count,
                      "device": properties.name, "gpu_memory_bytes": properties.total_memory}),
          flush=True)


def main() -> int:
    try:
        check_gpu()
    except Exception as exc:  # noqa: BLE001 - expose diagnostics and fail the isolated probe.
        print(f"GPU preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
