"""One-command no-GPU smoke test."""
import subprocess
import sys


if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "tests/test_vllm_smoke.py", "-q"]))