"""Validate and execute a benchmark configuration."""
import argparse
import asyncio
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, default=Path("configs/experiment.yaml"), nargs="?")
    args = parser.parse_args()
    subprocess.run([sys.executable, "scripts/validate_environment.py", str(args.config)], check=True)
    from contextbench.experiments.config import load_config
    from contextbench.experiments.orchestrator import run
    print(asyncio.run(run(load_config(args.config))))


if __name__ == "__main__":
    main()