"""Validate and run the small enterprise, prefix-cache, and scaling pilot matrix."""
import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from contextbench.experiments.config import load_config
from contextbench.experiments.orchestrator import run
from run_context_scaling import execute_sizes


async def execute(args):
    configs = [args.enterprise, args.prefix_cold, args.prefix_warm]
    for config_path in configs:
        await run(load_config(config_path))
    await execute_sizes(args.scaling, args.source, args.scaling_output, ("2k", "8k", "32k"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enterprise", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--prefix-cold", type=Path, default=Path("configs/pilot-prefix-cold.yaml"))
    parser.add_argument("--prefix-warm", type=Path, default=Path("configs/pilot-prefix-warm.yaml"))
    parser.add_argument("--scaling", type=Path, default=Path("configs/pilot-scaling.yaml"))
    parser.add_argument("--source", type=Path, default=Path("data/enterprise"))
    parser.add_argument("--scaling-output", type=Path, default=Path("results/pilot/context-scaling"))
    args = parser.parse_args()
    subprocess.run([sys.executable, "scripts/validate_environment.py", str(args.enterprise)], check=True)
    asyncio.run(execute(args))
    roots = [str(Path(load_config(path).output_dir).resolve()) for path in
             (args.enterprise, args.prefix_cold, args.prefix_warm)]
    roots.append(str((args.scaling_output / "runs").resolve()))
    subprocess.run([sys.executable, "scripts/validate_pilot.py", *roots], check=True)


if __name__ == "__main__":
    main()