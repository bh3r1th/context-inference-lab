"""Validate and run the small enterprise, prefix-cache, and scaling pilot matrix."""
import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from contextbench.experiments.config import load_config
from contextbench.experiments.orchestrator import run
from contextbench.experiments.phases import phase_config
from run_context_scaling import execute_sizes


async def execute(args):
    if args.phase == "all":
        for config_path in (args.enterprise, args.prefix_cold, args.prefix_warm):
            await run(load_config(config_path))
        await execute_sizes(args.scaling, args.source, args.scaling_output,
                            ("2k", "8k", "32k"))
        return
    phases = ("uncached", "cached") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "uncached":
            await run(phase_config(args.enterprise, phase,
                                    Path(args.enterprise_output) / phase))
            await execute_sizes(args.scaling, args.source,
                                args.scaling_output / phase, ("2k", "8k", "32k"), phase=phase)
        else:
            await run(phase_config(args.enterprise, phase,
                                    Path(args.enterprise_output) / phase))
            for config_path in (args.prefix_cold, args.prefix_warm):
                await run(phase_config(config_path, phase,
                                       args.prefix_output / phase / config_path.stem))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enterprise", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--prefix-cold", type=Path, default=Path("configs/pilot-prefix-cold.yaml"))
    parser.add_argument("--prefix-warm", type=Path, default=Path("configs/pilot-prefix-warm.yaml"))
    parser.add_argument("--scaling", type=Path, default=Path("configs/pilot-scaling.yaml"))
    parser.add_argument("--source", type=Path, default=Path("data/enterprise"))
    parser.add_argument("--scaling-output", type=Path, default=Path("results/pilot/context-scaling"))
    parser.add_argument("--enterprise-output", type=Path, default=Path("results/pilot/enterprise"))
    parser.add_argument("--prefix-output", type=Path, default=Path("results/pilot/prefix"))
    parser.add_argument("--phase", choices=("uncached", "cached", "all"), default="all")
    args = parser.parse_args()
    check_config = args.enterprise if args.phase != "cached" else args.prefix_cold
    subprocess.run([sys.executable, "scripts/validate_environment.py", str(check_config)], check=True)
    asyncio.run(execute(args))
    if args.phase == "all":
        roots = [str(Path(load_config(path).output_dir).resolve()) for path in
                 (args.enterprise, args.prefix_cold, args.prefix_warm)]
        roots.append(str((args.scaling_output / "runs").resolve()))
        subprocess.run([sys.executable, "scripts/validate_pilot.py", *roots,
                        "--required-phases", "all"], check=True)


if __name__ == "__main__":
    main()