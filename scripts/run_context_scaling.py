"""Run the five context-size workloads using one shared server configuration."""
import argparse
import asyncio
import copy
from pathlib import Path

import yaml

from contextbench.experiments.config import ExperimentConfig
from contextbench.experiments.orchestrator import run
from generate_scaling_workloads import TARGETS, generate


async def execute(config_path: Path, source: Path, output: Path):
    return await execute_sizes(config_path, source, output, TARGETS)


async def execute_sizes(config_path: Path, source: Path, output: Path, sizes):
    run_paths = []
    generate(source, output / "template")
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    unknown = set(sizes) - set(TARGETS)
    if unknown:
        raise ValueError(f"unknown scaling sizes: {sorted(unknown)}")
    for label in sizes:
        values = copy.deepcopy(base)
        values["experiment_id"] = f"context-scaling-{label}"
        values["dataset"] = str((output / label).resolve())
        values["output_dir"] = str((output / "runs").resolve())
        run_paths.append(await run(ExperimentConfig.model_validate(values)))
    return run_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/context-scaling.yaml"))
    parser.add_argument("--source", type=Path, default=Path("data/enterprise"))
    parser.add_argument("--output", type=Path, default=Path("results/context-scaling"))
    parser.add_argument("--sizes", default=','.join(TARGETS),
                        help="comma-separated sizes, for example 2k,8k,32k")
    args = parser.parse_args()
    asyncio.run(execute_sizes(args.config, args.source, args.output,
                              [size.strip() for size in args.sizes.split(',') if size.strip()]))


if __name__ == "__main__":
    main()