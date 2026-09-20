"""Resume an interrupted uncached full benchmark at completed-cell boundaries."""

import argparse
import asyncio
import json
import subprocess
import tempfile
from pathlib import Path

import yaml

from contextbench.experiments.config import ExperimentConfig, load_config
from contextbench.experiments.orchestrator import run
from run_context_scaling import execute_sizes


SIZES = ("2k", "8k", "16k", "32k", "64k")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def completed_enterprise_cells(root: Path):
    completed = {}

    if not root.exists():
        return completed

    for summary_path in root.glob("*/*/summary.json"):
        summary = read_json(summary_path)
        if not summary or summary.get("status") != "completed":
            continue

        strategy = summary.get("strategy")
        concurrency = summary.get("concurrency")

        if strategy is None or concurrency is None:
            continue

        completed[(strategy, int(concurrency))] = str(summary_path.parent)

    return completed


def completed_scaling_cells(root: Path):
    completed = {}

    if not root.exists():
        return completed

    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue

        metadata = read_json(run_dir / "metadata.json")
        if not metadata:
            continue

        experiment_id = metadata.get("experiment_id", "")

        if not experiment_id.startswith("context-scaling-"):
            continue

        label = experiment_id.removeprefix("context-scaling-")

        for summary_path in run_dir.glob("*/summary.json"):
            summary = read_json(summary_path)

            if not summary or summary.get("status") != "completed":
                continue

            concurrency = summary.get("concurrency")

            if concurrency is None:
                continue

            completed[(label, int(concurrency))] = str(summary_path.parent)

    return completed


def write_state(path: Path, enterprise_done, scaling_done):
    state = {
        "enterprise_completed": [
            {
                "strategy": strategy,
                "concurrency": concurrency,
                "path": artifact,
            }
            for (strategy, concurrency), artifact in sorted(enterprise_done.items())
        ],
        "scaling_completed": [
            {
                "size": size,
                "concurrency": concurrency,
                "path": artifact,
            }
            for (size, concurrency), artifact in sorted(scaling_done.items())
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def full_benchmark_is_running():
    result = subprocess.run(
        ["pgrep", "-f", "scripts/run_full_benchmark.py"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


async def run_enterprise_cell(config_path, output_root, strategy, concurrency):
    base = load_config(config_path)

    values = base.model_dump()
    values["strategies"] = [strategy]
    values["concurrency"] = [concurrency]
    values["phase"] = "uncached"
    values["output_dir"] = str(output_root.resolve())

    config = ExperimentConfig.model_validate(values)

    print(f"RUN enterprise {strategy}-c{concurrency}", flush=True)
    return await run(config)


async def run_scaling_cell(
    config_path,
    source,
    fixture_root,
    run_root,
    size,
    concurrency,
):
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base["concurrency"] = [concurrency]

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(base, handle, sort_keys=False)
        temp_config = Path(handle.name)

    try:
        print(f"RUN scaling {size}-c{concurrency}", flush=True)

        paths = await execute_sizes(
            temp_config,
            source,
            fixture_root,
            (size,),
            phase="uncached",
            run_output=run_root,
        )

        return paths
    finally:
        temp_config.unlink(missing_ok=True)


async def execute(args):
    enterprise_cfg = load_config(args.enterprise)

    enterprise_expected = [
        (strategy, concurrency)
        for strategy in enterprise_cfg.strategies
        for concurrency in enterprise_cfg.concurrency
    ]

    scaling_expected = [
        (size, concurrency)
        for size in SIZES
        for concurrency in enterprise_cfg.concurrency
    ]

    enterprise_done = completed_enterprise_cells(args.enterprise_root)
    scaling_done = completed_scaling_cells(args.scaling_run_root)

    enterprise_missing = [
        cell for cell in enterprise_expected if cell not in enterprise_done
    ]

    scaling_missing = [
        cell for cell in scaling_expected if cell not in scaling_done
    ]

    print("\nENTERPRISE COMPLETED")
    for strategy, concurrency in enterprise_expected:
        status = "✓" if (strategy, concurrency) in enterprise_done else "→"
        print(f"{status} {strategy}-c{concurrency}")

    print("\nSCALING COMPLETED")
    for size, concurrency in scaling_expected:
        status = "✓" if (size, concurrency) in scaling_done else "→"
        print(f"{status} {size}-c{concurrency}")

    write_state(args.state, enterprise_done, scaling_done)

    if not args.execute:
        print("\nPLAN ONLY — nothing executed.")
        return

    if full_benchmark_is_running():
        raise SystemExit(
            "Refusing resume: run_full_benchmark.py is currently active."
        )

    for strategy, concurrency in enterprise_missing:
        await run_enterprise_cell(
            args.enterprise,
            args.enterprise_root,
            strategy,
            concurrency,
        )

        enterprise_done = completed_enterprise_cells(args.enterprise_root)
        scaling_done = completed_scaling_cells(args.scaling_run_root)
        write_state(args.state, enterprise_done, scaling_done)

    for size, concurrency in scaling_missing:
        await run_scaling_cell(
            args.scaling,
            args.source,
            args.scaling_fixture_root,
            args.scaling_run_root,
            size,
            concurrency,
        )

        enterprise_done = completed_enterprise_cells(args.enterprise_root)
        scaling_done = completed_scaling_cells(args.scaling_run_root)
        write_state(args.state, enterprise_done, scaling_done)

    print("\n✅ Resume workload complete.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--enterprise",
        type=Path,
        default=Path("configs/experiment-uncached.yaml"),
    )
    parser.add_argument(
        "--scaling",
        type=Path,
        default=Path("configs/context-scaling.yaml"),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/enterprise"),
    )

    parser.add_argument(
        "--enterprise-root",
        type=Path,
        default=Path("results/enterprise/uncached"),
    )
    parser.add_argument(
        "--scaling-fixture-root",
        type=Path,
        default=Path("results/context-scaling"),
    )
    parser.add_argument(
        "--scaling-run-root",
        type=Path,
        default=Path("results/context-scaling/uncached/runs"),
    )

    parser.add_argument(
        "--state",
        type=Path,
        default=Path("results/resume_uncached_state.json"),
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute missing cells. Default is plan-only.",
    )

    args = parser.parse_args()
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
