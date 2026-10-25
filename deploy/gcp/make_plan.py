"""Describe independent Cloud Run smoke/pilot cells without changing benchmark inputs.

Each plan entry is one complete strategy/concurrency cell. The task launcher uses
Cloud Run's task index to select exactly one entry and starts a fresh vLLM process,
including for every cold-cache cell. Full-run planning deliberately waits for GPU
pilot measurements; this module does not reduce repetitions or split request plans.
"""

import argparse
import json
from pathlib import Path

from contextbench.experiments.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def cells(root: Path, config_path: str, label: str, *, strategies=None, scaling_size=None):
    config = load_config(root / config_path)
    selected = list(config.strategies) if strategies is None else list(strategies)
    if not set(selected) <= set(config.strategies):
        raise ValueError(f"plan selects a strategy absent from {config_path}")
    result = []
    for strategy in selected:
        for concurrency in config.concurrency:
            entry = {
                "id": f"{label}-{strategy}-c{concurrency}",
                "config": config_path,
                "strategy": strategy,
                "concurrency": concurrency,
            }
            if scaling_size is not None:
                entry["scaling_size"] = scaling_size
            result.append(entry)
    return result


def make_plan(stage: str, root: Path = ROOT) -> list[dict]:
    if stage == "smoke":
        return cells(root, "configs/smoke.yaml", "smoke")
    if stage != "pilot":
        raise ValueError("stage must be smoke or pilot; full runs require GPU pilot results")

    plan = cells(root, "configs/pilot.yaml", "pilot-enterprise")
    for size in ("2k", "8k", "32k"):
        plan.extend(cells(root, "configs/pilot-scaling.yaml", f"pilot-scaling-{size}",
                          scaling_size=size))
    for scenario in ("cold", "warm"):
        plan.extend(cells(root, f"configs/pilot-prefix-{scenario}.yaml",
                          f"pilot-prefix-{scenario}",
                          strategies=["prefix_cached_full_context"]))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--output", type=Path, help="Write JSON to this path instead of stdout")
    args = parser.parse_args()
    text = json.dumps(make_plan(args.stage), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
