"""Run the complete enterprise and context-scaling matrices after a passing pilot."""
import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from contextbench.experiments.config import load_config
from contextbench.experiments.orchestrator import run
from run_context_scaling import execute_sizes


async def execute(args):
    pilot = Path(args.pilot_validation)
    if not pilot.exists() or json.loads(pilot.read_text(encoding="utf-8")).get("status") != "PASS":
        raise SystemExit("Full benchmark blocked: pilot_validation.json must have status PASS")
    runs = [await run(load_config(args.enterprise))]
    runs.extend(await execute_sizes(args.scaling, args.source, args.scaling_output,
                                    ("2k", "8k", "16k", "32k", "64k")))
    manifest = {"status": "completed", "experiments": []}
    for path in runs:
        metadata_path = path / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest["experiments"].append({
            "experiment_id": metadata.get("experiment_id"),
            "run_id": metadata.get("run_id"),
            "path": str(path),
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "status": metadata.get("status"),
        })
    if any(item["status"] != "completed" for item in manifest["experiments"]):
        manifest["status"] = "completed_with_errors"
    output = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enterprise", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--scaling", type=Path, default=Path("configs/context-scaling.yaml"))
    parser.add_argument("--source", type=Path, default=Path("data/enterprise"))
    parser.add_argument("--scaling-output", type=Path, default=Path("results/context-scaling"))
    parser.add_argument("--pilot-validation", type=Path,
                        default=Path("results/pilot/pilot_validation.json"))
    parser.add_argument("--manifest", type=Path, default=Path("results/benchmark_manifest.json"))
    args = parser.parse_args()
    pilot = Path(args.pilot_validation)
    if not pilot.exists() or json.loads(pilot.read_text(encoding="utf-8")).get("status") != "PASS":
        raise SystemExit("Full benchmark blocked: pilot_validation.json must have status PASS")
    subprocess.run([sys.executable, "scripts/validate_environment.py", str(args.enterprise)], check=True)
    print(asyncio.run(execute(args)))


if __name__ == "__main__":
    main()