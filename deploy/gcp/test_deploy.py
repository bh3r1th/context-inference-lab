"""Verify job updates select one complete retry cell without executing GCP work."""

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("deploy.sh")
IMAGE = ("us-east4-docker.pkg.dev/project-ce0aa110-c8ea-4c1e-80b/"
         "context-inference-lab/benchmark@sha256:" + "a" * 64)
CELL = "pilot-scaling-32k-full_context-c8"


@pytest.fixture
def deploy(tmp_path):
    binary = tmp_path / "gcloud"
    output = tmp_path / "arguments.json"
    binary.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                      "with open(os.environ['GCLOUD_ARGUMENTS_FILE'], 'w') as f:\n"
                      "    json.dump(sys.argv[1:], f)\n")
    binary.chmod(0o755)

    def invoke(stage, cell=None, timeout="120"):
        output.unlink(missing_ok=True)
        env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
               "GCLOUD_ARGUMENTS_FILE": str(output),
               "CONTEXTBENCH_REQUEST_TIMEOUT_SECONDS": timeout,
               "VLLM_ENABLE_CUDA_COMPATIBILITY": "0"}
        result = subprocess.run(["bash", str(SCRIPT), IMAGE, stage,
                                 *([cell] if cell is not None else [])],
                                env=env, capture_output=True, text=True, check=False)
        return result, json.loads(output.read_text()) if output.exists() else None

    return invoke


def test_retry_deploys_one_task_and_passes_timeout_and_selector(deploy):
    result, args = deploy("pilot", CELL, "600")
    assert result.returncode == 0
    assert args[:4] == ["run", "jobs", "update", "context-inference-lab"]
    assert "--tasks=1" in args and "--parallelism=1" in args
    assert "--task-timeout=3600s" in args and "--max-retries=0" in args
    env = dict(part.split("=", 1) for part in next(
        arg.removeprefix("--update-env-vars=") for arg in args
        if arg.startswith("--update-env-vars=")).split(","))
    assert env["TASK_PLAN"] == "/app/deploy/gcp/pilot-plan.json"
    assert env["TASK_CELL_ID"] == CELL
    assert env["CONTEXTBENCH_REQUEST_TIMEOUT_SECONDS"] == "600"


@pytest.mark.parametrize("stage,tasks", [("smoke", 1), ("pilot", 18)])
def test_normal_deploy_clears_retry_selection_and_restores_timeout(deploy, stage, tasks):
    result, args = deploy(stage)
    assert result.returncode == 0
    assert f"--tasks={tasks}" in args
    env = next(arg for arg in args if arg.startswith("--update-env-vars="))
    assert ",TASK_CELL_ID=," in env
    assert "CONTEXTBENCH_REQUEST_TIMEOUT_SECONDS=120" in env


@pytest.mark.parametrize("stage,cell,timeout", [
    ("smoke", CELL, "600"), ("pilot", "bad,selector", "600"),
    ("pilot", CELL, "nan"), ("pilot", CELL, "0"), ("pilot", CELL, "3301"),
])
def test_invalid_deploy_input_fails_before_job_update(deploy, stage, cell, timeout):
    result, args = deploy(stage, cell, timeout)
    assert result.returncode != 0 and args is None
