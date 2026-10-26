#!/usr/bin/env bash
# Updates the existing job. It does not execute a GPU task.
set -euo pipefail

image=${1:?Usage: bash deploy/gcp/deploy.sh IMAGE_DIGEST [smoke|pilot] [PILOT_CELL_ID]}
stage=${2:-smoke}
cell_id=${3:-}
request_timeout=${CONTEXTBENCH_REQUEST_TIMEOUT_SECONDS:-120}
if [[ ! "$request_timeout" =~ ^[1-9][0-9]{0,3}$ ]] || (( request_timeout > 3300 )); then
    printf 'CONTEXTBENCH_REQUEST_TIMEOUT_SECONDS must be an integer from 1 to 3300.\n' >&2
    exit 1
fi
cuda_compatibility=${VLLM_ENABLE_CUDA_COMPATIBILITY:-0}
case "$cuda_compatibility" in
    0|1) ;;
    *) printf 'VLLM_ENABLE_CUDA_COMPATIBILITY must be 0 or 1.\n' >&2; exit 1 ;;
esac
project=project-ce0aa110-c8ea-4c1e-80b
region=us-east4
job=context-inference-lab
expected_prefix="$region-docker.pkg.dev/$project/context-inference-lab/benchmark@sha256:"
if [[ "$image" != "$expected_prefix"* || ! "${image##*@}" =~ ^sha256:[a-f0-9]{64}$ ]]; then
    printf 'Use the immutable image reference printed by build.sh.\n' >&2
    exit 1
fi
case "$stage" in
    smoke) tasks=1 ;;
    pilot) tasks=18 ;;
    *) printf 'Stage must be smoke or pilot. Full runs require GPU pilot measurements.\n' >&2; exit 1 ;;
esac

if [[ -n "$cell_id" ]]; then
    if [[ "$stage" != pilot || ! "$cell_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        printf 'A cell selector must be a safe pilot cell ID.\n' >&2
        exit 1
    fi
    tasks=1
fi

gcloud run jobs update "$job" --project="$project" --region="$region" \
    --image="$image" --command='' --args='' \
    --gpu=1 --gpu-type=nvidia-l4 --cpu=8 --memory=32Gi \
    --tasks="$tasks" --parallelism=1 --task-timeout=3600s --max-retries=0 \
    --service-account=809334676818-compute@developer.gserviceaccount.com \
    --update-env-vars="RESULTS_BUCKET=context-inference-lab,RESULTS_PREFIX=cloud-run,TASK_PLAN=/app/deploy/gcp/$stage-plan.json,TASK_CELL_ID=$cell_id,TASK_DEADLINE_SECONDS=3300,VLLM_ENABLE_CUDA_COMPATIBILITY=$cuda_compatibility,CONTEXTBENCH_GPU_MEMORY_UTILIZATION=0.92,CONTEXTBENCH_REQUEST_TIMEOUT_SECONDS=$request_timeout"

printf '\nJob updated, not executed. Launch with:\n'
printf 'gcloud run jobs execute %s --project=%s --region=%s\n' "$job" "$project" "$region"
