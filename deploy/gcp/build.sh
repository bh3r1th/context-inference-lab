#!/usr/bin/env bash
# Run from an authenticated Cloud Shell checkout, not from the Cloud Run container.
set -euo pipefail

project=project-ce0aa110-c8ea-4c1e-80b
region=us-east4
registry="$region-docker.pkg.dev/$project/context-inference-lab"
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
tag="$(git -C "$repo_root" rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
image="$registry/benchmark:$tag"
context_dir=$(mktemp -d)
trap 'rm -rf -- "$context_dir"' EXIT

# Stage an explicit build context. Never upload the checkout's environment files,
# Git credentials, virtualenv, previous results, or unrelated files to Cloud Build.
tar -C "$repo_root" --exclude='__pycache__' --exclude='.pytest_cache' \
    --exclude='.ruff_cache' --exclude='*.pyc' --exclude='.env' --exclude='.env.*' \
    --exclude='*credentials*.json' --exclude='test_*.py' \
    -cf - pyproject.toml README.md src scripts configs data deploy/gcp \
    | tar -C "$context_dir" -xf -

gcloud builds submit "$context_dir" --project="$project" --region="$region" \
    --config="$context_dir/deploy/gcp/cloudbuild.yaml" \
    --substitutions="_IMAGE=$image"

digest=$(gcloud artifacts docker images describe "$image" --project="$project" \
    --format='value(image_summary.digest)')
if [[ ! "$digest" =~ ^sha256:[a-f0-9]{64}$ ]]; then
    printf 'Build returned no valid image digest; deployment is not confirmed.\n' >&2
    exit 1
fi
printf '\nBuilt image for deployment:\n%s/benchmark@%s\n' "$registry" "$digest"
