# Separate from the legacy GPU image; keep the serving API aligned with the configs.
# Official v0.29.0 linux/amd64 manifest, inspected during deployment preparation.
ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.29.0@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
FROM ${VLLM_BASE_IMAGE}

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/huggingface \
    VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
    RESULTS_BUCKET=context-inference-lab RESULTS_PREFIX=cloud-run \
    TASK_PLAN=/app/deploy/gcp/smoke-plan.json

# Explicit copies exclude credentials, virtual environments, old results and Git metadata.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/ ./data/
COPY deploy/gcp/ ./deploy/gcp/
RUN python3 -m pip install 'google-cloud-storage==3.4.1' -e . \
    && python3 -m pip check \
    && python3 deploy/gcp/make_plan.py --stage pilot --output deploy/gcp/pilot-plan.json

ENTRYPOINT ["python3", "-m", "deploy.gcp.task"]
