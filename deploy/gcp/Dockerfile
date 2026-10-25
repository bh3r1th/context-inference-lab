# Separate from the legacy GPU image; keep the serving API aligned with the configs.
# Official CUDA 12.9 linux/amd64 variant; avoid the default CUDA 13 image.
ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.29.0-cu129@sha256:3e10e8189823e0f7ae4620c271bcdaaf64127ec7d0edc351591a508498b7684a
FROM ${VLLM_BASE_IMAGE}

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/huggingface \
    VLLM_ENABLE_CUDA_COMPATIBILITY=0 \
    RESULTS_BUCKET=context-inference-lab RESULTS_PREFIX=cloud-run \
    TASK_PLAN=/app/deploy/gcp/smoke-plan.json

# Explicit copies exclude credentials, virtual environments, old results and Git metadata.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/ ./data/
COPY deploy/gcp/ ./deploy/gcp/
RUN python3 -m pip install 'google-cloud-storage>=3.4.1,<4' -e . \
    && python3 deploy/gcp/check_dependencies.py \
    && python3 deploy/gcp/make_plan.py --stage pilot --output deploy/gcp/pilot-plan.json

ENTRYPOINT ["python3", "-m", "deploy.gcp.task"]
