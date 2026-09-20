FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements-gpu.txt .
RUN python3.11 -m pip install --upgrade pip \
    && python3.11 -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-gpu.txt
COPY . .
RUN python3.11 -m pip install --no-deps -e .
ENTRYPOINT ["python3.11", "scripts/run_benchmark.py"]