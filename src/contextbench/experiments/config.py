from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from contextbench.context.config import ContextConfig, StrategyName
from contextbench.models import StrictModel


class Server(StrictModel):
    base_url: str = "http://localhost:8000/v1"
    metrics_url: str | None = "http://localhost:8000/metrics"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    model_revision: str
    vllm_version: str
    launch_command: str
    prefix_caching: bool = False
    api_key_env: str = "VLLM_API_KEY"
    timeout_s: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    max_tokens: int = Field(default=256, gt=0)
    structured_output: bool = True
    structured_output_mode: Literal["vllm", "json_schema"] = "vllm"
    version_url: str | None = None


class ExperimentConfig(StrictModel):
    experiment_id: str | None = Field(default=None, min_length=1)
    seed: int = Field(default=42, ge=0)
    dataset: str = "../data/synthetic"
    output_dir: str = "../results"
    strategies: list[StrategyName] = Field(default_factory=lambda: [
        "full_context", "prefix_cached_full_context", "retrieval", "compiled_context"], min_length=1)
    concurrency: list[int] = Field(default_factory=lambda: [1, 8, 32], min_length=1)
    repetitions: int = Field(default=10, gt=0)
    warmup_requests: int = Field(default=2, ge=0)
    retrieval_top_k: int = Field(default=8, gt=0)
    context: ContextConfig = Field(default_factory=ContextConfig)
    telemetry_interval_s: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    local_gpu_metrics: bool = False
    cache_scenario: Literal["cold", "warm"] | None = None
    phase: Literal["uncached", "cached", "all"] = "all"
    event_ids: list[str] | None = Field(default=None, min_length=1)
    server: Server
    strategy_servers: dict[StrategyName, Server] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_matrix(self):
        if self.event_ids is not None and len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("duplicate event_ids")
        if "retrieval_top_k" in self.model_fields_set and "context" in self.model_fields_set:
            if self.retrieval_top_k != self.context.retrieval.top_k:
                raise ValueError("retrieval_top_k conflicts with context.retrieval.top_k")
        elif "retrieval_top_k" in self.model_fields_set:
            self.context.retrieval.top_k = self.retrieval_top_k
        else:
            self.retrieval_top_k = self.context.retrieval.top_k
        if any(c <= 0 for c in self.concurrency):
            raise ValueError("concurrency must be positive")
        if len(set(self.concurrency)) != len(self.concurrency):
            raise ValueError("duplicate concurrency")
        if len(set(self.strategies)) != len(self.strategies):
            raise ValueError("duplicate strategies")
        reference = self.server
        for strategy, server in self.strategy_servers.items():
            for field in ("model", "model_revision", "vllm_version", "max_tokens",
                          "structured_output", "structured_output_mode"):
                if getattr(server, field) != getattr(reference, field):
                    raise ValueError(f"strategy server {strategy} changes shared parameter {field}")
        return self

    def server_for(self, strategy: StrategyName) -> Server:
        return self.strategy_servers.get(strategy, self.server)


def load_config(path: Path) -> ExperimentConfig:
    config = ExperimentConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    for field in ("dataset", "output_dir"):
        setattr(config, field, str((path.parent / getattr(config, field)).resolve()))
    return config
