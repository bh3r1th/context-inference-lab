from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Decision(StrictModel):
    event_id: str
    decision: Literal["approve", "reject", "abstain"]
    limit_usd: int | None = Field(ge=0)
    applied_rules: list[str]
    reason: str = Field(min_length=1)


@dataclass(frozen=True)
class Context:
    """Static sources and complete event prompt; count excludes chat-template framing."""
    text: str
    source_ids: tuple[str, ...]
    prompt: str = ""
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: tuple[dict[str, str], ...] = ()

    @property
    def selected_source_ids(self) -> tuple[str, ...]:
        return self.source_ids


class ContextStrategy(Protocol):
    def build(self, event: dict[str, Any]) -> Context: ...


@dataclass
class InferenceResult:
    text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_s: float | None = None
    latency_s: float | None = None
    decode_tokens_per_s: float | None = None
    finish_reason: str | None = None
    response_id: str | None = None
    error: str | None = None
    prefill_s: float | None = None
    server_queue_s: float | None = None
    kv_cache_utilization: float | None = None
    decode_time_s: float | None = None
    stream_generation_s: float | None = None
    server_ttft_s: float | None = None
    server_mean_itl_s: float | None = None
    server_output_tokens_per_s: float | None = None
    output_tokens_per_s: float | None = None
    cached_input_tokens: int | None = None
    cached_input_fraction: float | None = None
    gpu_memory_used_mib: float | None = None
    gpu_utilization_percent: float | None = None
    started_at: str | None = None
    completed_at: str | None = None
    model: str | None = None
    response_model: str | None = None
    http_status: int | None = None
    reasoning_text: str = ""
    raw_usage: dict[str, Any] | None = None
    raw_server_metrics: dict[str, Any] | None = None
    stream_events: list[dict[str, Any]] = field(default_factory=list)
    metric_sources: dict[str, str] = field(default_factory=dict)
    unsupported: dict[str, str] = field(default_factory=dict)


class InferenceBackend(Protocol):
    async def generate(self, context: Context, event: dict[str, Any], seed: int) -> InferenceResult: ...
