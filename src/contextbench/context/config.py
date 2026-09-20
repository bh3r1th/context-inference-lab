"""Context configuration has no model-serving dependency."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from contextbench.models import StrictModel

StrategyName = Literal["full_context", "retrieval", "compiled_context", "prefix_cached_full_context"]


class RetrievalConfig(StrictModel):
    method: Literal["bm25", "embedding"] = "bm25"
    top_k: int = Field(default=8, gt=0)
    chunk_chars: int = Field(default=1200, gt=0)
    overlap_chars: int = Field(default=120, ge=0)
    bm25_k1: float = Field(default=1.5, gt=0, allow_inf_nan=False)
    bm25_b: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", min_length=1)
    embedding_revision: str | None = None
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, gt=0)
    local_files_only: bool = True

    @model_validator(mode="after")
    def check_overlap(self):
        if self.overlap_chars >= self.chunk_chars:
            raise ValueError("overlap_chars must be smaller than chunk_chars")
        return self


class TokenizerConfig(StrictModel):
    method: Literal["regex", "huggingface"] = "regex"
    model: str | None = None
    revision: str | None = None
    local_files_only: bool = True

    @model_validator(mode="after")
    def check_model(self):
        if self.method == "huggingface" and not self.model:
            raise ValueError("huggingface token counting requires a tokenizer model or local path")
        return self


class ContextConfig(StrictModel):
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)


def load_context_config(path: Path) -> ContextConfig:
    return ContextConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
