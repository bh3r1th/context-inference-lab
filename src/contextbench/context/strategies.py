"""Typed source-only strategies; no inference backend or answer-key dependencies."""
import json
import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from contextbench.context.adapters import (
    EmbeddingEncoder,
    SentenceTransformerEncoder,
    TokenCounter,
    make_token_counter,
)
from contextbench.context.config import ContextConfig, RetrievalConfig
from contextbench.context.prompt import render_messages, render_prompt
from contextbench.context.sources import chunk_documents, covered_rules, load_sources
from contextbench.models import Context, ContextStrategy


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class BaseContextStrategy:
    def __init__(self, root: Path, *, config: ContextConfig | None = None,
                 token_counter: TokenCounter | None = None):
        self.config = config or ContextConfig()
        self.documents, self.rules = load_sources(root)
        self.token_counter = token_counter or make_token_counter(self.config.tokenizer)

    def finish(self, text: str, event: dict[str, Any], ids: tuple[str, ...],
               metadata: dict[str, Any]) -> Context:
        messages = render_messages(text, event)
        prompt = render_prompt(messages)
        count = self.token_counter.count(prompt)
        if type(count) is not int or count < 0:
            raise ValueError("token counter must return a nonnegative integer")
        static_prefix = messages[0]["content"]
        dynamic_suffix = messages[-1]["content"]
        return Context(text, ids, prompt, count, {
            **metadata,
            "tokenizer": dict(self.token_counter.metadata),
            "prefix_layout": "system_static_prefix_then_user_dynamic_event",
            "static_prefix_sha256": hashlib.sha256(static_prefix.encode()).hexdigest(),
            "static_prefix_token_count": self.token_counter.count(static_prefix),
            "dynamic_suffix_sha256": hashlib.sha256(dynamic_suffix.encode()).hexdigest(),
        }, messages)


class FullContextStrategy(BaseContextStrategy):
    def __init__(self, root: Path, *, config: ContextConfig | None = None,
                 token_counter: TokenCounter | None = None, prefix_cached: bool = False):
        super().__init__(root, config=config, token_counter=token_counter)
        self.prefix_cached = prefix_cached

    def build(self, event: dict[str, Any]) -> Context:
        text = "\n\n".join(f"SOURCE {d.id}\n{d.text}" for d in self.documents)
        return self.finish(text, event, tuple(d.id for d in self.documents), {
            "strategy": "full_context",
            "document_count": len(self.documents),
            "selected_rule_ids": [r.id for r in self.rules],
            "source_id_kind": "document"})


def normalized(vectors: Sequence[Sequence[float]], count: int,
               dimension: int | None = None) -> list[tuple[float, ...]]:
    if len(vectors) != count:
        raise ValueError("embedding encoder returned an incorrect number of vectors")
    result = []
    for vector in vectors:
        values = tuple(float(v) for v in vector)
        if dimension is None:
            dimension = len(values)
        if not values or len(values) != dimension or not all(math.isfinite(v) for v in values):
            raise ValueError("embeddings must have consistent nonempty dimensions and finite values")
        norm = math.hypot(*values)
        if not norm or not math.isfinite(norm):
            raise ValueError("embedding vectors must have a finite nonzero norm")
        result.append(tuple(v / norm for v in values))
    return result


class RetrievalContextStrategy(BaseContextStrategy):
    def __init__(self, root: Path, top_k: int | None = None, *,
                 config: ContextConfig | None = None, token_counter: TokenCounter | None = None,
                 encoder: EmbeddingEncoder | None = None):
        super().__init__(root, config=config, token_counter=token_counter)
        options = self.config.retrieval.model_dump()
        if top_k is not None:
            options["top_k"] = top_k
        self.options = RetrievalConfig.model_validate(options)
        self.chunks = chunk_documents(self.documents, self.options.chunk_chars,
                                      self.options.overlap_chars)
        if not self.chunks:
            raise ValueError("no nonempty source chunks found")
        self.encoder = None
        if self.options.method == "embedding":
            self.encoder = encoder or SentenceTransformerEncoder(self.options)
            self.vectors = normalized(self.encoder.encode([c.text for c in self.chunks]),
                                      len(self.chunks))
        else:
            if encoder is not None:
                raise ValueError("an encoder requires retrieval.method: embedding")
            self.counts = [Counter(tokens(c.text)) for c in self.chunks]
            self.lengths = [sum(c.values()) for c in self.counts]
            self.avg_len = sum(self.lengths) / len(self.chunks) or 1.0
            self.df = Counter(t for c in self.counts for t in c)

    def build(self, event: dict[str, Any]) -> Context:
        query = json.dumps(event, sort_keys=True, allow_nan=False)
        if self.encoder is not None:
            vector = normalized(self.encoder.encode([query]), 1, len(self.vectors[0]))[0]
            scores = [math.fsum(a * b for a, b in zip(vector, row)) for row in self.vectors]
        else:
            query_terms = set(tokens(query))
            k1, b = self.options.bm25_k1, self.options.bm25_b
            scores = [sum(math.log(1 + (len(self.chunks) - self.df[t] + .5) / (self.df[t] + .5))
                          * count[t] * (k1 + 1)
                          / (count[t] + k1 * (1 - b + b * self.lengths[i] / self.avg_len))
                          for t in sorted(query_terms) if count[t])
                      for i, count in enumerate(self.counts)]
        selected = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], i))[
            :self.options.top_k]
        chunks = [self.chunks[i] for i in selected]
        return self.finish("\n\n".join(f"SOURCE {c.id}\n{c.text}" for c in chunks), event,
                           tuple(c.id for c in chunks), {
            "strategy": "retrieval", "source_id_kind": "chunk",
            "retrieval": self.options.model_dump(), "total_chunks": len(self.chunks),
            "encoder": type(self.encoder).__name__ if self.encoder is not None else None,
            "selected_rule_ids": covered_rules(self.documents, chunks),
            "chunks": [{"id": c.id, "document": c.document, "start": c.start, "end": c.end,
                        "rule_ids": list(c.rule_ids), "score": scores[i]}
                       for c, i in zip(chunks, selected)]})


class CompiledContextStrategy(BaseContextStrategy):
    def __init__(self, root: Path, *, config: ContextConfig | None = None,
                 token_counter: TokenCounter | None = None):
        super().__init__(root, config=config, token_counter=token_counter)
        if not self.rules:
            raise ValueError("no annotated RULE records or structured lookup rows to compile")

    def build(self, event: dict[str, Any]) -> Context:
        selected = [r for r in self.rules if all(
            k in event and type(event[k]) is type(v) and event[k] == v
            for k, v in r.when.items())]
        return self.finish(json.dumps([r.as_dict() for r in selected], sort_keys=True), event,
                           tuple(r.id for r in selected), {
            "strategy": "compiled_context", "source_id_kind": "rule",
            "compiler": "annotated-rules-v1", "predicate_semantics": "typed-scalar-equality",
            "total_rules": len(self.rules), "selected_rule_ids": [r.id for r in selected],
            "references": {r.id: r.as_dict()["references"] for r in selected}})


# Preserve the original public names and positional top_k constructor.
FullContext = FullContextStrategy
Retrieval = RetrievalContextStrategy
CompiledContext = CompiledContextStrategy


def make_strategy(name: str, root: Path, top_k: int | None = None, *,
                  config: ContextConfig | None = None,
                  token_counter: TokenCounter | None = None,
                  encoder: EmbeddingEncoder | None = None) -> ContextStrategy:
    if name in {"full_context", "prefix_cached_full_context"}:
        return FullContextStrategy(root, config=config, token_counter=token_counter,
                                   prefix_cached=name == "prefix_cached_full_context")
    if name == "retrieval":
        return RetrievalContextStrategy(root, top_k, config=config,
                                        token_counter=token_counter, encoder=encoder)
    if name == "compiled_context":
        return CompiledContextStrategy(root, config=config, token_counter=token_counter)
    raise ValueError(f"unknown strategy: {name}")
