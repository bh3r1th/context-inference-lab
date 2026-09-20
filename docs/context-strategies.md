# Context strategies

Build and inspect context without an inference server:

```bash
contextbench inspect-context --dataset data/enterprise --strategy full_context --event-id EV-000000
contextbench inspect-context --dataset data/enterprise --strategy retrieval --config configs/context/bm25.yaml --event-id EV-000004
contextbench inspect-context --dataset data/enterprise --strategy compiled_context --event-id EV-000003 --format prompt
contextbench inspect-context --dataset data/enterprise --strategy retrieval --event-id EV-000004 --top-k 3 --output context.json
```

Use `python -m contextbench.cli` in place of `contextbench` in a source checkout with
`PYTHONPATH=src`. `--event-file event.json` accepts one input object instead of
`--event-id`. Paths resolve from the working directory. `--config` accepts context-only
YAML, with no server settings. `--top-k` and `--retrieval-method` override the file.
Invalid settings and missing/duplicate event IDs produce a nonzero exit.

## Typed contract

```python
from pathlib import Path
from contextbench.context import Context, ContextStrategy, make_strategy

strategy: ContextStrategy = make_strategy("compiled_context", Path("data/enterprise"))
context: Context = strategy.build({"event_id": "EXAMPLE", "city": "Harbor-0001"})
print(context.prompt, context.token_count, context.selected_source_ids)
```

Every strategy implements `build(event: dict[str, Any]) -> Context`:

| Field | Meaning |
| --- | --- |
| `text` | Selected static sources, retained for prefix analysis and compatibility |
| `prompt` | Complete plain-text rendering of instructions, sources, and event |
| `messages` | Corresponding system/user messages; the event appears once |
| `token_count` | Count of `prompt` using the configured token counter |
| `source_ids` / `selected_source_ids` | Selected document, chunk, or rule IDs |
| `metadata` | Strategy, ID type, tokenizer, selected rules, settings, and provenance |

JSON inspection emits these fields. The existing backend consumes `messages` directly.
The runner records `context_token_count` and `context_metadata` alongside server usage.
Context code never imports or calls an inference backend.

## Strategies and provenance

`FullContextStrategy` includes all `.md`/`.txt` documents under `documents/` in stable
path order and every `lookup.json` row. Lookup rows already rendered in a document
remain visible in this full-context baseline. Source IDs are document names. The
static text and system message stay identical across events; the user message changes.
`prefix_cached_full_context` aliases the same construction.

`RetrievalContextStrategy` splits at paragraph boundaries, using overlapping character
windows for paragraphs longer than `chunk_chars`. `overlap_chars` must be smaller than
`chunk_chars`. Chunk IDs are `document:start-end`, with zero-based, end-exclusive Unicode
character offsets into loaded text or rendered lookup rows. Identical sources/settings
produce identical IDs. Selected chunks include scores, offsets, and overlapping rule IDs.
`metadata.selected_rule_ids` counts only rules whose entire annotation is covered by
the union of selected chunks; a partial chunk cannot count as complete rule coverage.

The index is built once; each event's sorted JSON is the retrieval query. Top-k is capped
at corpus size. Ties use source order; zero-score chunks can fill top-k.

- `bm25`: Unicode word tokens, configurable `bm25_k1`/`bm25_b`, no optional dependencies.
- `embedding`: optional Sentence Transformers encoder, CPU by default, cosine similarity.
  Document vectors are computed once and query vectors per event. Invalid dimensions,
  nonfinite values, and zero vectors fail explicitly. Inputs above the model's token
  limit fail instead of silently truncating; reduce chunk size or use a larger model
  limit. There is no fallback to BM25.

`CompiledContextStrategy` preprocesses `RULE {...}` annotations and lookup rows into a
structured catalog preserving ID, predicate, fact, document, section, and line reference.
Typed scalar equality matches predicates: `true` differs from `1`, and missing differs
from explicit `null`. Unconditional rules are included for every event. Identical duplicate
IDs merge all references; conflicting definitions fail. Selection conservatively includes
global completeness rules and matching exceptions. It does not compute event decisions.

Compilation supports annotated sources, not extraction from arbitrary prose. Full and
retrieval strategies also support unannotated prose. Compilation fails clearly if there
are no structured facts. Supported lookup rows are `id/city/limit_usd` (the benchmark
schema) or `id/when/fact`. Lookup references use one-based row numbers in rendered data;
document references use physical line numbers after newline normalization.
No strategy reads `expected.json`, `records.json`, `sources.json`, or `manifest.json`.

## Optional embeddings and tokenizers

```bash
python -m pip install -e '.[embeddings,tokenizers]'
contextbench inspect-context --dataset data/enterprise --strategy retrieval --config configs/context/embedding.yaml --event-id EV-000003
contextbench inspect-context --dataset data/enterprise --strategy compiled_context --config configs/context/tokenizer.yaml --event-id EV-000003
```

Example configs use `local_files_only: true`. Supply cached models or change it to
`false` to allow downloading on first use. Model names accept repository IDs or local
paths. Pin `embedding_revision` and tokenizer `revision` for reproducibility. Remote
model code is disabled. The encoder adapter follows the
[Sentence Transformers API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).

The default regex counter counts Unicode words and punctuation and labels its result
`is_estimate: true`; this is not an LLM token measurement. The Hugging Face counter uses
the configured tokenizer without truncation or added special tokens, yielding an exact
count for the rendered prompt text under that tokenizer. It excludes chat-template
framing, so it can differ from server `input_tokens`. See the
[tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer).
`metadata.tokenizer` identifies the method, model/revision, estimate status, and scope.
`TokenCounter` and `EmbeddingEncoder` protocols also support injected implementations.

## Experiment configuration

Use the same configuration under `context` in experiment YAML:

```yaml
context:
  retrieval:
    method: bm25
    top_k: 8
    chunk_chars: 1200
    overlap_chars: 120
    bm25_k1: 1.5
    bm25_b: 0.75
  tokenizer:
    method: regex
```

Legacy `retrieval_top_k` remains supported. If both it and `context` are supplied,
their top-k values must agree. Configuration/source snapshots preserve reproducibility.

Tests cover all strategies, source isolation, required-rule coverage, provenance,
chunking, ranking/ties, token accounting, inspection commands, config validation, and
backend message consistency. Embedding tests use controlled vectors and a mocked adapter;
they establish plumbing, not semantic quality or performance of a downloaded model.
