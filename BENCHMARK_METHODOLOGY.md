# Benchmark Methodology

## Scope

ContextBench is a finite-batch benchmark of context construction and inference serving. Every
strategy in a matrix receives the same shuffled `(event, repetition)` plan, request seed, expected
answer key, concurrency cell, output cap, decoding settings, and model revision. Strategy-specific
endpoints may differ only in serving URL and the explicitly tested prefix-cache setting.

The static policy/document material is placed in the system message. Event JSON is a separate user
message after that material. Thus `full_context` and `prefix_cached_full_context` have token-identical
static prefixes across events; the latter declares a cache-enabled vLLM endpoint. Prefix caching
changes block reuse in the serving engine and cache observations. It does not change the event,
prompt content, model, decoding parameters, answer key, or correctness definition.

Launch the cached endpoint with the explicit vLLM option `--enable-prefix-caching` and launch the
baseline with `--no-enable-prefix-caching` (or the equivalent setting for the installed vLLM
release). Record both complete commands in the config. The client does not toggle or clear either
server cache.

## Measurements

Each request is flushed to raw JSONL with event ID, seed, context/prompt hashes, response text,
usage, server metrics, SSE arrivals, errors, evaluation, and provenance. Client TTFT is monotonic
time from HTTP request start to the first nonempty content or reasoning delta. Client latency ends
when the stream closes or errors. Server TTFT, queue, decode, and token-rate metrics are retained
separately when vLLM returns them. Batch duration is wall time for the measured worker pipeline,
including context construction, inference, evaluation, and request-log writes, but excludes setup,
warmups, and telemetry boundary scrapes. Client queue time is admission-to-worker dispatch time.

Token totals use server response usage only. Context token counts are strategy-side provenance and
are never substituted for missing server usage. p50, p95, and p99 use linear interpolation over
successful measured request observations and include sample counts. Failed requests remain in raw
logs, failure rates, quality denominators, and missing-request accounting; failed requests are not
included in successful timing percentiles.

Warmups run before measurement and are stored separately. They are excluded from all measured
summaries, while warmup failures remain visible. Prefix-cache cold and warm scenarios use separate
configs and run directories; cold requires starting a fresh cache-enabled server process (the
client never clears a live cache), while warm uses recorded warmup requests before measurement.
Each request records the static-prefix hash/token count and dynamic event hash, making the reusable
layout auditable. Cache hits are accepted only from returned usage or server cache counters; latency
is never used as a cache-hit proxy.

## Context-size scaling

The scaling runner creates equivalent fixed-event workloads with approximately 2K, 8K, 16K, 32K,
and 64K input-token targets, runs concurrency 1, 8, and 32, and stores results under
`results/context-scaling`. Model, output cap, task structure, generation settings, answer key,
and event order remain fixed. The target is the rendered full-context prompt measured by the
repository's regex tokenizer; the smallest case may be above 2K because the fixed normative rules
must remain present. The padding is non-normative reference text, so this experiment isolates
serving behavior from retrieval quality rather than measuring a useful retrieval policy.

## Reproduction and limitations

`metadata.json` captures resolved configuration, dataset and source hashes, model and revision,
declared and observed vLLM versions, endpoint discovery, package/platform data, GPU metadata, and
the server launch command. Use the pinned GPU requirements and Dockerfile, validate the environment,
then run `scripts/run_benchmark.py`. GPU telemetry is local-host sampling unless an external server
collector is used. Prometheus samples can miss short peaks and may include other clients.

The supplied GPU image assumes NVIDIA CUDA 12.4.1 with cuDNN and Python 3.11, and pins vLLM to
`0.8.5.post1` in `requirements-gpu.txt`. vLLM/CUDA compatibility must still be verified against
the selected GPU driver and model; the validator checks endpoint health and model accessibility
before benchmark requests begin.

The synthetic datasets are small and repeated events are not independent observations. Seeds do not
guarantee bitwise-identical GPU behavior across concurrency. Server scheduling, thermal state,
cache state, network conditions, endpoint ordering, and model implementation can confound results.
Minimum Safe Context is the smallest observed configuration meeting the selected quality constraints;
it is not a proof of safety outside this dataset and configuration.

## Single-GPU sequential workflow

On a single GPU, run the pilot and full benchmark in two phases so only one vLLM server occupies
VRAM at a time:

```bash
python scripts/run_pilot.py --phase uncached
python scripts/run_pilot.py --phase cached
python scripts/validate_pilot.py results/pilot/enterprise/uncached \
	results/pilot/enterprise/cached results/pilot/context-scaling/uncached/runs \
	results/pilot/prefix/cached/pilot-prefix-cold \
	results/pilot/prefix/cached/pilot-prefix-warm --required-phases uncached,cached
python scripts/run_full_benchmark.py --phase uncached
python scripts/run_full_benchmark.py --phase cached
```

Both explicit phases map their selected server to `localhost:8000`; `uncached` selects the three
non-cache strategies and scaling, while `cached` selects prefix-cached full context and its cold/warm
experiments. The default `--phase all` preserves the original two-endpoint behavior. Split results
are merged by the existing recursive analysis/report tooling and the full-run manifest.