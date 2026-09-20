# Publication Evidence

**Status: BLOCKED.** No publication claims are supported because no final validated benchmark run exists.

## Evidence Gate

- Pilot validation: `BLOCKED`
- Benchmark sanity report: `ISSUES FOUND: NO COMPLETED RUNS`
- Final analysis: `BLOCKED`
- Validated run IDs: none
- Measured major results: none

Accordingly, there are no supported values for context strategy, context tokens, context reduction, accuracy, critical-rule misses, TTFT p50/p95, throughput, KV-cache utilization, GPU memory, batch duration, concurrency, run ID, absolute improvement, or percentage improvement. No comparison against full-context baseline is calculated.

## Configuration, Not Measurement

The configured enterprise matrix specifies four strategies (`full_context`, `prefix_cached_full_context`, `retrieval`, `compiled_context`), concurrency `1, 8, 32`, and `10` repetitions over the configured enterprise dataset. These are planned settings only and must not be presented as executed results. Hardware, CUDA, vLLM, model revision, dtype, quantization, tensor parallelism, and max model length were unavailable on the execution host.

## Methodology Summary

The frozen benchmark uses the same seeded event/repetition plan and ground truth across strategies. Warmups are stored separately from measured requests. Request-level JSONL preserves response text, usage, timing, failures, evaluation, and telemetry provenance. Percentiles use linear interpolation over successful measured observations. Cache hits are accepted only from returned usage or server cache counters, never inferred from latency. Full-context and prefix-cached prompts use the same static policy prefix followed by dynamic event data.

## Variance and Failures

- Variance across repetitions: not measurable; no repetitions ran.
- Failed requests: no benchmark requests were sent.
- GPU/KV-cache collection: unavailable because no target GPU or vLLM endpoint was present.
- Throughput, latency, and quality aggregation: not performed on final results.

## Limitations and Contradictions

The target host had no `nvidia-smi`, no installed vLLM package, and no reachable vLLM endpoint. The workspace was not a Git checkout, so no commit SHA could be associated with a final run. No result can contradict or support the initial hypothesis because no validated measurement exists.

The blocked preflight evidence is preserved in [results/final/blocked-20260919T211910Z](results/final/blocked-20260919T211910Z). Generate a replacement publication package only after the pilot passes, the full benchmark completes, the sanity report passes with documented observations, and final analysis is generated from those validated artifacts.
