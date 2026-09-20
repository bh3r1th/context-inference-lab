# Benchmark orchestration

```bash
contextbench validate configs/experiment.yaml
contextbench run configs/experiment.yaml
```

The default matrix is four strategies (`full_context`, `prefix_cached_full_context`,
`retrieval`, `compiled_context`) by concurrency `[1, 8, 32]`. It runs 12 cells sequentially,
with bounded asynchronous request workers inside each cell. Override `strategies` or
`concurrency` to run a smaller matrix. Set `event_ids` to select a subset of the dataset.

## Configure the endpoints

`server` supplies the default endpoint and model settings. `strategy_servers` overrides
the complete server configuration for a specific strategy. The supplied experiment
config uses YAML anchors to route cached full context to port 8001 and the other
strategies to port 8000. Fill in model revisions, vLLM versions, and exact launch commands
for both endpoints before running. Both should serve the same model/revision for a fair
comparison; each endpoint's settings and discovery results are preserved in metadata.

Prefix caching must be enabled on the cached server. The client records its declared
setting and never toggles it, restarts a server, or clears a cache. A cached cell whose
selected server declares `prefix_caching: false` is recorded as a setup failure; the
remaining cells continue. A single cache-enabled server is also supported, but its
cache setting applies to every strategy using it. Do not interpret that configuration
as a comparison against uncached baselines. `configs/cached.yaml` still supports an
independent cached-only run.

For the controlled prefix-cache comparison, use `configs/prefix-cache.yaml` after starting
separate servers with `--no-enable-prefix-caching` and `--enable-prefix-caching`. Use
`configs/prefix-cache-warm.yaml` for the warm-cache scenario; it applies the same strategy and
concurrency matrix while recording warmups separately. Cold-cache runs require a fresh
cache-enabled server process because the client never clears a live vLLM cache. Cache hits are
reported only from returned cached-token usage or server prefix-cache counters, never inferred
from latency.

## Workload, warmup, and reproducibility

`repetitions` controls how often every selected event appears in each measured cell.
The runner constructs `(event, repetition)` pairs, shuffles them once with a local RNG
using `seed`, then reuses the exact same ordered plan for all cells. Request seeds are
`seed + request_index`, independent of concurrency. JSONL records are appended in
completion order; use `request_index` to recover planned admission order.

Before each measured cell, `warmup_requests` requests cycle through the beginning of
the same workload, using that cell's worker limit. All warmups finish before measurement
starts. Warmup latency, failures, and token usage never enter measured aggregates.
Warmup failures remain in their own log and do not prevent measured attempts.
Set `warmup_requests: 0` to disable warmup. Cache state can persist between cells;
the runner does not promise that each cell starts with a cold cache.

Every invocation creates a unique `run_id` and a new directory. `experiment_id` defaults
to a generated ID, or can be supplied to group several unique runs. Cell and request IDs
include their run, strategy, concurrency, phase, and request index. Existing run directories
and raw files are never reused. `workload.json` contains event inputs;
`workload_manifest.json` records each index, event ID, repetition, and request seed.
Dataset and client-source hashes, configuration, timestamps, and endpoint/model/GPU
metadata remain available for replay.

## Failure behavior and raw records

Context-building, inference, and evaluation failures produce per-request records with
`error` and `error_stage`. All available response text and measurements are retained,
including results whose evaluation fails. Workers continue to subsequent events.
Strategy preparation failures produce failed cell summaries with `planned_requests`,
`unattempted_requests`, an error reason, and null batch measurements; they do not create
fake request measurements. Later cells and strategies still run.

Discovery failures are recorded without preventing requests to an endpoint that may
still serve completions. Telemetry errors are also isolated. Run status is `completed`
or `completed_with_errors`; each cell has its own status. `error_count` counts entries
in the run error log, while request failure counts live in each cell's summary.
An interrupted run saves summaries for the active partial cell and all completed cells,
marks itself `interrupted`, and propagates cancellation. Invalid datasets are rejected
before execution. Filesystem/persistence failures stop further work because continuing
without saving raw measurements would violate the logging contract.

Raw request, warmup, and telemetry logs are flushed as records arrive. Summaries are
checkpointed atomically after every cell and at shutdown. The original completion-order
JSONL is never rewritten when CSV rows are sorted or summaries are recomputed.

## Outputs and metric definitions

Each run contains:

- `metadata.json`, `dataset/`, `workload.json`, `workload_manifest.json`, and output schema.
- `errors.jsonl`: setup, discovery, batch, and telemetry errors with timestamps and scope.
- `summary.json` and `summary.csv`: one aggregate row per processed matrix cell.
- `report.json` and `report.md`: measured strategy/concurrency comparison artifacts.
- `<strategy>-c<N>/requests.jsonl` and `requests.csv`: raw measured request results.
- `<strategy>-c<N>/warmup.jsonl` and `warmup.json`: separate raw warmup results.
- `<strategy>-c<N>/telemetry.jsonl` and `summary.json`: samples and cell aggregate.

| Aggregate | Definition |
| --- | --- |
| TTFT/latency p50, p95, p99 | Linear-interpolated quantiles over observed values from successful request attempts, with sample counts |
| Batch duration | Measured request pipeline wall time, including context building, evaluation, and logging; excludes setup, warmup, and boundary scrapes |
| Requests/sec | Successful measured attempts divided by batch duration |
| Average context tokens | Mean of measured strategy `context_token_count` values |
| Context reduction | Relative to the `full_context` average at the same concurrency |
| Critical-rule miss rate | Requests with one or more missed critical rules divided by measured requests |
| KV-cache usage / GPU memory | p95 of observed server KV series / local GPU memory samples; `n/a` when unavailable |
| Attempts/sec | All measured request pipeline attempts divided by batch duration |
| Input/output token throughput | Successful requests' server token totals divided by batch duration; null if any successful request lacks usage |
| Average input/output tokens | Mean of observed server counts across measured attempts, including failed requests; observed/missing counts accompany the mean |
| Failure rate | Measured pipeline attempts with an error divided by measured attempts |

Wrong answers and schema-invalid model outputs remain correctness failures; they are
not transport/pipeline failures unless an error is reported. Setup failures have no
measured attempts, so their failure rate and batch duration are null. Unattempted events
are reported separately. Missing usage and timings remain null with reasons, never zero
or tokenizer-based substitutions. The CSV expands TTFT/latency percentiles and token
rates into columns; JSON retains complete nested provenance and metric coverage.
