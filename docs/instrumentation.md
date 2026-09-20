# Inference instrumentation

The client sends streamed Chat Completions to a configurable vLLM/OpenAI-compatible
endpoint. It uses `temperature: 0`, one completion, a recorded request seed, and
`stream_options.include_usage: true`. Model name, timeout, and maximum output tokens
are configurable under `server`. Benchmark concurrency is a list of positive worker
counts, such as `[1, 8, 32]`; each cell processes the same seeded finite workload.

Structured output defaults to vLLM's `structured_outputs.json`. Set
`structured_output_mode: json_schema` for the OpenAI-compatible `response_format`
schema request, or `structured_output: false` to disable constraints. Unsupported
server options fail visibly; there are no silent retries or format fallbacks.

## One-event smoke test

Run the local HTTP/SSE fixture test, requiring neither a GPU nor model weights:

```bash
python -m pytest tests/test_vllm_smoke.py -q
```

It sends one synthetic event through the context strategy, transport, measurement,
evaluation, and artifact pipeline. Returned token counts and server timings are
explicit fixture data, not evidence of real inference performance.

For a live smoke run, set `server.model`, `model_revision`, `vllm_version`, and
`launch_command` in `configs/smoke.yaml`, then run:

```bash
contextbench validate configs/smoke.yaml
contextbench run configs/smoke.yaml
```

The config selects only `EV-000`, with one request, concurrency one, and no warmup.
`event_ids` also works for larger experiment configs. Endpoint URLs may be local or
remote. Authentication reads the environment variable named by `api_key_env` and
does not write that key into artifacts.

## Measurement meanings

Each request stores flat numeric fields, a `metric_sources` mapping, and an
`unsupported` mapping explaining every unavailable numeric field. Unavailable values
are JSON `null`; CSV exports use literal `null`. Missing server usage is never filled
from tokenizer counts or character estimates. Context strategy token counts remain
separate metadata and are not inference measurements.

| Field | Source and scope |
| --- | --- |
| `input_tokens`, `output_tokens` | Server `usage.prompt_tokens` / `completion_tokens` |
| `ttft_s` | Client monotonic time from HTTP start to the first nonempty content or reasoning delta |
| `latency_s` | Client monotonic time from HTTP start through stream close/error |
| `stream_generation_s` | Observed first-to-last nonempty output delta span; null with fewer than two deltas |
| `server_ttft_s` | Server `metrics.time_to_first_token_ms`, converted to seconds |
| `decode_time_s` | Server `metrics.generation_time_ms`, converted to seconds |
| `server_queue_s` | Server `metrics.queue_time_ms`, converted to seconds |
| `server_mean_itl_s` | Server `metrics.mean_itl_ms`, converted to seconds |
| `server_output_tokens_per_s` | Server `metrics.tokens_per_second`; includes the inference interval, not just decode |
| `decode_tokens_per_s` | `(output_tokens - 1) / decode_time_s`, only with server timing, usage, and a successful response |
| `output_tokens_per_s` | Successful request's server output count divided by its client end-to-end latency |
| `client_queue_s` | Client batch admission to worker dispatch; independent of server queue time |
| `cached_input_tokens` | Server `usage.prompt_tokens_details.cached_tokens` |
| `cached_input_fraction` | Reported cached tokens divided by a positive reported input count |
| `prefill_s` | Null: supported response fields do not isolate pure prefill |
| Request GPU/KV fields | Null: sampled host/server measurements cannot be assigned to individual requests |

Client timings include network buffering and scheduling. An SSE delta may contain
multiple tokens; it is never treated as one token. Server TTFT is kept separate from
client TTFT and is not relabeled as pure prefill. Missing decode timings stay null,
even when several text chunks were observed. Failed or truncated streams retain
partial text, usage, and raw events, but do not receive throughput rates.

Recent vLLM servers can expose the response `metrics` object with
`--enable-per-request-metrics`; streaming delivery uses the final usage chunk.
The adapter checks returned fields rather than guessing support from a version string.
Consult the [per-request metrics documentation](https://docs.vllm.ai/en/latest/features/per_request_metrics/)
for server requirements and the timing definitions. Older or compatible endpoints
without those fields remain usable with explicit unavailable values.

Batch summary rates use measured batch wall time: successful requests/sec, attempted
requests/sec, and successful input/output tokens/sec. Token rates are null if any
successful request lacks that token count. Token totals include observed and missing
request counts, and are null when no counts were observed. Latency summaries include
sample counts and availability reasons. Warmup and telemetry boundary scrapes are
outside measured batch time; context construction, evaluation, and request logging
are inside it. Raw warmup metrics are stored separately.

## GPU, KV, and prefix-cache telemetry

`telemetry.jsonl` preserves timestamps and the full Prometheus response. Scrapes run
before the workload, periodically, and after it. Recognized vLLM samples retain their
metric names and engine/model labels, including queue/decode/prefill histograms.
Those histograms are server aggregates and never populate request timing fields.
KV gauges retain native values. See the [vLLM production metrics reference](https://docs.vllm.ai/en/latest/usage/metrics/).

Prefix-cache hits/queries, their `_total` variants, external-prefix counters, and older
hit-rate gauges are retained. Counter interval deltas require matching finite series
at every sampled boundary with no observed decrease. Resets or missing observations
produce null deltas with a reason; a hit ratio additionally needs positive query delta
and valid hits. These are server-wide observations and may include other clients.
Raw metric units follow the serving version. GPU/KV sample quantiles are not
time-weighted averages; polling can miss short peaks.

`local_gpu_metrics: true` enables `nvidia-smi` on the client host. Samples include GPU
name, UUID, driver version, PCI address, utilization percent, and used/total MiB.
Unsupported individual fields remain null without discarding valid fields. Only use
local observations as serving hardware metadata when the server runs on the same host.
Remote GPU discovery is unavailable through the OpenAI API; the artifact records that
limitation explicitly. Disabled collectors and failed scrapes also retain reasons.

## Reproducibility artifacts

`metadata.json` records run timestamps, configured model/revision/vLLM version and
launch command, sampling parameters, the `/models` response, optional `/version`
discovery, GPU metadata availability, package/platform information, and dataset/source
hashes. Declared and observed versions remain separate. `version_url` can override
the version endpoint for a proxy installation.

Every `requests.jsonl` row retains request index, event ID, seed, admission/start/end
timestamps, HTTP status, response ID/model, source IDs, context/prompt hashes, raw
usage, raw server metrics, response text, reasoning text, and SSE data with monotonic
arrival offsets. `requests.csv` exposes the same records; nested fields are JSON.
An interrupted run preserves flushed records and marks the run failed. Failure paths,
missing metrics, numeric validation, cache resets, concurrency, and the one-event HTTP
smoke pipeline are covered by tests.
