# ContextBench

A small Python research repository for measuring how enterprise context design affects
LLM serving performance and answer quality:

```text
static documents + event JSON → context strategy → vLLM → JSON decision
                                                        ↓
                       raw measurements + evaluation → CSV / JSON / JSONL
```

The implemented path uses a real OpenAI-compatible streaming HTTP request to vLLM.
Tests also exercise the entire path against a local HTTP/SSE fixture server. Fixture
responses prove plumbing, not model performance. No GPU benchmark results are bundled.

## Quick start

Python 3.11+ is required for the client. Run commands from this repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
pytest -q
contextbench validate configs/experiment.yaml
contextbench schema
```

The checked-in smoke dataset is ready to use. For the larger enterprise workload with
six case types, seeded generation, source-level ground truth, and small/medium/large
configs, see [the dataset guide](data/README.md). Generate its small configuration with:

```bash
python -m contextbench.workload --config configs/datasets/small.json --output data/enterprise
```

Regenerate the original smoke fixture or increase its document size:

```bash
python scripts/generate_fixtures.py --paragraphs 80
```

Provision a separate Linux GPU environment for vLLM (no cloud deployment is included).
Install a specific vLLM release compatible with your CUDA and GPU environment, and record
that version. The adapter uses `structured_outputs.json`, rather than removed legacy
`guided_json` fields. Follow the [vLLM installation documentation](https://docs.vllm.ai/en/stable/getting_started/installation/)
and [structured output API](https://docs.vllm.ai/en/stable/features/structured_outputs/).

Example server launch, replacing `MODEL_COMMIT` with an immutable model revision:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --revision MODEL_COMMIT --dtype auto --max-model-len 32768 \
  --no-enable-prefix-caching --port 8000
```

Record the actual model commit, installed vLLM version, and full launch command in
`configs/experiment.yaml`; execution rejects the supplied metadata placeholders.
Choose a model/context limit and GPU memory budget that fit the complete rendered prompt
plus output. Document size is configurable; this client never silently truncates context.
The example model is illustrative, not a guarantee it fits your hardware. Use the same
model revision, tokenizer, chat template, dtype, parallelism, output cap, server seed, and
scheduler settings in every comparison. Disable unrelated server traffic.

```bash
contextbench run configs/experiment.yaml
```

The default matrix includes cached full context. Configure a second endpoint on port
8001 with prefix caching enabled via `strategy_servers` in `configs/experiment.yaml`.
For a standalone cached-only run, `configs/cached.yaml` is also available:

```bash
contextbench run configs/cached.yaml
```

The client does not toggle a global cache on a live server or reset a shared cache.
`prefix_caching` is an operator declaration, not remotely verified configuration. Keep
launch logs with results. The runner routes cached and baseline strategies to their
configured endpoints. vLLM's [automatic prefix cache](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
reuses matching prefix blocks; the strategy name alone cannot enable it.

If authentication is enabled, export `VLLM_API_KEY` in your shell. `.env.example` documents
the variable; files are not implicitly loaded. Keys are never copied into run metadata.
Config-relative dataset and output paths work regardless of the command's current directory.

## Architecture and extension points

See the [context strategy guide](docs/context-strategies.md) for typed results,
BM25/embedding configuration, token counting, provenance, and offline inspection:

```bash
contextbench inspect-context --dataset data/enterprise --strategy compiled_context --event-id EV-000003
```

```text
data/synthetic/                    generated documents, events, independent expected answers
src/contextbench/
  models.py                        ContextStrategy / InferenceBackend protocols, Decision schema
  context/strategies.py             full, BM25 retrieval, source-rule compilation and selection
  inference/vllm.py                 streaming transport, usage and client timing
  inference/runner.py               finite-backlog concurrent request workers
  evaluation/scoring.py             strict schema, exact decision fields, rule coverage
  metrics/                         quantiles, local GPU sampling, server telemetry
  experiments/                     validated config, warmup, orchestration, artifact export
  cli.py                           run / validate / schema
configs/                           uncached and cached experiment matrices
scripts/                           deterministic synthetic fixture generator
tests/                             unit and local HTTP integration tests
results/                           one uniquely named directory per run (gitignored)
```

`ContextStrategy.build(event) -> Context` has no serving dependency. Add a strategy in
`context/` and register it in `make_strategy` and the config literal. The returned context
contains source IDs so selections are auditable. `InferenceBackend.generate(context,
event, seed) -> InferenceResult` is the only backend interface consumed by the load runner.
To add a serving engine, implement that protocol and change orchestration's backend
construction. Backend-specific capability discovery stays in its adapter. No framework,
vector database, paid API, UI, or cloud service is required.

| Strategy | Construction | Experiment interpretation |
|---|---|---|
| `full_context` | All documents in filename order | Uncached full-input baseline |
| `retrieval` | Paragraph chunks, configurable BM25 or embedding cosine retrieval | Top-k retrieval can miss exceptions and dependencies |
| `compiled_context` | Parse annotated source rules once; select facts by event predicates | Deterministic compiler baseline, not an LLM knowledge extractor |
| `prefix_cached_full_context` | Exactly the same prompt content as full context | Requires separately launched cache-enabled server |

The fixed instructions and documents precede the event in the chat prompt. Events do not
alter the static prefix. Compilation is preprocessing, measured separately as `preparation_s`;
per-event selection is measured as `context_build_s`. Retrieval builds its index in preprocessing.
Expected answers are loaded only by orchestration/evaluation, never context strategies.
The annotated-source format makes extraction reliable by construction; results do not establish
that arbitrary enterprise prose can be compiled without information loss.

## Workload and correctness

All fixtures are synthetic. Four large policy/reference documents include nonbinding archive
notes, explicit rule records, and a city-limit lookup table represented as structured rule rows.
Five curated event cases cover normal approval, cross-document restricted-vendor rejection,
a buried medical exception, city-lookup rejection, and unknown-city abstention. Rules are
inserted midway through each document. Increase `--paragraphs` for size sweeps; the generator
overwrites only its known fixture filenames. The small curated set is a smoke experiment,
not a representative statistical sample of enterprise decisions. Repeated events are not
independent semantic observations; extend the generator and held-out cases for substantive studies.

The strict Pydantic `Decision` model rejects extra fields, incorrect types, negative limits,
and invalid decisions. Its JSON Schema is used for server-side constrained decoding and
written into every run. Disable `structured_output` explicitly to study unconstrained JSON.
Correctness requires matching event ID, decision, limit, and the exact set of applicable rule
IDs. Free-text reason must be nonempty but is not semantically scored. Rule recall and missed
critical IDs are citation-based proxies, not proof that the model internally followed a rule.
Transport failures and truncated completions cannot count as correct. Invalid JSON receives
zero coverage and all expected critical rules are reported as missed.

## Measurement contract

See the [instrumentation guide](docs/instrumentation.md) for field provenance,
availability rules, and the one-event smoke test (`configs/smoke.yaml`).

| Metric | Definition / availability |
|---|---|
| Input/output tokens | Server streaming usage, never character-based estimates; null if absent |
| TTFT | Client send start to first nonempty content/reasoning delta; server TTFT stored separately |
| End-to-end latency | Client send start through stream completion/error, excluding context preparation |
| Decode time / tokens/sec | Server generation time when returned; rate requires server timing and usage |
| Client queue | Time from finite-batch enqueue until a worker takes the request |
| Server queue, prefill | Queue from per-request server metrics when returned; pure prefill stays null |
| GPU utilization/memory | Optional local `nvidia-smi` samples; disabled by default for remote-server safety |
| KV-cache utilization | Known vLLM cache gauge samples, preserving engine labels, at server scope |
| Prefix/cache hits | Request cached-token usage when returned; server counter intervals remain server-scoped |
| Batch duration | Worker start through completion, evaluation and raw-record writes; excludes warmup and setup |
| Requests/sec | Successful transport completions / batch duration; attempts/sec also exported |
| Context size | Exact source characters/bytes; configured prompt token count (regex estimates labeled); server input tokens include chat framing |
| Quality | Strict validity, exact correctness, rule recall, missed critical rules |

Client TTFT measures the first observable output delta. SSE chunks can contain multiple
tokens, so decode throughput is never inferred from chunk timing. Returned per-request
server timings are recorded separately; unavailable metrics remain null with a reason
and source. Server histograms remain server-scoped. Capability is determined by returned
fields, not a version-string guess.

Telemetry preserves `/metrics` snapshots, including server queue/prefill histograms and cache
hit counters when the server exposes them. See [vLLM metric definitions](https://docs.vllm.ai/en/stable/design/metrics/).
Known KV gauge names are extracted; their native values are preserved (vLLM uses a fraction
for these gauges despite the `perc` suffix). Other version-specific metrics remain in raw text.
Missing gauges are marked unsupported. GPU/KV summaries are sample quantiles, not time-weighted
averages or per-request measurements. Sampling is approximate, can miss short peaks, and includes
boundary snapshots. Local GPU sampling describes the client host; enable it only when that host
is the server. For remote GPUs collect telemetry on the serving host or add an exporter adapter.

Latency p50/p95/p99 use linear interpolation over successful requests, with sample counts.
Errors remain in raw results and in quality denominators. Token totals include observation counts
so partial usage coverage is visible. Avoid interpreting p99 from a handful of observations.

## Experimental controls and artifacts

The [orchestration guide](docs/orchestration.md) describes the default 12-cell matrix,
per-strategy endpoints, failure isolation, deterministic workload plans, and summary columns.

This is a finite batch with a fixed backlog and configurable worker count, not a constant-rate
traffic generator. The same seeded event ordering and per-request seeds are reused for each
matrix cell. Warmup requests are outside measured batches and recorded separately. There are
no implicit retries. Two warmups in the cache configuration prime its document prefix; those
results describe warm-cache operation. For cold-cache studies, use one concurrency cell per
fresh server process and `warmup_requests: 0`. Cache state persists between cells otherwise.

The default matrix order is fixed. Repeat whole runs with a changed strategy/concurrency order
to assess thermal, scheduling and ordering bias. Pin model and serving artifacts, capture GPU
hardware/driver details and server logs, and freeze client dependencies with `pip freeze` for
archival experiments. Seeds alone do not guarantee bitwise GPU results at different concurrency.

Each run stores:

- `metadata.json`: resolved config, seed, declared model revision/launch settings, actual `/models`
  response, runtime package versions, dataset hashes, completion/failure status.
- `dataset/`, `workload.json`, `output.schema.json`: exact source/input/answer snapshots, request
  order, and response schema for replay.
- `<strategy>-c<N>/requests.jsonl`: flushed raw text, errors, source IDs, measurements and evaluation.
- `requests.csv`, `summary.json`: request table and aggregate metrics for each cell.
- `telemetry.jsonl`, `warmup.jsonl`, `warmup.json`: raw samples and warmup request results.
- Run-level `summary.json` and `summary.csv`: aggregate results checkpointed after every cell.
- `workload_manifest.json`, `errors.jsonl`: deterministic request/repetition plan and scoped errors.

The request output is flushed as requests finish; IDs/indexes allow reconstruction of order.
An interrupted run can therefore retain useful partial JSONL even if its final CSV is incomplete.
Runs use unique directories and do not overwrite prior results. Endpoint errors remain explicit;
discovery, warmup, request, and strategy failures are recorded while remaining work continues.
Persistence failures stop the run rather than continuing without raw logs.

## Validation

```bash
pytest -q
```

Tests cover prefix identity, source-only compilation, deterministic retrieval, exception selection,
schema strictness, exact scoring and critical misses, invalid experiment configs, quantiles,
HTTP errors, usage omission, incomplete streams, and full multi-strategy/multi-concurrency HTTP
execution into exported result files. The integration server returns labeled fixture answers;
run against your GPU-backed vLLM endpoint to establish real model quality and performance.
