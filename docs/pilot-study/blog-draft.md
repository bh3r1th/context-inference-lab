# What 180 LLM requests on Cloud Run taught me about context and correctness

> Draft for author review. This article reports a completed pilot study. The
> originally contemplated full benchmark was not executed.

I moved a context-inference experiment to Google Cloud Run Jobs so the work could
run independently of my laptop. The eventual green task statuses were useful,
but they were only one part of the result.

Across the selected pilot runs, all 180 measured requests completed and returned
valid JSON. Only 63 matched the expected decision and monetary limit. When I
also required the correct rule citations, that count fell to 36.

The experiment involved **ten unique synthetic cases, repeated across 18 cells**,
not 180 independent business decisions. That distinction matters throughout this
article. These are small pilot observations, not a production quality assessment
or a statistical ranking of context strategies.

## The question and the workload

The task was deliberately concrete: given an expense event and a set of policies,
return a JSON decision with an event ID, approve/reject/abstain, a monetary limit,
the relevant rule IDs, and a reason.

The fictional policies included city limits, department caps, restricted-vendor
controls, emergency exceptions, and incomplete evidence requiring abstention.
The pilot selected the first ten cases from a seeded 60-event synthetic dataset.
It repeated those cases across strategy, concurrency, scaling, and cache conditions.
Expected answers were used for evaluation, not given to the context builders.

I compared four ways to construct the model input:

| Strategy | What the model receives |
| --- | --- |
| Full context | All source documents and lookup data |
| Prefix-cached full context | The same input content, served with prefix caching enabled |
| BM25 retrieval | Up to eight paragraph chunks selected using keyword scoring |
| Compiled context | Annotated source rules selected by deterministic event predicates |

The compiled strategy needs a particular qualification: the sources already had
structured rule annotations. The compiler selected applicable rules; it did not
extract reliable knowledge from arbitrary prose, predict the answer, or read the
evaluation file. Results from this setup do not establish that general enterprise
documents can be compiled without information loss.

## The serving setup

The Cloud Run job used one NVIDIA L4, eight vCPUs, and 32 GiB of system memory.
The model was `Qwen/Qwen2.5-7B-Instruct`, pinned to revision
`a09a35458c702b33eeacc393d103063234e8bc28`.

The successful runtime reported vLLM `0.29.0+cu129`, Torch `2.13.0+cu129`, and host
driver `580.126.20`. The model ran in bfloat16 with a 72K context configuration,
structured JSON output, and a 256-token output cap.

Each task launched a fresh vLLM server, ran one complete pilot cell, and uploaded
its configuration, logs, raw responses, telemetry, and summaries to Cloud Storage.
Cloud Run parallelism was one. The client concurrency inside each cell was one
or eight, independently of that task-level setting.

The image contained both a smoke plan and a pilot plan. Job settings selected
the plan, and the Cloud Run task index selected a cell. Reusing the same image
could therefore run one smoke task or 18 pilot tasks. A later single-cell selector
let me rerun only the failed pilot cell.

## Getting a server to start was its own experiment

Several failures happened before useful inference.

An initial runtime failed CUDA initialization with error 803. The working setup
used the official CUDA 12.9 vLLM variant and native host driver libraries. Because
both the runtime variant and compatibility-library setting changed, I cannot
attribute that recovery to the CUDA version alone or claim all CUDA 13 setups
would fail on this hardware.

The image also exposed a version-provenance detail: its Python package version
was `0.29.0+cu129`, while the server's version endpoint reported `0.29.0`. The
deployment checks needed to recognize that specific official build and retain
the full installed package version in the task artifacts.

The next failure was a small KV-cache shortfall. At a GPU memory utilization
setting of 0.90, vLLM reported 3.84 GiB available for KV cache, while the configured
73,728-token sequence length required 3.94 GiB. Raising the budget to 0.92 produced
4.29 GiB of available KV cache. The context length, model, and output cap stayed
the same; CUDA graph memory profiling remained enabled.

After that, the model started, but environment validation sent an invalid empty
`Authorization: Bearer ` header to the local server. Omitting the header when no
API key was configured fixed that client-side issue.

Those failures were different problems. A healthy GPU did not prove the serving
configuration fit, and a listening server did not prove the client could send a
valid request.

Literal excerpts and source hashes for these failures and the startup deadline
are preserved in the [troubleshooting evidence](deployment-history.json).

## Three different timeout settings mattered

The Cloud Run GPU task had a one-hour limit. That was separate from the
launcher's model-readiness deadline and the HTTP client's read timeout.

One later attempt reached the default 15-minute model-startup deadline before
the server became ready. It failed before the benchmark sent any requests. That
was a startup-time problem, not evidence that the HTTP request timeout was too low.

The most informative pilot failure occurred after the server was already ready:
the 32K scaling cell at client concurrency eight. Seven measured requests
completed. Three recorded `ReadTimeout` near the configured 120-second read limit
while waiting for streamed content.

The server logs showed one running request and several waiting or deferred
requests. The cache held about 80,240 tokens, while these prompts averaged roughly
40,591 actual model input tokens. Eight overlapping client requests did not mean
the GPU could keep eight long sequences active together.

I preserved that failed attempt and reran the entire cell with a 600-second read
timeout. I did not combine its seven successful responses with three replacement
responses. The rerun retained the dataset, seeds, concurrency, warmup and model
settings, and completed all ten measured requests.

Its p95 time to first content was about 188 seconds, and p95 request latency was
about 197 seconds. Those observations explain why 120 seconds of read inactivity
could be insufficient. A read timeout is not a total request deadline, however:
receiving streamed data can keep a longer request alive.

The recovery also did not establish better answer quality. Only one response in
that successful ten-request rerun passed the full decision-and-citation contract.

## What the enterprise pilot measured

The following table shows the enterprise comparison at client concurrency one.
Each row contains ten measured responses. Input tokens come from server usage;
TTFT is client time to first nonempty content, not a direct measurement of pure
prefill time.

| Strategy | Mean input tokens | TTFT p95 | Decision + limit correct | Fully correct with citations |
| --- | ---: | ---: | ---: | ---: |
| Full context | 32,033 | 12.45 s | 2/10 | 2/10 |
| Prefix-cached full context | 32,033 | 0.24 s | 2/10 | 2/10 |
| BM25 retrieval | 939 | 0.34 s | 4/10 | 2/10 |
| Compiled annotated rules | 564 | 0.26 s | 9/10 | 2/10 |

![Input size and time to first content in the enterprise pilot](figures/enterprise_context_latency.png)

The prefix-cached enterprise cells included a configured warmup. Their input
content was unchanged, but their cache state and server configuration differed
from the uncached baseline. The separately named cold-prefix cells began with
fresh servers; subsequent requests in a cell could reuse the prefix. A cold
start did not make every request cold.

The shorter-input strategies and caching were associated with lower first-content
latency in these measurements. That is useful evidence for follow-up work, but
the quality columns prevent me from describing the rows as equal-quality speedups.
Ten observations per cell also provide a limited basis for p95 comparisons.

## A decision score and a citation score answer different questions

The compiled strategy illustrates why I kept both scores. It matched the expected
decision and limit in nine of ten cases at both concurrency levels. But only two
of ten responses at each level had all required citations and no unrelated ones.

For one restricted-vendor case, the expected output was a rejection with a limit
of USD 350 and citations to `R_BASE`, `L_CITY_0001`, and `R_VENDOR`. The compiled
response returned the right rejection and limit but cited only `L_CITY_0001` and
`R_VENDOR`. It passed decision correctness and failed strict correctness.

That does not make all the pilot's failures citation problems. Across the 180
selected responses, 27 had the correct decision/limit but failed citations, while
117 had an incorrect decision or limit. Full-context responses included wrong
limits, unjustified rejection, and incorrect handling of incomplete evidence.

![Decision correctness and strict citation correctness at concurrency one and eight](figures/enterprise_quality.png)

The strict citation contract was part of the workflow being measured. I retained
it and reported the decision-only score alongside it, rather than changing the
definition after seeing the outputs. Free-text reasons were required to be
nonempty but were not semantically graded.

The [response review](quality-review.md) includes the expected and actual outputs
for concrete examples, including successful cases. The aim is to make the
negative results inspectable, not to explain them all with one presumed model
failure mode.

## I also found a bug in my own coverage reporting

The raw `minimum_safe_context` field claimed that full context was missing
required evidence. The bookkeeping compared different namespaces: full-context
source IDs were document filenames, retrieval IDs were chunk locations, and the
expected IDs were rule names.

The context builders already recorded complete selected rule IDs. Comparing those
with the expected rule IDs showed that 162 of 180 contexts contained every required
rule. The 18 incomplete contexts were retrieval results. In the enterprise
retrieval comparison, only two of twenty contexts contained all required rules.

I corrected that calculation offline and preserved the original fields beside
the derived values. Future runs now expose the clearer name
`required_context_covered`; the old name remains a deprecated compatibility alias.

This correction did not turn an incorrect answer into a correct one. Strict
answer correctness stayed at 36 of 180. Rule presence is neither proof that a
model followed the rules nor evidence of a minimum safe context size.

## Why I am stopping at the pilot for this article

The selected tasks' application-recorded durations summed to 3.78 hours. Their
measured request batches summed to only 24.4 minutes. The remainder included
fixture generation, GPU/model initialization, warmups, artifact work and cleanup.
Those numbers are not a Cloud Billing measurement and do not include all platform
scheduling or image-start time.

More importantly, the full scaling configuration specifies 600 requests per cell:
60 events repeated ten times. The two successful 32K pilot cells took about four
minutes each for ten requests. Simple linear scaling suggests roughly four hours
of measured work for a full cell, before startup.

That is an estimate from a small pilot, not a full-run measurement. But it is a
large mismatch with a one-hour Cloud Run task budget. More tasks do not
automatically split an existing cell. Partitioning requests would change the
admission and cache history, so it needs an explicit protocol decision. Keeping
whole cells would call for a longer-lived GPU execution platform.

The pilot also did not cover client concurrency 32 or the 16K and 64K scaling
conditions. The size labels use a regex counter: the nominal 32K condition already
produced about 40.6K actual model input tokens. I have not established model-token
fit for the full 64K case against the configured 73,728-token limit.

## What is reproducible, and what remains incomplete

The [study package](README.md) includes the explicit attempt selection, input
hashes, effective run configurations, per-cell tables, derived response evidence,
the coverage correction, and scripts to reproduce the analysis without a GPU.
The original failed attempt remains part of that record. The successful rerun is
labeled with its different timeout.

I checked archived ground truth against the dataset's separate consistency oracle
and independently rescored the responses. The analysis reproduces the recorded
answer scores; it is not a new inference run.

There is a provenance gap: the uploaded artifacts did not record the complete
deployed image digests or exact execution commits. Historical client files match
a known repository commit by content, and the model revision and runtime versions
are recorded, but those facts do not identify the entire built image. Recovery
commands are included in the study guide. Until that metadata is recovered, I
cannot claim exact image-level reproduction. I also have no measured dollar-cost
result to report.

This pilot gave me a working deployment path, evidence of queueing on long inputs,
and a clear separation between input efficiency, structured output, and answer
correctness. A fuller study needs more held-out cases, repeated trials, an agreed
full-run dataset, and an execution plan that fits the actual workload. Those are
future conditions to test; the pilot's useful result is already documented.
