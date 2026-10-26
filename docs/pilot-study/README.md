# Cloud Run context-inference pilot

This directory closes the **pilot study**, not the originally contemplated full
benchmark. It contains 18 selected completed cells, 180 measured responses and
ten unique synthetic event cases. Low answer quality is retained as a result.
No additional GPU runs were performed to produce this review.

Start with:

- [Final results and limitations](FINAL_RESULTS.md)
- [Incorrect-answer and citation review](quality-review.md)
- [Blog draft for author review](blog-draft.md)
- [Publication checklist and editorial handoff](publication-checklist.md)
- [Explicit attempt selection](selection.json)
- [Per-cell measurements](results/cells.csv)
- [Derived response evidence](results/requests.jsonl)
- [Evidence hashes, run configurations and provenance gaps](results/evidence_manifest.json)
- [Hashed troubleshooting-log excerpts](deployment-history.json)
- [Machine-readable results](results/summary.json)
- [Figures in PNG and SVG](figures/)

The selected set uses 17 cells from `context-inference-lab-q2vr4` plus the entire
successful 32K/concurrency-eight rerun from `context-inference-lab-bpjvd`. The
original failed attempt remains in the manifest. Its seven successful responses
were not mixed with new responses to manufacture a complete cell. The rerun uses
a 600-second HTTP read timeout; the other 17 cells use 120 seconds.

## What changed after inference

Rule coverage originally compared document/chunk IDs with required rule IDs.
The correction compares `context_metadata.selected_rule_ids` with the expected
rule IDs and publishes `required_context_covered`. Derived records retain
`original_minimum_safe_context` and `original_missing_required_context_sources`.
Original ZIPs, raw run files, answer scores and measurements are unchanged.

Future runs use this corrected coverage calculation. The old `minimum_safe_context`
name remains a deprecated compatibility alias for rule presence; it does not
demonstrate a minimum context size or a correct answer. This software correction
was made **after** the archived GPU runs. No image rebuild is needed to reproduce
the offline review or read the published results.

## Reproduce the numerical review without a GPU

Use a checkout containing this study, with Git history containing the client
reference commit in `selection.json`. The uploaded run artifacts omit Git metadata;
the reference commit is a content match for client files, not proof of the full
execution/image commit.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

mkdir -p /tmp/contextbench-pilot-inputs
unzip /path/to/context-inference-lab-q2vr4-artifacts.zip \
  -d /tmp/contextbench-pilot-inputs
unzip /path/to/context-inference-lab-bpjvd-artifacts.zip \
  -d /tmp/contextbench-pilot-inputs

python -m scripts.finalize_cloud_pilot \
  --artifacts /tmp/contextbench-pilot-inputs \
  --selection docs/pilot-study/selection.json \
  --verify-manifest docs/pilot-study/results/evidence_manifest.json \
  --output /tmp/contextbench-pilot-reproduced

diff -ru docs/pilot-study/results /tmp/contextbench-pilot-reproduced
```

The exporter verifies raw file hashes, archived dataset hashes, historical client
source hashes, completion markers, raw/aggregate consistency, event/seed plans,
answer scores and the dataset's separate decision oracle. It writes only to the
requested derived output directory, outside the raw input tree. It fails on
checksum, selection, scoring or completion mismatches. It never runs inference,
updates GCS or creates the full-benchmark gate file `pilot_validation.json`.

To recreate the figures with the recorded local analysis dependencies:

```bash
python -m pip install -r docs/pilot-study/analysis-requirements.txt
MPLCONFIGDIR=/tmp/contextbench-matplotlib \
  XDG_CACHE_HOME=/tmp/contextbench-analysis-cache \
  python -m scripts.plot_cloud_pilot \
    --cells docs/pilot-study/results/cells.csv \
    --output /tmp/contextbench-pilot-figures
```

The analysis environment is recorded separately in `analysis-environment.json`.
It is not the GPU container environment. Figures have no confidence intervals:
ten repeated cases per cell do not establish a population latency distribution
or statistically reliable differences between configurations.

The downloadable evidence kit also preserves the two original artifact ZIPs and
the four separately uploaded troubleshooting-log exports referenced in
`deployment-history.json`. Their hashes are recorded; the unrelated Cloud Shell
home-directory ZIP is excluded. The included Git bundle preserves source history
for offline checkout, including the historical client reference commit.

## Recover missing cloud image identities

Exact deployed image digests were not included in the uploaded run artifacts.
`selection.json` leaves those fields null. The pinned base image, model revision,
runtime versions and effective launch commands are known; the complete built
image identity is not. Run this read-only command in authenticated Cloud Shell
if the executions are still available:

```bash
for execution in context-inference-lab-q2vr4 context-inference-lab-bpjvd; do
  gcloud run jobs executions describe "$execution" \
    --project=project-ce0aa110-c8ea-4c1e-80b --region=us-east4 \
    --format='yaml(metadata.name,metadata.creationTimestamp,spec.taskCount,spec.parallelism,spec.template.spec.containers.image,spec.template.spec.containers.resources,spec.template.spec.timeoutSeconds,status)' \
    > "$execution-cloud-metadata.yaml"
done
```

These projections omit environment-variable values. If a reference is only a
mutable tag, retain the execution's resolved digest or the original Cloud Build
digest before claiming exact image reproducibility. Actual Cloud Billing charges
are also absent; this study makes no dollar-cost or cost-efficiency claim.
