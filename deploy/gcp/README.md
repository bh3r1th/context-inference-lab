# Cloud Run GPU pilot

This deployment targets the existing job, without modifying the application's
source, frozen experiment configs, legacy Dockerfile, or GPU requirements:

| Setting | Value |
| --- | --- |
| Project | `project-ce0aa110-c8ea-4c1e-80b` |
| Job / region | `context-inference-lab` / `us-east4` |
| Resources per task | 1 NVIDIA L4, 8 vCPUs, 32 GiB host RAM |
| Parallelism | 1 |
| Cloud Run task timeout | 3,600 seconds |
| Results bucket | `gs://context-inference-lab` |
| Runtime service account | `809334676818-compute@developer.gserviceaccount.com` |
| Image repository | `us-east4-docker.pkg.dev/project-ce0aa110-c8ea-4c1e-80b/context-inference-lab/benchmark` |

One task's 32 GiB request fits the previously reported 40 GiB memory quota when
enough quota remains available. GPU and CPU quota, billing eligibility, model
memory fit, and actual runtime still require a successful GPU execution. Saving
a job configuration alone does not verify these.

## What is prepared

The image uses the official vLLM **0.29.0-cu129** Linux/amd64 image, pinned to an
inspected manifest digest. Its installed package version is **0.29.0+cu129**, the
official CUDA 12.9 build of release **0.29.0**, matching the client's
`structured_outputs.json` API. The launcher accepts this specific build for the
configured release and records the full installed version in task artifacts.
Other build suffixes do not satisfy the configured release. It does not install
the legacy requirements that pin vLLM 0.8.5. The base uses **CUDA 12.9.1**, with native host driver libraries by
default (`VLLM_ENABLE_CUDA_COMPATIBILITY=0`). The previous CUDA 13 image with forced
compatibility libraries failed GPU initialization with error 803 on this job.
Actual driver compatibility still requires the GPU smoke execution. The build
uploads an explicit context excluding environment files,
credentials, Git metadata, virtual environments, and old results.

The pinned upstream image deliberately installs NCCL 2.30.7 for DeepEPv2 while
PyTorch 2.13.0+cu129's metadata requests 2.29.7 for `nvidia-nccl-cu12`. The override is
[documented in vLLM's Dockerfile](https://github.com/vllm-project/vllm/blob/v0.29.0/docker/Dockerfile#L854).
`check_dependencies.py` runs `pip check` and permits only that exact conflict,
after verifying the exact installed vLLM/Torch/NCCL versions and the upstream override
file. Additional conflicts, different versions, or missing override evidence
fail the build, with expected and installed versions included for mismatches.
The check prints the acknowledged exception and verified package versions
explicitly; it does not claim ordinary `pip check` passed. The compatible Cloud Storage SDK
already present in the base is retained instead of unnecessarily downgraded.

The launcher selects one complete experiment cell using `CLOUD_RUN_TASK_INDEX`.
It checks bucket write access, runs a GPU preflight, starts one local vLLM server, waits for the exact
model, executes the existing benchmark runner, checks current-run artifacts, and
uploads partial/final outputs. A new server starts for every task. Cold-cache
cells therefore start with an empty cache; warm cells retain their configured
warmups. This changes server lifetime across cells compared with a continuous
multi-cell run and is recorded by the per-task config and launch logs.

The 72K model context limit, model revision, event selections, repetitions,
output cap, and request seeds are preserved. An L4 memory failure is reported;
the launcher does not silently shrink the experiment or quantize the model.
Model weights download to `/tmp/huggingface`, whose storage consumes host RAM on
Cloud Run. The 32 GiB allocation needs validation during model loading.

Before model loading, a separate process records `nvidia-smi` GPU/driver/memory
details, the Torch CUDA build, and compatibility settings in `gpu-preflight.log`.
It initializes CUDA and checks a small tensor operation. A failure or its
120-second deadline stops the task before model startup. The probe uses the same
vLLM environment initialization as the server. Enabling forward-compatibility
libraries is an explicit runtime choice, not an automatic retry or driver
upgrade; first inspect the actual driver and probe output. Cloud Run's host
driver cannot be upgraded from this container.

`deploy.sh` explicitly selects native libraries by default, including when
updating a job with old environment overrides. A supported forward-compatibility
test can be selected by setting `VLLM_ENABLE_CUDA_COMPATIBILITY=1` when invoking
`deploy.sh`, after reviewing the driver diagnostics. That mode is probed too;
the launcher never suppresses a CUDA initialization failure.

The smoke plan contains **one request**. The pilot plan contains **18 tasks and
180 measured requests**, excluding warmups. Increasing task count without matching
the plan is rejected. Full-run planning remains gated on GPU pilot measurements;
it is not implemented or claimed ready by this initial deployment.

## Before running

Use an authenticated Cloud Shell checkout of the branch or commit containing
`deploy/gcp/`. Check the Git commit before building so the image can be traced
back to the reviewed source. A deployment bundle can also be extracted at the
repository root when testing changes before merge.

Required APIs: Cloud Run, Artifact Registry, Cloud Build, and Cloud Storage.
The Cloud Shell identity needs build submission, job update/execution, image
read, and permission to use the runtime service account. The selected Cloud Build
service account needs access to its staging source bucket, Artifact Registry
Writer on the image repository, and write access to `gs://context-inference-lab`
for build logs. Cloud Build uses GCS-only logging, so this build does not require
the Logs Writer role. The runtime
service account needs bucket write access; the reported Storage Admin grant
should cover it, but the launcher verifies access before loading the model.

No API key is needed for this public model and task-local server. Network access
must permit Hugging Face model downloads and its download infrastructure as well
as GCS. Existing project network restrictions are not changed by these scripts.

## Build and execute the smoke task

Run from the repository root in Cloud Shell:

```bash
bash deploy/gcp/build.sh
```

This submits a remote Cloud Build and prints an immutable image reference after
the image has been pushed successfully. Build logs are stored in
`gs://context-inference-lab` and streamed by `gcloud builds submit`. The derived
image uses `python3`, the interpreter configured by the upstream runtime image.
Use that exact image reference:

```bash
IMAGE='us-east4-docker.pkg.dev/project-ce0aa110-c8ea-4c1e-80b/context-inference-lab/benchmark@sha256:REPLACE_WITH_BUILD_DIGEST'
bash deploy/gcp/deploy.sh "$IMAGE" smoke
gcloud run jobs execute context-inference-lab \
  --project=project-ce0aa110-c8ea-4c1e-80b --region=us-east4
```

`deploy.sh` updates the existing job and does not execute it. It clears old
command/argument overrides, selects the image entrypoint, sets one task, and
disables automatic retries for this diagnostic run. The execute command submits
remote work; closing the laptop does not cancel an accepted execution.

Inspect execution status in Cloud Run and its task logs. Artifact prefixes are:

```text
gs://context-inference-lab/cloud-run/EXECUTION/task-INDEX-CELL/attempt-ATTEMPT-UUID/
```

Each attempt preserves its own config, GPU preflight, vLLM and benchmark logs, task status,
dataset snapshots, raw request/telemetry files, and available summaries. Only a
verified successful run receives `complete.json`. Smoke success requires the
checked-in event's expected answer from the actual model, not merely a listening port.

The launcher has an internal 3,300-second deadline including initialization,
a 900-second model-readiness limit, and periodic uploads. Final uploads on
shutdown are best effort; an abrupt platform kill can lose data since the last
successful upload. Retries start fresh and retain failed attempts separately;
this initial pilot does not implement automatic continuation or skip completed
cells across executions. Do not run overlapping executions of the job.

## Pilot after smoke success

Only after smoke succeeds and startup leaves sufficient time for useful work:

```bash
bash deploy/gcp/deploy.sh "$IMAGE" pilot
gcloud run jobs execute context-inference-lab \
  --project=project-ce0aa110-c8ea-4c1e-80b --region=us-east4
```

This launches the 18 pilot cells sequentially, one GPU at a time. Each gets its
own 60-minute limit; the overall execution can last longer than one hour. The
pilot deliberately includes 32K scaling cells so failures are visible. Do not
interpret an incomplete task or failed request as successful benchmarking.

Use the pilot's startup times, completed-cell durations, memory diagnostics,
quality measurements, and failures to decide whether the full frozen experiment
fits this GPU and how many full-run tasks are needed. A full cell that cannot
finish within the remaining task budget needs a different execution platform;
splitting its requests would change the experiment's admission/cache behavior.

## Local verification

With the repository development environment activated:

```bash
python -m pytest -q tests deploy/gcp
bash -n deploy/gcp/build.sh deploy/gcp/deploy.sh
```

These tests exercise client and deployment orchestration with fixtures. They do
not establish GPU readiness, cloud IAM/quota eligibility, successful image build,
or real model performance. Those require the remote smoke/pilot executions above.
