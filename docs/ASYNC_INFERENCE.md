# Shared asynchronous VLA execution

Each episode keeps one agent loop, one `NavigationSession`, and an exclusively
leased execution worker/Habitat pair. Multiple workers use one model service.
The planner's HTTP adapter was already asynchronous; tool RPC now also uses
`httpx.AsyncClient`. Blocking simulator calls remain inside the isolated execution
worker. Neither navigation tools nor planner prompts change for concurrency.

## Model service

Start `python -m navgpt.vla.inference --config inference.json` in the backend's
CUDA environment. Install this repository's HTTP runtime dependencies there (or
use an isolated dependency overlay); the native PyTorch model still requires
`requirements-vla.txt`. Do not replace native action-head inference with chat
completion requests.

Example LightNav configuration:

```json
{
  "backend": "lightnav",
  "model_path": "/models/LightNav-0",
  "port": 18381,
  "max_sessions": 8,
  "max_pending": 8,
  "max_batch_size": 4,
  "max_wait_ms": 8,
  "model_options": {"max_num_seqs": 4},
  "journal": "/owned-state/lightnav-inference.jsonl"
}
```

LightNav reuses its library's preprocessing, per-session tracking buffer, ViT
cache, microbatch scheduler, and trajectory decoder. The library is a deployment
dependency, not modified by this repository. Its `InferenceConfig.max_num_seqs`
must match the desired decode concurrency: the high-level tracking constructor's
default is only one. A session reset clears only its private cache. Cancelled
queued sessions are filtered before GPU submission; cancellation during a batch
preserves result-to-session alignment and discards the cancelled result.

For `backend: "navgpt"`, use the official checkpoint and native model options.
One model/processor is shared, with a separate agent, frame history, preprocessing
cache and RNG per session. One dedicated executor thread serializes native
processing and GPU forwards. Other episodes can run planners and simulation
while awaiting this queue. Native GPU batching is deliberately not implemented.

Service capacities have distinct meanings:

- `max_sessions`: maximum resident actor contexts (default 8).
- `max_pending`: admitted actor operations, queued plus running (default 8).
- Registered VLA `max_concurrency`: episode leases admitted by the platform.
- Platform `max_workers`: simultaneously running episode processes (default 4).

Queues reject overflow before mutating actor state. There is no automatic
movement retry. Excessive concurrency can increase latency even when throughput
improves; tune capacities from the recorded waiting and execution times.

## Execution pool and API

The existing `navgpt.vla.worker` entry point also runs a model-free execution
worker. Supply `inference_url` instead of `model_path`:

```json
{
  "backend": "lightnav",
  "port": 19680,
  "environment_url": "http://127.0.0.1:19580",
  "inference_url": "http://127.0.0.1:18381",
  "artifact_dir": "/owned-state/execution-0"
}
```

Register the model as `kind: "vla"`, with
`metadata: {"shared_inference": true}` and `max_concurrency: 4`. Register each
worker as `kind: "execution"`; its metadata must contain `environment_url` and
`inference_url`, matching registered endpoints. No GPU reservation is needed for
execution workers; reserve memory separately for models and Habitat processes.

An experiment selects a shared `vla_service`, a planner, and an `execution_pool`:

```json
{
  "name": "Shared LightNav development",
  "planner_service": "PLANNER_ID",
  "vla_service": "INFERENCE_ID",
  "execution_pool": [
    {"environment_service": "ENV_0", "execution_service": "EXEC_0"},
    {"environment_service": "ENV_1", "execution_service": "EXEC_1"}
  ],
  "episodes": [0, 1879],
  "purpose": "development",
  "config": {
    "dataset": {"name": "r2r", "split": "train"},
    "harness": {"mode": "agentic"},
    "vla": {"backend": "lightnav"}
  }
}
```

The pool and legacy single `environment_service` are mutually exclusive.
Existing combined workers and frozen experiments remain supported. Shared actor
services require an execution pool and a planner tool runtime; legacy VLA-only
runs keep their existing path.

The model protocol is version 2. `POST /sessions` takes `session_id` and `backend`;
`POST /sessions/{id}/call` takes `request_id`, `operation`, and `data`. Operations
are `reset`, `replace_context`, `observe`, `act`, `state`, and `info`, with the
existing backend payloads/results. GET session state and POST cancel/close are
separate lifecycle endpoints. Duplicate requests and overlapping same-session
operations are rejected. There is no global reset endpoint.

The execution-tool RPC stays version 1. The client supplies and records a unique
session identity before opening it, so a lost open response is recoverable. The
worker creates and resets its actor session before accepting tools. Preflight
checks both execution and inference source fingerprints and endpoint bindings.

## Scheduling and recovery

Migration 0002 adds the selected binding to each attempt and a unique index on
active `(job_id, episode)`. A single transaction selects an unclaimed episode,
an independent worker/environment pair, and shared planner/VLA slots. No manual
job sharding is required. Frozen experiments are never rewritten by selection.

Pause drains every active attempt. Cancellation stops each attempt and signals
its remote worker. A blocking GPU operation is retained until it actually ends;
a cancelled prediction cannot become a new movement. An already dispatched
simulator operation may finish, so an uncertain motion lease is never released
merely because HTTP was cancelled.

Uncertain attempts retain their own leases. Healthy in-flight siblings can
finish; further claims from an interrupted job require reconciliation. Stop the
affected owned execution worker and environment, then use the existing reconcile
action. It cancels/closes only that attempt's shared inference session, using its
`execution_session.json` receipt, and refuses release while inference is still
in flight. It does not require stopping the shared model serving other routes.
A CUDA/engine failure stops admission across that model service.

## Observability and verification

The workbench supports execution-pool selection and displays active routes,
completed routes, actor sessions and pending/waiting operations. Read
`GET /api/inference/status` for shared service health and counters.

Each actor prediction carries `inference_metrics` into its existing trace,
including session/request IDs and queue/execution time. LightNav adds batch,
vision and decode timings; native adds preprocessing and forward timings. These
are execution telemetry, not additional planner tools or evaluation hints.

`scripts/validate_shared_inference.py` replays a local development-image manifest
at concurrency 1, 2 and 4. It compares tokens, actions, waypoints and history
states, and reports latency/throughput. It never moves an environment. Navigation
integration uses a separate owned deployment and development manifest. Fixed
image throughput is not an end-to-end navigation speedup or a success-rate claim.

Keep old measurements on their frozen releases while validating a new release.
Roll back by draining the new jobs and choosing the existing legacy deployment
for a new experiment; do not reuse or overwrite completed attempts.


## Eight-lane LightNav development deployment (2026-11-08)

The hard-50 run uses eight exclusive Habitat/execution pairs, one shared
LightNav checkpoint and the existing Qwen3.6-27B planner. Platform workers,
planner admission and VLA episode admission are each eight. The actual planner
vLLM server was restarted with `--max-num-seqs 8`; LightNav uses batch/decode
width eight, a 4 GiB KV cache, and sixteen session/pending slots. A maximum batch
size is a ceiling, not a guarantee that requests arrive together.

The frozen selection is `evidence/async-lightnav-hard50-selection-20261108.json`:
50 unique train trajectories across 34 available scenes, excluding the earlier
scheduled set, with 20 stairs, 20 long/multiturn and 10 qualified-endpoint routes.
Each scene contributes at most two routes. Reference geometry is used only for
selection/evaluation, never planner input. These are development measurements,
not an unbiased benchmark sample. Each route retains the 900-second/100-turn
budget. Earlier jobs/results remain preserved and paused; this is a new run.

The native VLA's earlier four-route integration used an incorrect LightNav
history-camera configuration. Those attempts are retained as infrastructure
failures, not a fair backend comparison. Preflight now validates the camera
contract; this hard-50 run uses only LightNav, as requested.
