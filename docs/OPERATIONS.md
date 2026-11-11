# Operations

## Configuration and deployment

Copy `deployment/platform.example.json` to a private state/config folder and replace
hostnames, interpreters and model paths. Services are immutable registrations. To change
a model/profile, create a distinct registration and experiment; do not change weights
behind an endpoint while experiments are queued. Model/template changes require a new
experiment even when the model ID is reused.

Holo runs the API and worker. Register A100 loopback endpoints through an owned SSH
forward. `deployment/direct_lan_socket.py` binds SSH to the requested LAN interface and
has no VPN fallback. Do not configure an A100 reverse proxy to the workstation VPN.
The current local deployment configuration is under `state/deployment/`, excluded
from version control. Remote code and extracted train annotations live under the new
`/data/nav_storage/open-nav-harness/` tree. Existing model weights and scene assets are shared
read-only. The owned LightNav worker loads its inference engine in-process. The legacy
bridge remains available with its own WebSocket session on the existing policy server;
neither deployment takes ownership of that neighboring server.

Use the systemd templates in `deployment/systemd/` after substituting absolute paths.
The worker unit uses `KillMode=process`: restarting the scheduler leaves owned episode
process groups alive for receipt-based recovery. Use job cancellation to stop an episode.
Do not run multiple workers against one state folder. API and worker may restart
independently. Pause jobs before deploying source changes; pending frozen configurations
will otherwise fail the implementation fingerprint check.

The service launcher reserves memory per assigned GPU and defaults to a 1 GiB margin.
This is admission based on measured free memory, not a hardware isolation guarantee.
Other users can allocate memory after the check; runtime failures remain visible.
Environment renderer `--gpu` is the visible CUDA index, whereas the inherited NavGPT
VLA launcher sets `CUDA_VISIBLE_DEVICES` itself from its `--gpu` argument. Configure
templates accordingly. CPU-only bridges use `gpu_ids=[]`, `reserve_mib=0`.

## Failure handling

- **Pause:** finish the current episode, then stop claiming new episodes.
- **Cancel:** terminate the owned episode group; never replay a movement automatically.
- **Blocked attempt:** read its error/logs. Stop its owned environment and VLA through
  the service page; then select **校验并释放资源锁** in replay. Restart the services and
  resume the job. Reconciliation verifies the attempt process is gone and both motion
  services have recorded stopped states. A committed episode result (including a
  timeout failure) is kept as completed and will not rerun; an attempt without a
  committed result remains interrupted. Old evidence is retained.
- **Unknown service:** inspect its owned systemd logs and retry stop. Only after confirmed
  stop should it start again. Do not adopt a process that merely happens to answer on
  the configured port. A startup deadline never silently kills unrelated services.
- **External resource uncertainty:** operator investigation is required. This version
  deliberately has no force-unlock button for unowned movement services.

Model generation and serving-tokenizer HTTP requests share a bounded retry policy:
at most three attempts for transport errors/429/502/503/504, with 1 s and 2 s backoff.
Token counting caps each HTTP timeout at 15 s (or the shorter profile timeout), records
attempts and latency, and still fails closed if the counter remains unavailable.
Context-limit errors trigger normal context pruning, not HTTP retries; cancellation
propagates immediately. Counts are never replaced with an estimated value.
No tool dispatch is retried. Unknown tools and invalid arguments reject the entire
unexecuted batch and return feedback for a new model decision. Duplicate call IDs,
truncated responses and model termination without explicit STOP remain recorded failures.
Protocol probes should be run before collecting expensive navigation traces.

The agentic context places its single current-state projection after retained history.
It labels plan statuses and memory captions as model assessments and reports the current
map-read precondition without selecting a tool or imposing a review workflow. Motion
feedback uses `subtask_return` to describe policy handoff; the compact agentic response
omits the redundant `vla_suggests_arrival` flag. Legacy raw evidence and other conditions
are unchanged. Turn budgets are cumulative upper limits, not requested headings.

The November 8 optimization was first validated locally. See
[fault tests and historical decision replays](evidence/planner-reliability-optimization-20261108.json).
Single-turn replay did not correct the two tested false STOP decisions and does not
establish a navigation success-rate improvement. Keep the original hard-50 job paused;
deploy a matching planner/worker release and create a new development experiment for
closed-loop evaluation rather than resuming an experiment frozen against old source.

Current execution limits, reset schemas and STOP rules are documented in the
[tool contract](TOOL_CONTRACT.md). LightNav's native episode history and the opt-in
completion-boundary policy are documented in [LightNav](LIGHTNAV.md). Historical
ablation and migration results remain under [evidence](evidence/).

## Backup and restore

1. Pause all jobs and allow active episodes to finish; confirm no running attempts.
2. Stop the API and scheduler, preserving remote service ownership.
3. Run `uv run open-nav-harness --state STATE backup --output BACKUP/platform.sqlite3`.
4. Copy `runs/` and private deployment settings to the same backup.
   The database backup alone is not a replay backup. Keep private settings files mode 0600.
5. Restore into a new state directory, run `open-nav-harness --state RESTORED init` to apply
   migrations, check `/health`, then start the API and worker with that same state path.

The backup uses SQLite's online backup API; never copy a live database while omitting
its WAL. Full filesystem restoration should use the original absolute state path because
attempt receipts include their original paths. Moving a backup requires updating stored
paths while offline and must not be used to resume an active attempt.

## Historical results and exports

Configure `import_roots` explicitly. Import references an existing directory with a
`summary.json`; source files remain read-only. Image reads are restricted to allowed
image extensions and resolved containment within the registered root. Historical
records stay evaluation-only and are not relabelled as exact provider inputs.

CSV contains per-attempt navigation, walking, tool, token and timing fields. Unknown
model costs remain empty. ZIP exports retain requests, content-addressed images,
subtask traces, evaluator artifacts and review status. Evaluator reference paths are
privileged supervision and must be excluded from student input construction.

Service logs and attempt logs are for local administrators. Keep deployment credentials
in environment variables instead of command-line arguments; never paste secrets into
experiment metadata. The console and API allow full access without authentication;
there is no administrator password or login session.

## Owned actor worker deployment

Use a trusted service template invoking `python -m navgpt.vla.worker --config FILE`.
Register it as a managed `vla` service with metadata `{"tool_execution": true}`; experiment
freezing then sets `harness.execution_url` to that leased endpoint automatically.
Both the planner host and the worker must run the same source release. The worker JSON
contains `backend`, `model_path`, `model_options`, `environment_url`, `artifact_dir`, and
`port`; it always binds to `127.0.0.1`. The configured environment URL must match the
experiment's environment URL, with the same port forwarded to Holo.

LightNav `model_options` accepts `gpu_memory_utilization` (default 0.35); use its existing
pinned vLLM 0.19.1 inference environment. Native VLA accepts its existing model options
such as `max_nav_vis_tokens`, `frame_cache` and `max_sample_size`. The worker additionally
needs Pydantic v2, requests and Pillow. Install these in an owned dependency overlay or
owned environment; do not alter a neighboring project's environment in place.

The two workers can coexist when explicit GPU reservations fit. Preserve the original
LightNav and native VLA services: do not take over their ports or stop their processes.
An incomplete RPC session is a fence even if its TCP connection disappears. Stop the
owned actor/environment, reconcile the blocked attempt, restart, then use a new attempt.
Remote `artifact_dir/<session_id>/calls.jsonl` preserves receipts if a response was lost;
this directory belongs in evidence backups alongside local `state/runs`.

## A100 planner pool, 2026-11-08

The user authorized GPUs **0, 2, 3, 4** for this deployment. GPUs **1 and 5**
belong to longstanding unrelated services and must remain untouched. Their parent
PIDs (265160 and 2429405), original start dates and healthy endpoints were checked
after deployment. The shared original inference environment remains vLLM 0.19.1 /
PyTorch 2.10.0.

| GPUs | Service | Capacity |
| --- | --- | --- |
| 0, 3, 4 | Three Qwen3.6-27B NVFP4 replicas, native vLLM data parallel serving | 4 requests per replica, 12 total |
| 2 | One shared LightNav instance; the current execution pool has eight environments | 8 concurrent actor requests |
| 1, 5 | Protected existing services | Unchanged |

The planner service is `eaa85c5c908141f6b57e16a231cdc511`, template
`qwen27b-nvfp4-pool`, served model `onh-qwen27b-nvfp4`, and API
`http://127.0.0.1:18103/v1` through the owned LAN SSH tunnel. It is registered in
the async workbench on port 18889. Use its service ID for new experiments; frozen
experiments retain their original planner references. The former GPU-3 BF16
planner was stopped through its original platform's ownership-checked service
operation after single-GPU validation passed. Historical attempts were preserved.

The original NVIDIA checkpoint was downloaded directly on A100 from
`nv-community/Qwen3.6-27B-NVFP4` on ModelScope and verified against per-file SHA256
values. It is stored at `/data/nav_storage/models/NVIDIA-Qwen3.6-27B-NVFP4`.
The serving environment is independently installed at
`/data/nav_storage/open-nav-harness/venvs/planner-vllm025`; no shared environment was
upgraded. vLLM 0.25.0 supports this checkpoint's `modelopt_mixed` loader on SM80;
the old 0.19.1 loader rejects that combination. A100 uses Marlin weight-only
FP4/FP8 kernels, not native FP4 tensor-core computation. The checkpoint is unmodified.

The configuration uses `--data-parallel-size 3`, `--max-num-seqs 4`,
`--max-model-len 16384`, `--max-num-batched-tokens 8192`,
`--gpu-memory-utilization 0.65`, `--kv-cache-dtype bfloat16`, prefix caching,
CUDA graphs, `qwen3_coder` tool parsing and `qwen3` reasoning parsing. Planner
requests disable thinking and retain the production output limit of 2048 tokens.
Native vLLM manages the shared endpoint and request distribution; there is no
additional custom load-balancing service. Some historical graphics-only simulator
processes also remain on GPUs 3/4; the allocation above describes the current pool,
not exclusive ownership of every process on those devices.

Eight fixed multimodal requests were replayed without executing navigation tools:

| Deployment | Batch wall time | Median request latency | Valid tool responses at 384-token test cap |
| --- | ---: | ---: | ---: |
| Original BF16, one GPU | 50.69 s | 34.03 s | 7/8 |
| NVFP4 / vLLM 0.25, one GPU | 32.39 s | 29.07 s | 8/8 |
| NVFP4 / vLLM 0.25, three GPUs | 18.53 s | 12.96 s | 7/8 |

The same `ep4132` request reached the test-only token cap in the original BF16 and
final three-replica batches. A follow-up on the deployed pool using the production
2048-token cap returned a valid tool call after 389 output tokens. Replica request
counters increased by 3, 3 and 2, confirming all three engines served the batch.
These are single-batch deployment checks: software, quantization, caching and
replica count changed together, outputs differ, and no navigation success rate
was measured. No formal benchmark was started as a deployment side effect.

Evidence: [deployment snapshot](evidence/nvfp4-planner-deployment-20261108.json),
[replay results](evidence/nvfp4-planner-replay-20261108.json),
[checkpoint hashes](evidence/nvfp4-planner-checkpoint-20261108.json), and
[installed dependencies](evidence/nvfp4-planner-environment-20261108.txt).
Administrative scripts, dependency wheels and installation reports are retained
under `state/deployment/nvfp4` locally and
`/data/nav_storage/open-nav-harness/deployment/nvfp4-20261108` on A100.

## Independent planner replicas, 2026-11-10

The data-parallel deployment above has no session affinity: identical requests were
distributed across engines, so one episode's growing prompt rarely reused a prefix
cache. It is replaced by three independent single-GPU vLLM servers running one owned
unit, `deployment/vllm_replicas.py`. The platform sets `CUDA_VISIBLE_DEVICES` from the
service's GPU list; replica *i* gets the *i*-th GPU and the *i*-th `--ports` entry. If
any replica exits, the launcher stops the others and the unit fails, so a partial
pool is never reported ready.

The planner service lists its endpoints in `metadata.replica_urls`, with the first as
its URL. Health requires every replica to answer. Planner slot *i* maps to replica
*i* mod *n*; `claim()` picks a free slot on the least occupied replica and rewrites
the attempt's `config.planner.profile.base_url`, so an episode stays on one replica
for its whole lifetime. The chosen URL is frozen in the attempt's `input.json` and
the `attempt_started` event; the experiment itself is unchanged. These changes are
outside the implementation digest, so actor and environment services keep running
across this release.

## Episode termination

The Planner loop exits when movement reports that the environment ended,
including step-budget exhaustion. Exhaustion remains a scored navigation failure
without an implicit STOP or an infrastructure error. Actor STOP only hands control
back to the Planner. Remaining calls in a generated batch are not dispatched after
termination. Historical development checks remain in the
[baseline evidence](evidence/terminal-lifecycle-crop-baseline-20261109.json).
