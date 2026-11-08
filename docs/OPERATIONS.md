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
`/data/nav_storage/nav-harness/` tree. Existing model weights and scene assets are shared
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

### Agentic execution/reset ablation, November 9

`execute_subtask` now accepts `task_id`, optional `instruction`, and optional
`reset_context`. Model-selected action, distance and cumulative-turn caps have been
removed, including the hidden 80-action/3-m/180-degree agentic segment defaults.
Policy STOP, three consecutive blocked moves, the episode action budget and episode
timeout still end execution. An operator may set `harness.vla_steps_per_call`; the
paired experiment explicitly sets it to zero (disabled). The older strict subtask
condition retains its original configured segment limits.

`harness.allow_context_reset=false` removes the reset argument from both the actual
validated tool schema and its briefing. It does not force a reset every call or remove
visual history. Automatic anchoring on task changes and bounded frame selection are
identical in both variants. The `begin` event records `reset_requested` and
`reset_reason` (initial/task_change/explicit/continue).

The new release includes the common November 8 reliability/context fixes. Both reset
variants use that same release; comparison against the older eight-route run is
descriptive, not an isolated estimate of removing caps. LightNav exposes no seeded
reset, so a single paired run is preliminary. Historical reset use is confounded by
task difficulty. See [historical calls](evidence/reset-context-history-20261109.json)
and [exact tool schemas](evidence/execute-subtask-schema-ablation-20261109.json).

The paired four-route run is complete: with manual reset exposed SR=0/4, without
it SR=1/4. Both groups had two errored attempts. Five explicit resets occurred only
on route 4132 (three same-task, two task switches); route 7006 improved without reset
but had used no explicit reset in the other group either. The run does not establish
that manual reset helps or hurts. Removing caps allowed a single 436-action/268-second
rollout that travelled 41.86 m with only 0.58 m net displacement. Three attempts across
the groups exhausted the global action budget and then repeated terminal calls until
900 seconds: the existing runtime recognizes only `stop_called`, while the tool's
already-over path retains `step_budget_exhausted`. One other attempt exhausted its
2048-token output allowance. These faults are recorded, not fixed in this experiment.
Do not interpret total wall time as navigation latency alone or automatically resume
the historical hard-50 job. See [per-attempt metrics and tool timing](evidence/reset-context-ablation-results-20261109.json)
and [deployment, matched configuration and findings](evidence/reset-context-ablation-20261109.json).



### LightNav official input migration, November 9

The subsequent migration supersedes the bounded-history LightNav ablation above.
The owned actor snapshots upstream `3015508b70fb6e30bbd66701e94c9b916dcb8df8` into
its immutable release and uses continuous native episode history. It advertises
`history_mode=episode`, `context_replace=false`, and
`input_contract=lightnav_official_episode_v1`; preflight rejects an older replacement
server. LightNav tools no longer expose manual reset regardless of the older
`allow_context_reset` flag. NavGPT VLA's bounded profile remains available.

Only the chosen task instruction is sent, without a topology caption or summary.
Task-manager `begin`/anchor events still describe logical plan segments; they do not
reset LightNav. `policy_context.frame_ids` describes the available episode buffer,
not the smaller set selected by the native SlowFast sampler. The actor owns that
selection. Frame IDs retain causal order, including retraces; an endpoint inferred
again with revised text is not appended twice.

The fixed-frame GPU probe checked eight points through a 500-frame episode against
the official Habitat policy: native processor samples, raw tokens, waypoints and
velocity commands matched. The same-endpoint instruction change, 101 numerical
controller cases and malformed-action zero-STOP fallback also matched. A separate
live test sent eight simultaneous sessions with different history lengths; the
largest actual batch was seven and all outputs matched the serial official reference.
The first concurrent probe used an incorrectly retained reference-engine cache; its
failed evidence is preserved separately, and the corrected probe resets the official
single-session reference at episode changes. This was a validation-script error.

See [migration provenance](evidence/lightnav-official-migration-20261109.json),
[fixed-frame parity](evidence/lightnav-official-parity-20261109.json), and
[live batching](evidence/lightnav-official-live-batch-20261109.json). These are
implementation checks, not benchmark results or proof of a success-rate improvement.
The historical hard-50 job remains paused; new source requires a new experiment.
The subsequent two-route integration smoke succeeded on both routes (train indices
0 and 7648, NE 0.58 m and 2.57 m). The latter used one retrace and five VLA
delegations. Both traces preserved continuous episode history and plain actor
instructions; neither had a runtime error. See [smoke results and replay links](evidence/lightnav-official-smoke-20261109.json).
This small smoke is not a success-rate estimate.


## Backup and restore

1. Pause all jobs and allow active episodes to finish; confirm no running attempts.
2. Stop the API and scheduler, preserving remote service ownership.
3. Run `uv run nav-harness --state STATE backup --output BACKUP/platform.sqlite3`.
4. Copy `runs/` and private deployment settings to the same backup.
   The database backup alone is not a replay backup. Keep private settings files mode 0600.
5. Restore into a new state directory, run `nav-harness --state RESTORED init` to apply
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

## Development acceptance, 2026-11-07

[Recorded results](evidence/lightweight-harness-20261107.json) contain the two owned-worker
pilots, all eight runs of the two-route train-split matrix, per-route tool/inference/token
counts, replay checks and final direct/MCP GPU probes. The formal 50-route benchmark was
not started. Matrix source `9fdb040` precedes the final schema/trace optimization; final
worker source `6a32b07` was checked with explicit one-action integration probes.

Both 27B pairings and the 35B/LightNav pairing succeeded on 2/2 routes. The 35B/native
pairing succeeded on 0/2: its timed-out route never delegated to the VLA and used 46
short-motion calls. These are development observations, not a model ranking. All traces
remain evaluation-only. LightNav's owned and legacy runtime profiles produced identical
outputs on two of three replay frames, but differed on the third; do not claim bitwise
equivalence or attribute all end-to-end timing differences to transport.

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
`qwen27b-nvfp4-pool`, served model `navh-qwen27b-nvfp4`, and API
`http://127.0.0.1:18103/v1` through the owned LAN SSH tunnel. It is registered in
the async workbench on port 18889. Use its service ID for new experiments; frozen
experiments retain their original planner references. The former GPU-3 BF16
planner was stopped through its original platform's ownership-checked service
operation after single-GPU validation passed. Historical attempts were preserved.

The original NVIDIA checkpoint was downloaded directly on A100 from
`nv-community/Qwen3.6-27B-NVFP4` on ModelScope and verified against per-file SHA256
values. It is stored at `/data/nav_storage/models/NVIDIA-Qwen3.6-27B-NVFP4`.
The serving environment is independently installed at
`/data/nav_storage/nav-harness/venvs/planner-vllm025`; no shared environment was
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
`/data/nav_storage/nav-harness/deployment/nvfp4-20261108` on A100.

## Terminal lifecycle and cropped-history baseline, 2026-11-09

The Planner loop now exits when a movement tool reports that the environment has
ended, including step-budget exhaustion. Exhaustion remains a scored navigation
failure without an implicit STOP or an infrastructure error. A VLA STOP that only
hands control back to the Planner does not end the episode. Return-tool terminal
flags are normalized into the same lifecycle state. The isolated terminal probe
now executes one tool instead of continuing to issue STOP calls; regression tests
also cover remaining calls in a generated tool batch. Validation: 150 tests plus
4 subtests, required static checks and frontend build passed.

Only the `completed_subtask` history policy was retained for the requested
development rerun. The full-history job was cancelled and excluded. Source digest:
`297ccc1234ef1739a2c4e5b57497212a2c83e9c6e8287cc737ff9afa3d488c78`.

| Train episode | Success | NE (m) | Wall time (s) | execute_subtask calls | VLA predictions | History crops |
| --- | --- | --- | --- | --- | --- | --- |
| 456 | 1 | 1.7874 | 218.8 | 7 | 91 | 3 |
| 7006 | 1 | 0.2670 | 164.3 | 4 | 66 | 3 |

Each crop retained one boundary frame; repeated calls within the same unfinished
subtask retained their history. Both episodes ended through an explicit Planner
STOP, so step-budget termination was verified by the isolated probe and tests,
not triggered by these rollouts. Neither episode returned to a node. Episode 456
still made two final zero-motion calls: benchmark success does not demonstrate
precise compliance with the requested stair position.

These are two single-rollout development checks, not a success-rate estimate.
Compared with the previous crop run, both pose restoration and lifecycle handling
changed, and concurrent load decreased during cancellation. Neither outcome nor
latency changes can be attributed to one fix. Detailed traces, metrics, crop
boundaries and limitations are in the
[baseline evidence](evidence/terminal-lifecycle-crop-baseline-20261109.json).
