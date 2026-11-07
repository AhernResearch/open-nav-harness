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
read-only. The LightNav bridge uses its own WebSocket session on the existing policy
server; it does not restart that server.

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

Model HTTP generation may retry transport errors/429/502/503/504 up to three times.
No tool dispatch is retried. Unknown tools and invalid arguments reject the entire
unexecuted batch and return feedback for a new model decision. Duplicate call IDs,
truncated responses and model termination without explicit STOP remain recorded failures.
Protocol probes should be run before collecting expensive navigation traces.

## Backup and restore

1. Pause all jobs and allow active episodes to finish; confirm no running attempts.
2. Stop the API and scheduler, preserving remote service ownership.
3. Run `uv run nav-harness --state STATE backup --output BACKUP/platform.sqlite3`.
4. Copy `runs/`, `admin-password`, and private deployment settings to the same backup.
   The database backup alone is not a replay backup. Keep password files mode 0600.
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
experiment metadata. Rotate the admin password by replacing `admin-password` and
restarting the API; existing cookies then become invalid.

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
