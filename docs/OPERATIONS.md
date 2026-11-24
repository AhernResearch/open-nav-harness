# Operations

## Deploying this branch

This release removes the experiment/job/attempt pipeline. Before deploying it over an existing state
folder:

1. Drain the legacy queue. No job may have a `running` or `blocked` attempt, because nothing in this
   release can reconcile or resume them. Check with
   `sqlite3 STATE/platform.sqlite3 "SELECT status, count(*) FROM attempts GROUP BY status"`
   and finish, cancel or reconcile them with the previous release first.
2. Remove `max_workers` and `import_roots` from the platform settings file. `PlatformSettings` forbids
   unknown keys, so the API and worker refuse to start with them.
3. Update `conversations_config`: navigation entries
   that lack `url` (inherited environment bindings) are dropped on load; define sim-engine targets
   instead.

Legacy tables (`experiments`, `jobs`, `attempts`, `leases`, `imports`) stay in the database, unread.
Old `state/runs/<attempt>` folders are untouched evidence. Migrations are only added, never edited.

## Configuration

Copy `deployment/platform.example.json` to a private location (the default `state/` is git-ignored)
and replace hosts, interpreters and model paths. Settings:

- `host`, `port`, `ssh_config`: bind address and SSH options for service control.
- `service_templates`: administrator-defined launch commands (`host`, `argv`, `working_directory`,
  `environment`, `startup_timeout_s`). Only these ever become argv; browser input never does.
- `conversations_config`: path to a JSON file with `profiles` (PlannerProfile by alias),
  `default_model`, `mcp` (alias to `url` and optional `credential_env`) and `navigation` (alias to a
  sim-engine target: `url`, `dataset`, `camera`, `seed`, `catalog`, `tool_set`, `executor`, `budgets`,
  `map`). This one file feeds both conversations and the run composer (`/api/runs/compose`).

Credentials are environment-variable references (`credential_env`) set in the API and worker process
environments, never values in settings, specs or run records. Keep private files mode 0600. The
console and API have no login; `serve --host 0.0.0.0` gives LAN users full access, so put an HTTPS
reverse proxy and network controls in front when that matters.

`deployment/planner-responses.example.json` is an external Responses-API planner profile
(`POST /api/services`). `deployment/direct_lan_socket.py` is an SSH `ProxyCommand` that binds the
connection to a named LAN interface and does not fall back to another route.

## Processes

```bash
uv run open-nav-harness --state STATE --settings SETTINGS serve     # API
uv run open-nav-harness --state STATE --settings SETTINGS worker    # run scheduler
```

Management flags go before the subcommand and both processes must use the same `--state`. Only one
worker may own a state folder (`worker.lock`). Systemd templates are in `deployment/systemd/`
(substitute absolute paths). API and worker restart independently.

The worker runs episodes as asyncio tasks, so restarting it ends them. On the next start every episode
left `running` becomes `failed_or_uncertain` ("interrupted by a worker restart") and its simulator
session is released; it is never resumed. Drain or cancel runs before a planned restart.

Digest freeze: each run stores a digest of the source under `kernel/`, `core/` and `providers/`. A
worker runs only runs whose digest matches the code it loaded. After a code change, restart the worker
before submitting new runs; queued episodes of runs frozen against older files fail with "kernel
source changed since the run was frozen", and runs frozen against newer code wait for a restart.
Platform, conversation and frontend code are outside the digest.

## Services

`/workbench` registers external endpoints and starts or stops owned ones. Owned units are started over
SSH with `systemd-run --user`, carry `OPEN_NAV_HARNESS_OWNER=<service id>` and `CUDA_VISIBLE_DEVICES`
from the service's `gpu_ids`, and are stopped only when that ownership verifies. External services
cannot be started or stopped, and the platform never adopts a process that merely answers on a port.
Start requires free GPU memory (`reserve_mib` plus a 1 GiB margin, net of other starting services).
This is admission from measured memory, not isolation: other users can allocate afterwards.

Health probes are per kind: planner `/models` (all of `metadata.replica_urls` must answer when set),
environment `/health`, vla and execution `/info`. sim-engine and the LightNav actor serve
`/v1/health`, so the platform probe does not match them; check those endpoints directly.

`deployment/vllm_replicas.py` runs one independent vLLM server per visible GPU as one supervised unit
(one `--ports` entry per GPU; if any replica exits all stop). Serve a profile's `base_url` from one
endpoint; a run keeps its planner endpoint for the whole episode.

`python -m deployment.probe_planner --url URL/v1 --model ID --output DIR` checks a planner deployment
before runs: two-image input, a schema-valid tool call, and that counted input tokens equal the
generation usage. It is a protocol test, not a navigation score; output folders are create-only.
The Qwen-VL token estimate (`token_count_mode: "estimate"`, `providers/vision.py`) is for gateways
without a counting API. Remote counting failures never fall back to it silently.

## Failure handling

- Cancel (`POST /api/runs/{id}/cancel`): queued episodes become `cancelled`; running ones are
  cancelled and recorded as `cancelled`. Motion is never replayed.
- `failed_or_uncertain`: a lost motion reply or a worker restart. Read the episode's
  `failed_or_uncertain.json` and `session.json`; the simulator session is released on restart.
  Evidence is kept and the episode is not rerun automatically; submit a new run to repeat it.
- `SimulatorBusy` (429, the simulator is shared with another client): the episode is requeued
  before anything is written, then retried after a short backoff.
- Unknown service state: read the owned unit's logs and retry stop. Start again only after the stop is
  confirmed.
- Planner HTTP generation retries 3 times on transport errors and 429/502/503/504. Token counting fails
  closed when the counter is unavailable.

## Backup and restore

1. Cancel or drain runs and conversations; stop the API and worker.
2. `uv run open-nav-harness --state STATE backup --output BACKUP/platform.sqlite3` (SQLite online
   backup; never copy a live database without its WAL).
3. Copy `STATE/runs/`, `STATE/conversations/`, `STATE/catalog/` and the private settings files. The
   database alone does not carry episode evidence.
4. Restore into a new state directory, run `open-nav-harness --state RESTORED init` to apply
   migrations, check `/health`, then start API and worker on that path. Episode folders are referenced
   by run id, not absolute path, but never resume an interrupted episode from a restored copy.

## Data admission

`purpose=collection` requires a dataset whose name ends `_train`. Completed episodes record
`training_admission: pending_review`; successful execution never admits an episode to training.
`evaluation.json` (goals, reference paths, metrics) is simulator-side evidence and must be excluded
from model and student input construction. Run bundles download from
`GET /api/runs/{id}/episodes/{episode}/bundle` (gzip tar with an `X-Content-SHA256` header) once the
episode has finished.
