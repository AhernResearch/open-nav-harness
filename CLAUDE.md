# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Open Nav Harness is a navigation Planner platform: a kernel agent loop over sim-engine sessions, with
runs, conversations and service management behind a FastAPI API and a React console. It is a NavGPT-3
derivative; the `navgpt` package has been removed. Don't edit or manage processes owned by neighboring
repos (sim-engine owns Habitat), and keep `provenance.json` and the inherited licenses intact.
`AGENTS.md` has the contributor rules. Docs are in English except `docs/SEMANTIC_MAP.md` (Chinese).

## Commands

The control plane needs Python 3.11+ (uv) and Node 22. GPU services use their own environments.

```bash
uv sync --locked --extra dev
npm --prefix frontend ci

# Required checks (mirrors .github/workflows/checks.yml)
uv run pytest -q
uv run pytest tests/test_runs.py::test_name -q   # single test
uv run ruff check open_nav_harness
uv run mypy open_nav_harness
npm --prefix frontend run build                  # openapi-typescript types + tsc --noEmit + vite build
npm --prefix frontend test                       # Playwright; ONH_CHROME=/path/to/chrome if needed
```

After an API schema change, regenerate the spec first:
`uv run open-nav-harness --state /tmp/onh-schema openapi > frontend/openapi.json`, then build.

Running (management flags go before the subcommand; API and worker share `--state`; one worker per
state folder):

```bash
uv run open-nav-harness init
uv run open-nav-harness --settings deployment/platform.example.json serve    # :8878
uv run open-nav-harness --settings deployment/platform.example.json worker
uv run open-nav-harness --state DIR backup --output FILE
uv run python -m open_nav_harness.kernel SPEC.json --episodes ID... --output DIR   # no platform
uv run python -m deployment.probe_planner --url URL/v1 --model ID --output state/probes/NAME
python -m open_nav_harness.actors.lightnav_engine --config FILE    # actor env only
```

Tests use fake simulators, planners and models; never start a real benchmark from a test.

## Architecture

One package, `open_nav_harness/`, plus `frontend/`:

- `kernel/`: `loop.py` (one agent loop), `tools.py` (tool sets `point_v1`, `vla_v1`; a tool set is the
  permission boundary), `executors.py` (`PointExpert`, `VlaActor`), `environment.py` (sim-engine
  client), `evidence.py`, `recorder.py` (episode bundle), `episode.py` (preflight, `run_episode`,
  `run_batch`), `replicas.py` (`ReplicaPool`, planner replica pinning), `spec.py` (`KernelSpec`).
- `platform/`: `api.py`, `runs.py` (`RunStore`, `RunScheduler`, runs API), `worker.py` (runs only the
  scheduler), `conversations.py` and `navigation.py` (dialogue, MCP, binding an episode), `services.py`
  (templates, SSH start/stop, health), `kernel_feed.py`, `catalog.py`, `database.py` and Alembic
  `migrations/`.
- `providers/`: `compatible.py` and `responses.py` (wire clients), `runtime.py` (`ManagedRuntime`,
  dialogue loop only), `tools.py`, `vision.py`. A new wire protocol goes here.
- `core/`: config contracts, artifacts/journal, context budget, `version.py` (kernel digest).
- `actors/`: the LightNav actor service. `lightnav.py` is torch-free; `lightnav_engine.py` needs the
  actor's CUDA environment and is never imported by the control plane.
- `frontend/`: presentation only; run and tool rules stay server-side.

New models get a `PlannerProfile`, not model-name branches. New actors implement the HTTP contract in
`docs/LIGHTNAV.md`. Design details: `docs/PRODUCTION_KERNEL.md`, `docs/ARCHITECTURE.md`,
`docs/OPERATIONS.md`.

## Invariants (violations are bugs)

- Motion is never retried at any layer. A lost motion reply ends the episode as `failed_or_uncertain`
  with evidence kept. Only model HTTP generation retries.
- Goals, reference paths, rewards, geodesic metrics and global poses never reach the Planner or the
  actor; they stay in simulator-side evidence. When adding a field, decide which side it is on.
- An invalid tool call rejects the whole unexecuted batch with repair feedback; it is not an execution
  failure. Repeated malformed replies end the episode as `planner_protocol_failure`.
- Records are create-only: `Recorder` never replaces a file. `completed` is an execution status, not
  navigation success. Episodes stay `training_admission: pending_review`; collection needs a `*_train`
  dataset.
- Kernel digest freeze: a run stores the digest of `kernel/`, `core/` and `providers/` source, and a
  worker runs only runs whose digest matches the code it loaded. Changing those packages needs a
  worker restart.
- Service control requires generated ownership tokens (`OPEN_NAV_HARNESS_OWNER`). Browser input never
  becomes argv; argv comes only from trusted settings templates. Credentials are env-var references,
  never in specs or records.
- Migrations are added, never edited. Legacy tables stay in the database, unread.
- `state/` is local runtime data and is git-ignored.
