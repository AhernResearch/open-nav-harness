# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Open Nav Harness is an independent NavGPT-3 derivative: a navigation experiment platform. The Holo
workstation runs the web console, API, durable queue and Planner sessions. The A100 hosts model
serving, Habitat and actor (VLA) workers. `AGENTS.md` contains the contributor rules. Don't edit
or manage processes owned by neighboring repos, and keep `provenance.json` and the inherited
licenses intact. Docs are in English.

## Commands

The control plane needs Python ≥3.11 (uv) and Node 22. GPU services use their own existing
environments. Never install control-plane deps into Habitat.

```bash
uv sync --locked --extra dev            # add --extra sdk for Codex/Claude planners
npm --prefix frontend ci

# Required checks (mirrors .github/workflows/checks.yml)
uv run pytest -q
uv run pytest tests/test_memory_contract.py::test_name -q   # single test
uv run ruff check open_nav_harness
uv run ruff check --select F821,F823 navgpt                  # navgpt only gets undefined-name checks
uv run mypy open_nav_harness
npm --prefix frontend run build         # = openapi-typescript types + tsc --noEmit + vite build
ONH_CHROME=/opt/google/chrome/chrome npm --prefix frontend test   # Playwright; omit env if Chromium installed
```

When an API schema changes, regenerate the frontend types first:
`uv run open-nav-harness --state /tmp/onh-schema openapi > frontend/openapi.json` (the build then
regenerates `src/api-schema.ts`). Playwright starts its own server on port 18878 from
`.venv/bin/python`. Use it for any changed user workflow.

Running the platform (management flags go **before** the subcommand; API and worker must use
the same `--state`, and only one worker may own a state folder):

```bash
uv run open-nav-harness init
uv run open-nav-harness --settings deployment/platform.example.json serve    # http://127.0.0.1:8878
uv run open-nav-harness --settings deployment/platform.example.json worker   # separate process
uv run open-nav-harness --state DIR backup --output FILE
uv run python -m deployment.probe_planner --url URL/v1 --model ID --output state/probes/NAME
```

The component `navgpt` CLI (`run`, `preflight`, `check-data`, `check-checkpoint`) and its
YAML/JSON examples under `configs/experiments/` are documented in `docs/USAGE.md`. Machine paths live in `configs/paths.yaml`, copied from `paths.example.yaml`.

Unit and UI tests must never start the formal benchmark. GPU integration uses a separate
deployment and a train/dev manifest.

## Architecture

There are two packages in one distribution:

- `open_nav_harness/`: the control plane.
  - `platform/` holds the FastAPI API, the SQLite (WAL) queue with leases, the worker/scheduler, service lifecycle, per-episode subprocess execution and results/exports. Schema changes go through Alembic `migrations/`. Add a new migration and never edit an applied one.
  - `core/` holds config contracts, content-addressed artifacts, the context budget (`context.py`) and recovery evidence.
  - `providers/` holds the OpenAI-compatible `CompatibleClient` and `ManagedRuntime` (the explicit model/tool loop). A new wire protocol is added here.
- `navgpt/`: the inherited navigation stack.
  - `planner/` contains the runner, prompts, config, and the SDK runtimes (`runtimes/` for Claude/Codex).
  - `planner/tools/` contains `NavigationSession` (`session.py`), the only navigation tool implementation. It also has the `ToolRegistry` (schemas and strict validation, no MCP import), `mcp_tools.py` (a thin MCP adapter for SDK runtimes), `transport.py` (in-process or actor-worker RPC), `subtasks.py`, `agentic.py` and `memory.py` (`TopologyMemory`).
  - `environment/` is the Habitat service and evaluator metrics. It runs in its own interpreter.
  - `vla/` is actor inference: native NavGPT3 action head, LightNav, and `worker.py` (the owned actor worker that hosts a `NavigationSession` on the A100 over SSH-forwarded loopback HTTP).
- `frontend/`: the React/Vite console. It only presents data; queue and experiment rules stay server-side.

All planner runtimes share one tool registry and session. New models get a `PlannerProfile`,
not model-name branches. New actors implement the actor service contract without touching Planner
orchestration. Harness modes: `agentic` (model-owned plan/memory workflow), `subtask` (the earlier
strict review workflow, kept for comparisons) and `global` (full-route delegation).

The current agentic tool contract is `docs/TOOL_CONTRACT.md`. Tool inventories live in
`navgpt/planner/tool_profiles.py`, shared by settings, preflight and session registration.
`docs/ARCHITECTURE.md` describes ownership, durable execution and context/causality semantics. `docs/OPERATIONS.md` covers deployment,
failure handling and backup. `docs/LIGHTNAV.md` covers the LightNav backend.

## Invariants (violations are bugs)

- **Movement is never automatically retried** at any layer. A missing or uncertain movement reply leaves an unclosed edge and keeps resource leases blocked until an operator reconciles it. Only model HTTP generation retries transport errors.
- Goals, reference paths, geodesic metrics and success labels are never sent to the Planner or actor. They exist only as evaluation artifacts.
- Malformed tool arguments reject the whole unexecuted batch with repair feedback, which is not an execution failure. Duplicate call IDs, truncated responses and missing explicit STOP are recorded failures.
- Experiments freeze config, launch templates and source fingerprints. Changing navigation, provider, context or executor code invalidates queued experiments, so pause jobs before deploying. Completed attempts are never overwritten, and recovery creates a new attempt.
- `completed` is an execution status, not navigation success. Missing cost is `null`. Collection requires the `train` split, and exports stay `pending_review` or `evaluation_only` because training admission happens downstream.
- Service control requires generated ownership tokens (`OPEN_NAV_HARNESS_OWNER`). Browser input never becomes executable argv, because argv comes only from trusted config files. Credentials are env-var references, never experiment fields.
- `state/` is local runtime data (DB, runs, deployment config) and is git-ignored.
