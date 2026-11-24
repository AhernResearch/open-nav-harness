# Open Nav Harness

A navigation Planner platform over [sim-engine](../sim-engine) sessions. A kernel agent loop lets a
multimodal model observe, delegate legs to an executor, review the result and stop. The platform adds
runs from datasets, a dialogue/conversation view, service management and a React console.
It derives from NavGPT-3; `provenance.json` records the upstream commit. The inherited `navgpt`
package and the earlier experiment/queue pipeline have been removed, and Habitat itself lives in
sim-engine.

## What is implemented

- **Kernel** (`open_nav_harness/kernel/`): one agent loop over a sim-engine session, tool sets
  `point_v1` (pixel target, sim-engine plans and tracks) and `vla_v1` (language instruction walked by
  the LightNav actor), per-episode evidence and a create-only episode bundle.
- **Runs** (`platform/runs.py`): `POST /api/runs`, `/api/runs/compose`, a SQLite episode queue,
  a single worker, live stream and bundle download. `/harness` has the composer "从数据集运行" and an
  episode viewer.
- **Conversations** (`platform/conversations.py`, `navigation.py`): multi-turn dialogue with optional
  external Streamable HTTP MCP servers; a conversation can bind to a sim-engine episode and use the
  same tools.
- **Services** (`platform/services.py`, `/workbench`): register external endpoints, or start and
  stop owned ones from trusted templates over SSH, with health, logs and GPU admission.
- **Providers**: OpenAI-compatible Chat Completions and Responses clients with remote token counting
  (or an explicit resolution-aware estimate).
- **Actor**: the LightNav-0 service (`actors/`), run in its own CUDA environment.

Whether a deployed stack navigates well has not been established by this repository; tests use fake
simulators, planners and models. Diagnostic runs are not benchmark scores.

## Start

Python 3.11+, Node 22+, Linux and SSH. GPU services keep their own environments; do not install the
control-plane dependencies into them.

```bash
uv sync --locked --extra dev
npm --prefix frontend ci
npm --prefix frontend run build
uv run open-nav-harness init
uv run open-nav-harness --settings deployment/platform.example.json serve    # http://127.0.0.1:8878
uv run open-nav-harness --settings deployment/platform.example.json worker   # separate process
```

The API and worker need the same `--state` (default `state/`, git-ignored); only one worker may own a
state folder; management flags precede the subcommand. The console and API have no login: do not
expose them to untrusted networks. Credentials are environment-variable references.

Console: `/harness` (conversations, run composer, run viewer) and `/workbench` (services). To pick
instructions, put text copies of sim-engine episode files at `<state>/catalog/<dataset>_<split>.json.gz`.

A kernel run can also be started without the platform:
`python -m open_nav_harness.kernel SPEC.json --episodes ID... --output DIR [--lanes N]`.

## Configuration

- Platform settings: [`deployment/platform.example.json`](deployment/platform.example.json) (host,
  port, SSH config, service templates, `conversations_config`).
- `conversations_config`: JSON with `profiles` (planner profiles), `default_model`, `mcp` and
  `navigation` (sim-engine targets). It feeds both conversations and the run composer.
- A `PlannerProfile` freezes endpoint, model, budgets, sampling, `api_format`
  (`chat_completions` or `responses`) and `credential_env`. See
  [`deployment/planner-responses.example.json`](deployment/planner-responses.example.json) and
  `python -m deployment.probe_planner` to validate a deployment. A new model needs a profile, not code.
- Actor: [`deployment/actor-lightnav.example.json`](deployment/actor-lightnav.example.json), see
  [docs/LIGHTNAV.md](docs/LIGHTNAV.md).

## Guarantees

- Motion is never retried. A lost reply or a worker restart ends the episode as
  `failed_or_uncertain` with its evidence; nothing is replayed.
- Goals, reference paths, rewards and global poses never reach the Planner or the actor.
- Records are create-only. `completed` is an execution status, not navigation success; episodes stay
  `training_admission: pending_review`. Collection runs need a `*_train` dataset.
- A worker runs only runs whose kernel source digest matches the code it loaded.

## Checks

```bash
uv run pytest -q
uv run ruff check open_nav_harness
uv run mypy open_nav_harness
npm --prefix frontend run build
npm --prefix frontend test        # Playwright; set ONH_CHROME to use an installed Chrome
```

CI (`.github/workflows/checks.yml`) also regenerates `frontend/openapi.json` and builds the wheel.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [operations](docs/OPERATIONS.md), including the steps
  before deploying this branch.
- [Production kernel interface](docs/PRODUCTION_KERNEL.md), [LightNav actor](docs/LIGHTNAV.md),
  [MCP client and semantic map](docs/SEMANTIC_MAP.md).
- `docs/evidence/` holds historical measurements; `provenance.json`, `LICENSE`, `NOTICE` and
  `licenses/` record the inherited sources.
