# Open Nav Harness

A navigation experiment platform derived from NavGPT-3. Holo owns the web console,
Planner sessions and durable queue; A100 owns model serving and Habitat execution.
The navigation tools have one implementation shared by all Planner runtimes.

## What is implemented

- OpenAI-compatible multimodal Planners with an explicit model/tool loop, alongside
  the inherited Codex and Claude SDK runtimes.
- LightNav-0 and official NavGPT VLA through the existing actor contracts.
- First-observation planning, bounded subtask execution, evidence-based review,
  replanning, explicit STOP, and per-subtask VLA history replacement.
- Validated service registration, trusted remote launch templates, GPU admission,
  ownership-checked start/stop, health and logs. External services are read-only.
- Immutable experiment snapshots; a SQLite queue with exclusive environment/VLA
  leases, Planner concurrency limits, pause/cancel, restart recovery and distinct attempts.
- An open-access React console for services, experiments, queue, replay,
  comparisons, CSV/trace exports and read-only historical imports.
- Content-addressed images and the exact API request/response journal, including
  context selection and tool provenance. SDK-internal model inputs remain opaque.

## Start the control plane

Python **3.11+**, Node **22+**, Linux and SSH are required. GPU services keep their
own existing environments; do not install the control-plane dependencies into Habitat.

```bash
uv sync --locked --extra dev
npm --prefix frontend ci
npm --prefix frontend run build
uv run open-nav-harness init
uv run open-nav-harness --settings deployment/platform.example.json serve
# A separate terminal/process:
uv run open-nav-harness --settings deployment/platform.example.json worker
```

For Codex/Claude planners, additionally install `uv sync --locked --extra dev --extra sdk`
and use `uv run --extra sdk ...` (or that environment's Python interpreter) when launching.
Local API planners do not import MCP or either model SDK; the development extra includes
MCP only to test adapter compatibility.

Open `http://127.0.0.1:8878`; the console and API require no login.
`serve --host 0.0.0.0` enables LAN access with full management access.
Use an HTTPS reverse proxy when transport confidentiality is required.
Credentials are environment-variable references, never experiment fields.

The console has two views: `/harness` (default) is a live, read-only run view with an instruction
picker, plan, tool timeline, memory and map; `/workbench` manages services, experiments, queue and
comparisons. Put text copies of environment episode files in `<state>/catalog/<dataset>_<split>.json.gz`
to enable the picker.

The composer's model menu lists registered Planner profiles with live availability.
Select a model before sending; its endpoint and budgets are frozen into the experiment.
Unavailable models remain visible. Register additional running endpoints as external services
in Workbench; replicas behind one endpoint use one profile.

In the Workbench:

1. Register an existing endpoint, or select an administrator-defined launch template.
2. Start and check owned services. Select the Planner, environment and actor.
3. Freeze an experiment configuration and enqueue it.
4. Open each attempt for decisions, images, plans, VLA context and navigation metrics.

The default state directory is local and ignored by Git. Override it consistently
with `--state /absolute/path` for both API and worker. Only one worker may own a state
folder. The worker and API are separate processes so an API restart does not interrupt
an episode. All command-line management flags precede the subcommand.

## Model configuration

A `PlannerProfile` records the runtime, model ID, endpoint, context/output/image
budgets, thinking mode, sampling, timeout and concurrency. API Planners share one
managed tool loop. `api_format: "chat_completions"` uses multimodal Chat Completions
and a matching `/tokenize` processor. `api_format: "responses"` calls `/responses`
directly, without Codex SDK or a second agent loop. It preserves function-call IDs
and encrypted reasoning items across turns and uses `/responses/input_tokens`
for counting. An optional `token_count_url` overrides the counting endpoint.

See `deployment/planner-responses.example.json` for an external GPT service profile.
Set `ONH_GPT_API_KEY` in both API and worker process environments using a private
environment file outside the repository/releases; snapshots contain only the
variable name. Register the profile with `POST /api/services` after deploying this
code. The existing model picker uses registered Planner profiles unchanged.

For gateways without a token-count API, explicitly set `token_count_mode: "estimate"`.
This uses UTF-8 byte length plus `image_token_estimate` (default 4096) per image,
excluding image base64. It is a conservative budget heuristic, not an exact count
or a guaranteed upper bound for arbitrary images; it can prune history earlier.
Actual usage still comes from the generation response. Remote counting failures
never silently fall back to estimation. Use `temperature: null` for models that
do not accept temperature and `effort` for Responses reasoning effort.

Responses defaults to `replay_reasoning: true`. If a gateway rejects returned
encrypted reasoning, explicitly set it to `false`: omit reasoning items from
subsequent inputs and do not request encrypted content. Tool calls, results,
images and the raw trace remain intact; reasoning effort is unchanged. This is
a gateway compatibility option, never an automatic retry of a navigation action.
Size estimated-context budgets for a complete motion feedback block: the local
gateway profile uses 131072 because 12 returned images plus text exceeded its
initial 65536 budget before the Planner could review or stop.

New model families implementing this protocol need a profile, not a new navigation
loop. A different wire protocol belongs in `open_nav_harness/providers/`. New actors
implement the actor service contract; they do not change Planner orchestration.

The two initial A100 profiles are Qwen3.6-27B and Qwen3.6-35B-A3B NVFP4. Their owned
vLLM services use `qwen3_coder` tool parsing and `qwen3` reasoning parsing. The
installed vLLM 0.19.1 requires `--mm-processor-cache-gb 0` when using `/tokenize`
before multimodal generation; the default cache caused a reproducible cache miss.
The NVFP4 export also embeds a 4096-token truncation setting in `tokenizer.json`.
Its serving template overrides this with `--mm-processor-kwargs '{"truncation":false}'`,
without editing the shared checkpoint. A 30-image probe verifies inputs beyond that cap.
Validate a deployment before navigation:

```bash
uv run python -m deployment.probe_planner \
  --url http://127.0.0.1:18101/v1 --model onh-qwen27b \
  --output state/probes/new-profile
```

This checks two-image input, a schema-valid function call, and equality of the
counted input tokens and actual generation usage. It is a protocol test, not a
navigation score. Output folders cannot overwrite earlier probes.

## Reliability and trace semantics

Movement calls are **never automatically retried**. If an episode dies or a movement
reply is missing, its resource leases remain blocked. Stop its owned environment
and VLA, use the replay page to reconcile the attempt, then restart the services and
resume the job. Recovery creates a new attempt only if no episode result was committed;
completed outcomes, including timeout failures, are retained without rerunning them.
Recovery never overwrites prior evidence. External uncertain resources require operator recovery;
the platform will not kill an external process to free a lease.

`completed` is an execution status, not navigation success. Metrics retain explicit
STOP, task acceptance and errors. Comparison summaries use the first completed
attempt per episode, count completed protocol errors as failures and show interruptions
separately. Missing cost is `null`, never an invented zero-dollar estimate.

`purpose=collection` requires the explicit `train` split. Successful execution alone
never admits a trace to training: exports remain `pending_review`, or `evaluation_only`
for development/evaluation runs. Route/scene deduplication, semantic review and
8-point label conversion belong to the downstream training admission pipeline.
No training-ready SFT dataset is claimed by this platform.

## Validation and maintenance

```bash
uv run pytest -q
uv run ruff check open_nav_harness
uv run ruff check --select F821,F823 navgpt
uv run mypy open_nav_harness
npm --prefix frontend run build
# Or omit ONH_CHROME after installing Playwright's Chromium:
ONH_CHROME=/opt/google/chrome/chrome npm --prefix frontend test
```

API schemas generate the frontend request types. The UI uses ordinary components,
not a second implementation of queue or experiment rules. Database migrations are
versioned and immutable. Source fingerprints prevent a queued experiment from silently
running a different implementation after a code edit.

See [architecture](docs/ARCHITECTURE.md), [operations](docs/OPERATIONS.md), and the
[inherited CLI/tool reference](docs/USAGE.md). `provenance.json` records the upstream
commit and hashes of the inherited working tree, including the prior experiment fixes.
Navigation and control-plane source are self-contained in this repository. The current
A100 deployment reuses existing model weights, scene assets and Python environments;
some of those environments still live beneath the original checkout. Their paths are
explicit deployment configuration, and those shared assets are not modified.
