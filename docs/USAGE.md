# Navigation runtime reference

Start with the [README](../README.md) for installation, data, checkpoints and a
first run. Use `open-nav-harness` for service management, queueing, recovery and
replay. This reference covers the component CLI, configuration and service interfaces. All commands run from
the repository root.

## Architecture and naming

The repository uses three executable component names:

- **NavGPT Planner** (`navgpt.planner`) is the semantic decision policy. It
  interprets harness-assembled context, selects registered tools, reasons
  about progress and recovery, and makes the final stop decision. It does not
  own the context store or compilation policy.
- **NavGPT Environment** (`navgpt.environment`) owns Habitat simulator state,
  observations, and simulated movement.
- **NavGPT VLA** (`navgpt.vla`) owns low-level multi-view visual-action
  inference using the **NavGPT3** model and processor.

The **NavGPT-3 harness** owns the model-facing interaction framework. A
**NavGPT-3 Runtime** above the harness governs logical-thread lifecycle,
scheduling, permissions, and motion authority in the wider system design.
This repository contains the harness; the deployment runtime is released
separately.

Process boundaries use `api.py`, orchestration uses `runner.py`, model-visible
functions live in `planner/tools/`, provider adapters live in `planner/runtimes/`,
and checkpoint implementation code lives under `vla/runtime/`.

```text
harness context path ──> NavGPT Planner
          ▲                     │
          │ tool results        │ registered tool call
          │                     v
  route/node state <── MCP capability registry
          ▲                     │
          └── NavGPT Environment┼──> NavGPT VLA
                perception/action     waypoint prediction
```

NavGPT Planner runs on Python 3.11+. NavGPT Environment normally runs in the
older Python environment required by `habitat_sim 0.1.7`. NavGPT VLA runs in a
CUDA/PyTorch environment. Keeping these as HTTP-separated processes avoids an
unmaintainable combined dependency stack.

The control model sees only the navigation information and MCP tools selected by
the configured tool set. Benchmark scoring, when used, is an experimental
wrapper around completed runs rather than a model-visible capability.

The Planner chooses the semantic sequence of registered operations. The harness
shapes that behavior through the active tool set, model-visible context,
schema and argument validation, navigation-state contracts, returned evidence,
and termination semantics. NavGPT Environment commits simulator transitions.

Context management belongs to the harness: what the Planner sees, how it is
presented, and where it came from can be inspected outside the model. Every
episode records its briefing, tool calls, tool results and metrics. API runtimes
also record exact requests; SDK-managed conversation internals remain opaque.


### Repository layout

```text
navgpt/
  cli.py          navgpt run | preflight | check-data | check-checkpoint
  settings.py     experiment config loading and validation
  planner/        briefings, shared tool profiles, runtime adapters and episode runner
  environment/    R2R/RxR loading, Habitat observations, actions, maps, metrics
  vla/            NavGPT3 model, processors, inference service, checkpoint check
configs/
  paths.example.yaml   machine paths template
  experiments/         one self-contained config per experiment
docs/             usage reference, dataset and checkpoint guides
```

## Experiment configs

An experiment config is one self-contained YAML or JSON file with up to six
sections. Unknown sections, keys or values are an error. The run name, and so the output
directory, is the config's path under `configs/experiments/` joined with
underscores (`lightnav/planner_r2r.json` → `lightnav_planner_r2r`) unless `name` is set. Omitting `planner`
runs the VLA alone.

The historical `gpus` field remains accepted when loading frozen configurations.
Service placement and parallel lanes are managed by the platform and trusted
deployment templates; the component CLI does not start services.

**dataset** — which episodes are run.

| Key | Default | Meaning |
|---|---|---|
| `name` | `r2r` | `r2r` or `rxr` |
| `split` | `val_unseen` | split directory, or a label for `episodes_file` |
| `episodes` | `all` | `all`, or indices such as `0-99` or `3,7` |
| `episodes_file` | — | curated episode file: a key under `episodes` in `paths.yaml`, or a path |
| `ground_truth_split` | — | split whose dense ground truth scores an `episodes_file` |
| `ground_truth_file` | — | explicit dense ground-truth file instead |
| `languages`, `roles` | `[en-US, en-IN]`, `[guide]` | RxR instruction selection |

**environment** — the Habitat service.

| Key | Default | Meaning |
|---|---|---|
| `url` | `http://127.0.0.1:9200` | service address |
| `max_steps` | `500` | episode step limit, shared with the runner |
| `forward_view_px` | `512` | forward camera image size |
| `panorama_view_px` | `400` | size of each panorama view |
| `turn_angle_deg`, `step_size_m`, `allow_sliding` | 15, 0.25, true | VLN-CE action settings; the VLA-only configs turn 30° and stop RxR at 400 steps, as VLNCE-EVAL does |

**vla** — the NavGPT3 service.

| Key | Default | Meaning |
|---|---|---|
| `url` | `http://127.0.0.1:8000` | service address |
| `checkpoint` | `navgpt3-8b` | key under `checkpoints` in `paths.yaml`, or a path |
| `visual_token_budget` | `3072` | visual tokens shared across the multi-view history |
| `frame_cache` | `precompute` | `precompute` or `on_demand` image preprocessing |
| `gpu_memory_fraction` | `0` | cap on this process's GPU memory share (0: none) |
| `seed` | `20260910` | base seed for VLA resets in study runs (plus the episode index) |

**planner** — the model that makes decisions.

| Key | Default | Meaning |
|---|---|---|
| `runtime` | `claude` | `openai_compatible`, `claude` or `codex`; API runtimes require a `profile` |
| `model` | `claude-opus-5` / `gpt-6-astra` | provider model id; the default follows the runtime |
| `effort` | provider default | reasoning effort: `low`, `medium`, `high` or `xhigh` for either runtime, plus `max` (Claude) or `none`/`minimal` (Codex); the shipped configs use `high` |
| `service_tier` | inherited | Codex service tier |
| `tools` | follows mode | exact inventory from [tool profiles](../navgpt/planner/tool_profiles.py); agentic defaults to seven tools |
| `max_turns` | `200` | Planner responses per episode |
| `episode_timeout_s` | `2400` | wall-clock limit per episode |
| `max_budget_usd` | 5 (Claude) / null (Codex) | cost cap per episode; Codex reports no cost and needs null |

**harness** — how the Planner sees and drives the robot.

| Key | Default | Meaning |
|---|---|---|
| `interface` | `standard` | label and map presentation; historical variants remain readable |
| `mode` | `global` | `agentic` for model-owned plans/memory, `subtask` for strict review, `global` for route delegation |
| `review_every` | `0` | VLA steps between Planner reviews of a rollout; 0: only when the VLA stops |
| `route_finding` | true | `navigate_relative` follows the walkable route; false walks straight |
| `vla_steps_per_call` | `200` | step cap on one `navigate_by_instruction` call |

**output**

| Key | Default | Meaning |
|---|---|---|
| `dir` | `outputs` | where run directories are written |
| `save_frames` | false | keep observed frames and an action log for replay |

Machine paths live in `configs/paths.yaml` (template: `configs/paths.example.yaml`):
`data.r2r`, `data.rxr` and `data.scenes`; curated episode files under `episodes`;
checkpoint directories under `checkpoints`; optionally `codex_bin`, the Codex CLI
used by the Codex runtime. Commands accepting an experiment config also accept
`--paths` to use another paths file. Service interpreter paths belong in trusted
platform deployment templates.

## Commands

| Command | Environment | Purpose |
|---|---|---|
| `python -m navgpt.environment --config CFG [--port P] [--gpu G] [--blind]` | Habitat | serve the experiment's episodes |
| `python -m navgpt.vla --config CFG [--port P] [--gpu G]` | VLA | serve the experiment's checkpoint |
| `navgpt run --config CFG [--episodes SPEC] [--shard I/N] [--env-url U] [--vla-url U]` | Planner | run or resume the experiment |
| `navgpt preflight --config CFG` | Planner | check services, tools and model without running episodes |
| `navgpt check-data --config CFG` | Planner | check episodes, dense ground truth and scene files |
| `navgpt check-checkpoint --config CFG` | Planner | check the configured checkpoint's files and tensor names |

`navgpt run` is the component runner for separately managed services. Production
runs use the durable platform queue so configuration/source provenance, leases,
attempts and recovery are recorded together. Use [execution pools](ASYNC_INFERENCE.md)
for concurrency and [Workbench imports](OPERATIONS.md#historical-results-and-exports)
for read-only replay of historical run folders. The retired `launch`, `merge` and `view` subcommands have been removed.

## Planner tools

New interactive runs use the seven-tool agentic surface in the
[current tool contract](TOOL_CONTRACT.md). It covers plan patches, distance-bounded
execution, topology captions, no-progress handling and STOP evidence. The exact
inventory is frozen with each experiment and checked before dispatch.

Historical `global` and strict `subtask` modes remain supported by the same
`NavigationSession`. Their inventories live in
[tool profiles](../navgpt/planner/tool_profiles.py). Non-standard display variants,
capability bundles and periodic review settings remain available to read and
reproduce frozen configurations; the old per-model study matrix is no longer
shipped as runnable examples. GPU studies require an explicit isolated deployment
and development manifest.

## Planner runtimes

All adapters share navigation tools, briefings and turn accounting. The managed
API runtime uses the explicit model/tool loop described in the
[README](../README.md#model-configuration); optional SDK adapters use MCP.

- `runtime: claude` uses the Claude Agent SDK, signed in with a Claude
  subscription (`/login` or `CLAUDE_CODE_OAUTH_TOKEN`) or with API, cloud-provider
  or gateway credentials from the environment, which take precedence. It
  enforces `planner.max_budget_usd` per episode.
- `runtime: codex` uses the pinned Codex Python SDK with your local Codex
  sign-in: a ChatGPT plan or an OpenAI API key.
  Codex reports no cost, so `max_budget_usd` must be null; turn and time limits
  still apply. The adapter disables every other configured MCP server and the
  built-in shell, file, browser and search tools, and checks the registered tool
  inventory before the first prompt. Set `codex_bin` in `paths.yaml` when the
  model needs a newer Codex CLI than the one bundled with the SDK.

`preflight` checks the selected runtime. With an API key, the Claude path makes a
4-token model call; with a subscription, it sends one short request through the
CLI. The Codex path checks the local login and the advertised model list without
inference. A spent subscription usage window makes episodes wait and retry every
15 minutes, for up to 6 hours, instead of being scored.

## Service contracts

NavGPT Environment:

| endpoint | purpose |
|---|---|
| `GET /health` | dataset, split, language/role selection, episode count, dense-GT coverage, data fingerprint, step limit, available verbs |
| `POST /call/r2rce__<verb>` | simulator observation/action call |
| `POST /env-panel/field/<name>` | planner-only episode placement |
| `POST /env-panel/action/<name>` | planner-only control action |

NavGPT VLA:

| endpoint | purpose |
|---|---|
| `GET /info` | model identity and inference-service status |
| `POST /reset` | clear temporal state before an episode |
| `POST /act` | one multi-view inference step and waypoint action |
| `POST /observe` | append views to the visual history without predicting (ablation repair) |
| `GET /state` | history, cache and RNG fingerprints for seeded ablation runs |

Both services are stateful at the episode level and are not concurrency-safe:
one service pair serves one run at a time. The `r2rce__` verb prefix is shared
by both datasets; dataset identity comes from `/health`, and the runner refuses
a service whose dataset, split, step limit or ground-truth coverage does not
match the experiment.


## Outputs and replay

| Path | Contents |
|---|---|
| `summary.json` | run config, per-episode records, aggregate over scored episodes |
| `episode_<i>.jsonl` | event stream: briefing, tool calls and results, metrics |
| `work/ep<i>/raw.jsonl` | raw provider message stream, images stripped |
| `live/ep<i>/` | observed frames and action log (`output.save_frames`) |
| `labels_<i>.json` | node names and captions (tool sets with `annotate_node`) |

An episode is excluded from scoring when it never took a step after an error, when
the provider refused the request, or when an ablation run could not keep a
consistent VLA history; the aggregate and `scored_count` reflect that. Import
these folders through Workbench for read-only replay. Platform attempts already
expose their trajectory, tool timeline, observations and raw journal in the console.

## Navigation-state integrity

- The place graph is derived from the append-only walked path. IDs are
  chronological and stable, and teleport-sized jumps do not become graph edges.
- `observed_map` paints only RGB-D-observed cells; unknown space never becomes
  known-free space.
- VLA `should_stop` returns control to the Planner but does not end the episode;
  the Planner decides whether to inspect, continue, return, repair, or terminate.
- A VLA waypoint off the walkable floor becomes one discrete action, as in
  VLNCE-EVAL: a turn toward the VLA's heading change when it exceeds 0.1 rad or
  the previous VLA step left the pose unchanged, otherwise a forward step that
  slides along the obstacle.
- Tool registration, schema validation, movement limits, and state updates are
  enforced independently of the workflow sequence suggested in the briefing.


## Rendering

Real RGB observations require an EGL/OpenGL device. CUDA availability alone is
not enough. Habitat binds its GL context to the thread that creates the
simulator, so all simulator access is serialized through the dedicated simulator
thread in `navgpt/environment/r2r.py`.

`--blind` (on `python -m navgpt.environment`) keeps real episode data, pathfinding, and dynamics but returns synthetic
pixels. It is suitable for API plumbing tests, never for navigation results.


## License and attribution

The source code is released under the Apache License 2.0 in [LICENSE](../LICENSE),
with third-party attribution in [NOTICE](../NOTICE). Model weights and datasets are
distributed separately under their own terms; the source license does not grant
rights to them.

The Planner's evaluation loop, Claude adapter, and MCP tool-server pattern build on
[AgentCanvas](https://github.com/jianzhou0420/AgentCanvas). The NavGPT VLA service files `api.py`, `navigation.py`, and `images.py`
are derived from VLNCE-EVAL, built on EPIC Lab's
[NaVid-VLN-CE](https://github.com/jzhzhang/NaVid-VLN-CE), and keep its MIT license in
[navgpt/vla/LICENSE](../navgpt/vla/LICENSE). The NavGPT3 model and processor classes
extend the Apache-2.0 Qwen3-VL implementation in Hugging Face Transformers.
