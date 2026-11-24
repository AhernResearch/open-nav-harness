# Production kernel: interface draft

Status: draft for review (2026-11-23). Nothing here is implemented. Plan:
`~/.claude/plans/delegated-exploring-ladybug.md`.

## Ownership

| Component | Owns | Never owns |
|---|---|---|
| sim-engine | Habitat state, panorama capture, pixel grounding, path planning, tracking, legs, R2R scoring | Prompts, model calls, tool schemas |
| open-nav-harness kernel | Agent loop, tools, context, Planner/actor clients, trace and leg records | Simulator state, geometry, goals |
| open-nav-harness platform | Runs, queue, lanes, API, console | Navigation semantics |
| nav-data | Task construction, submission over HTTP, result filtering, releases | Agent loop, simulator calls |

The harness no longer ships a Habitat environment. `navgpt/` stays in place, unused by the new path,
until the parity check in phase 4.

## 1. sim-engine session extensions

Everything is added to the existing stateful session (`sim_server.py`, `pool.py SessionHost`,
`session.py Session`). Existing CAS, `request_id` idempotency, 409/422/429 semantics, position and
collision accounting and `metrics()` apply unchanged.

### Controller context

`Controller.execute(env, physics, command, remaining)` gains one argument, `context: SessionContext`.
It is read-only for the existing waypoint and velocity controllers, which ignore it.

```python
class SessionContext(Protocol):
    def observation(self, observation_id: str) -> PanoramaSnapshot   # raises Conflict if stale
    navmesh: NavMeshContext                                           # pf, planners, guard, expert_profile
    def positions_since(self, event_index: int) -> list[list[float]]
    def leg(self, leg_id: int) -> Leg
```

`SessionHost` builds one `NavMeshContext` per scene (`planning/geometry.py:20`), cached alongside the
warm environment.

### Observe (no motion)

`POST /v1/sessions/{id}/observe` with `{"request_id": str}`. The reply is:

```json
{"observation_id": "obs-3", "event_index": 41,
 "views": {"front": {"sha256": "...", "width": 640, "height": 480}, "right": {...}, "back": {...}, "left": {...}}}
```

- It captures `env.observe_views(env.pose().yaw)` (four RGB-D views from the session camera).
- RGB is fetched with `GET /v1/sessions/{id}/observations/{observation_id}/{view}` (PNG).
- Depth and sensor poses stay on the server; only grounding uses them.
- A snapshot is valid only while `event_index` is unchanged. Any motion makes it stale (409).
- Observing does not count as an action and never ends the episode.

### PointGoalCommand

```json
{"kind": "point_goal", "observation_id": "obs-3", "view": "left", "point": [412, 760]}
```

`point` uses the existing normalized 0–1000 grid with the origin at the top left.

1. **Ground.** `backproject` the pixel with the snapshot depth (`navigation/visual_geometry.py:8`).
   Then check it against `safe_floor_mask` evaluated around the pixel (`navigation/safe_image.py:41`)
   and `validate_floor_target`, and snap it to the NavMesh.
2. **Plan** with `plan_with_preferences` (`navigation/guidance.py:150`) from the current position.
3. **Drive** with `navigation/drive.py`, a new module. It is the tracker → `velocity_step` loop extracted
   from `collection/simulation.py:103 rollout`, so collection and sessions share one implementation.
   Every control step is stored as a session frame and action, exactly as `VelocityController` does.

A rejected grounding or plan is a normal reply with zero actions and a reason. It does not count
toward the `stalled` termination, because re-selecting at the same pose is expected.

The step reply gains a `leg` object:

```json
{"leg": {"leg_id": 4, "kind": "point_goal", "status": "reached",
         "reason": null, "start_event_index": 41, "end_event_index": 97,
         "distance_m": 3.12, "net_displacement_m": 2.95, "slide_steps": 2}}
```

`status` is one of `reached | stopped_short | blocked | leg_budget | episode_budget | invalid_target | unreachable`.
`reason` is a short public string such as `"pixel is not floor"`, `"floor is on another level"` or
`"no path with enough clearance"`. Grounded world coordinates are kept in the session's leg log
(`metrics()`), not in the reply.

### RetraceCommand

```json
{"kind": "retrace", "leg_id": 2}
```

It drives back along the recorded positions from the current event to `leg(2).end_event_index`,
reversed and simplified, with the same tracker. It records a leg of `kind="retrace"`. The route is
the robot's own odometry history, so a real robot could do the same.

### Guards

- The harness requires `protocol.reward_mode == "none"` at preflight. Otherwise each step reply
  would carry `training_reward` (geodesic progress).
- `metrics()` keeps its current shape and adds `legs` (the full leg log, with grounded goals).
  The harness reads it only after `finish`, and writes it to evaluation artifacts.

## 2. Harness kernel (`open_nav_harness/kernel/`)

```python
class Environment(Protocol):              # SimEngineEnvironment
    async def open(self, episode: EpisodeRef) -> EpisodeStart   # instruction, session id
    async def observe(self) -> Observation                      # observation_id, view images
    async def command(self, command: dict) -> StepReply         # CAS/request_id handled inside
    async def frames(self, start: int, end: int) -> list[Frame] # for VLA input and export
    async def finish(self, reason: str) -> dict                 # metrics, evaluation only
    async def close(self) -> None

class Executor(Protocol):                 # chosen by the run spec, invisible to the Planner
    async def run_leg(self, env: Environment, leg: LegRequest) -> LegOutcome

class PointExpert(Executor): ...          # one point_goal command
class VlaActor(Executor): ...             # nav-train compact_actor: begin_leg, then act → velocity steps

class Tool(ABC):
    name: str; schema: dict; description: str
    async def run(self, state: EpisodeState, args: dict) -> ToolResult   # text + image artifacts
```

`AgentLoop` keeps `providers/compatible.py CompatibleClient`, the Responses client,
`core/context.py ContextWindow` and `StrictValidator`. It replaces the navgpt-dependent part of
`providers/runtime.py`.

A turn proceeds like this:

1. Build the context.
2. Generate.
3. Validate every call in the batch. Any invalid call rejects the whole batch with repair
   feedback and no execution.
4. Execute the calls in order. `stop` must come last.
5. Append the results.

The loop handles three cases specially:

- A response with no tool call and with text that contains a tool-call-like pattern is recorded as
  `malformed_call` and answered with repair feedback.
- After 3 malformed responses in a row, the episode ends as `planner_protocol_failure`.
- A response with no tool call at all also gets repair feedback, once; the episode does not end
  silently.

`platform/conversations.py` currently builds on `ManagedRuntime` and navgpt tools. It moves to
`AgentLoop` with its own tool set in phase 3. Its behavior is unchanged.

### Tool set `point_v1`

| Tool | Arguments | Effect | Result to the Planner |
|---|---|---|---|
| `observe` | `views: list[front\|right\|back\|left]` (1–4, unique) | sim-engine observe | One captioned image per view |
| `delegate` | `view`, `point: [int,int]` 0–1000, `instruction: str` (1–200 chars) | The `Executor` runs one leg from the **latest** observation | `leg_id`, `status`, `reason`, distance, then four new views |
| `return_to` | `leg_id: int` | Retrace to that leg's end | Same as `delegate` |
| `update_plan` | `steps: list[{text, done}]` (1–20) | Harness-only checklist | `ok`. The latest plan is pinned in context |
| `stop` | `status: complete\|abort`, `reason: str` | `finish(policy_stop)` | Episode ends |

- `delegate` requires that the chosen view was returned at the current pose. Otherwise it fails as
  repairable feedback, without moving.
- Every motion result also lists the legs so far in a compact form, e.g.
  `legs: 1 reached "through the kitchen door"; 2 blocked "toward the stairs"`.
  This is how `return_to` IDs are known; there is no separate memory tool.

### Context

- The system prompt and the first user message are fixed.
- Tool text is always kept.
- Images are pruned oldest-first beyond `max_images`, and the latest motion result is always kept.
- The current plan is re-sent as a short pinned state block when older turns are pruned.

This reuses the `ContextWindow` bounded policy without the memory and map projections.

## 3. Runs API and concurrency

```http
POST /api/runs          RunSpec -> {run_id}
GET  /api/runs/{id}     status, counts, throughput
GET  /api/runs/{id}/episodes            per-episode status and summary
GET  /api/runs/{id}/events              SSE: episode started/turn/leg/finished
GET  /api/runs/{id}/episodes/{eid}/bundle   export (see §4)
POST /api/runs/{id}/action              pause | resume | cancel
```

```jsonc
// RunSpec
{"name": "r2r-train-point-pilot", "purpose": "collection",          // development|evaluation|collection
 "planner_service": "qwen27b", "tool_set": "point_v1",
 "environment": {"service": "sim-a100", "dataset": "r2r_train", "camera": "trained_ce30"},
 "executor": {"kind": "point_expert"},                               // or {"kind":"vla","service":"compact-14k"}
 "budgets": {"max_turns": 40, "max_actions": 500, "timeout_s": 1200, "leg_max_actions": 200},
 "lanes": 8,
 "episodes": ["4956", "3473"]}                                        // dataset episode_id strings
```

- A run is stored as a frozen experiment plus a job: one Alembic migration adds the `runs` view
  columns, and existing tables are reused. The source fingerprint covers `kernel/` and `providers/`.
- One worker process runs episodes as asyncio tasks. There is no subprocess per episode, and preflight
  runs once per run: tool schema hash, model list, sim-engine health and contract hash, reward mode.
- `EnvLanePool` holds at most `lanes` sim-engine sessions. sim-engine 429 is backpressure, and the
  lane waits. A lane whose motion reply is missing or errored is quarantined, and its episode becomes
  `failed_or_uncertain`. Other lanes continue. A quarantined lane is released only by an operator
  reconcile.
- `PlannerPool` gives each replica a semaphore of size `max_concurrency` and pins each episode to one
  replica for prefix caching. A request queues instead of failing when the pool is full.
- Restart: running episodes without `result.json` become `failed_or_uncertain`. They are never resumed
  or replayed.

## 4. Episode bundle

```
<attempt>/spec.json                  frozen RunSpec + episode ref + fingerprints
<attempt>/trace/requests.jsonl       exact wire requests/responses (existing Journal)
<attempt>/artifacts/<sha256>.png     content-addressed images
<attempt>/legs/0001.json             leg record (below)
<attempt>/result.json | failed_or_uncertain.json
<attempt>/evaluation.json            sim-engine metrics() after finish; never shown to models
```

```jsonc
// legs/0001.json
{"leg_id": 1, "kind": "point_goal", "instruction": "walk to the doorway on the left",
 "observation_id": "obs-1", "view": "left", "point": [412, 760],
 "start_views": {"front": "<sha256>", "right": "...", "back": "...", "left": "..."},
 "status": "reached", "reason": null, "start_event_index": 0, "end_event_index": 56,
 "executor": "point_expert", "turn_index": 3}
```

The per-step RGB, poses and commands of a leg are the sim-engine session frames between the two
event indices. The bundle includes them, so nav-data never calls sim-engine. Records are created once
and never rewritten. `training_admission` is `pending_review` for `purpose=collection`.

## 5. System prompt (`point_v1`)

```
You are the planner for an indoor robot. Follow the navigation instruction below and stop at its end.

Instruction:
{instruction}

The robot has four cameras (front, right, back, left). You move it by sending a walking
controller to a spot on the floor that you can see.

Tools:
- observe(views): photos from the chosen cameras at the current position.
- delegate(view, point, instruction): point at a floor spot in one of the latest photos, as [x, y]
  from 0 to 1000 with the origin at the top left. The controller walks there and returns new photos.
  `instruction` says what this leg does, e.g. "walk to the doorway on the left".
- return_to(leg_id): walk back along your own path to where an earlier leg ended.
- update_plan(steps): keep a short checklist of the route.
- stop(status, reason): end the episode; complete at the destination, abort if you cannot go on.

How to work:
- Look around first and write a plan from the instruction.
- Pick legs of a few metres toward the next landmark. Choose open floor where the robot could
  stand; a spot on a wall, furniture or another level is rejected and the robot does not move.
- After each leg, compare the new photos with the instruction before choosing the next leg.
- If you took a wrong turn, return_to an earlier leg and try another way.
- You succeed only if you stop within about 3 metres of the place the instruction describes.

Budget: {max_turns} responses, {max_actions} control steps.
```

The first user message is `Begin: observe all four views, then write a plan.`

## Decisions (2026-11-23)

1. **Point snapping:** the nearest safe-floor pixel within 2% of the image width is accepted; the
   pixel and world adjustment are recorded in the leg log.
2. **Other floors:** a leg may end on a level other than the start when the NavMesh connects it; the
   same-floor clause of `validate_floor_target` is not applied to session legs.
3. **Camera:** Planner views and VLA input share the session camera profile.
4. **Leg cap:** `leg_max_actions` defaults to 200.
5. **Conversations:** migrated to `AgentLoop` in phase 3 with unchanged behavior.

Pending: whether a response without a tool call gets one repair prompt (§2) instead of failing
immediately as the current invariant requires.

## Tool set `vla_v1` and the LightNav actor

`vla_v1` pairs with `{"kind": "vla", "url": ACTOR}` (the run spec rejects other pairings).
`delegate(instruction)` carries language only; `observe`, `return_to`, `update_plan` and `stop`
are unchanged. The leg ends as `reached` (the policy's STOP or the Habitat low-speed rule),
`leg_budget`, `episode_budget`, `decode_error`, `command_rejected` or `actor_error`; the last
also fails the episode after the leg so far is recorded.

- Actor service (`open_nav_harness/actors/lightnav.py`, model in `lightnav_engine.py`, run in the
  LightNav environment): `POST /v1/sessions`, `POST .../frames {first_index, frames[b64 PNG]}`,
  `POST .../act {request_id, instruction}`, `DELETE ...`, `GET /v1/health`. Frame `i` is the
  simulator's event `i`; appends are contiguous and idempotent; history is the whole episode.
  It receives policy RGB and the instruction only.
- `VlaActor` appends every frame the actor has not seen before each `act`, so retraces stay in
  the history. A velocity step is never retried; a lost reply is `MotionUncertain`.
- The harness numbers legs. `return_to(n)` becomes `retrace {event_index: end of leg n}`, which
  needs a simulator that lists `event_index` under `capabilities.retrace_targets`.
- Control period: LightNav's official Habitat configuration uses 0.1 s steps (lin 0 to 2.5 m/s,
  ang +-300 deg/s). Preflight refuses an actor whose `control.dt_s` differs from the dataset's
  `physics.velocity.time_step_s`, or whose camera differs from the simulator camera.
