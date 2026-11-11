# LightNav-0 backend for the NavGPT-3 harness

The current owned actor pins upstream revision
`3015508b70fb6e30bbd66701e94c9b916dcb8df8` (see `requirements-lightnav.txt`).
Deployment snapshots its source and Apache license into the release rather than
importing a mutable neighboring checkout. The checkpoint's preprocessing, SlowFast
sampler, unified trajectory prompt, RVQ decoder and greedy vLLM engine are reused.

The default actor input profile is **continuous episode history**. Observations append once
in causal order, including passive retraces. Neither a task switch nor a rolling
Planner window resets or compresses this timeline. Only an episode reset clears it.
A new instruction at the same endpoint runs inference without appending a duplicate
image. The legacy context-replacement endpoint is rejected. Planner tools expose
`execute_subtask(task_id, max_distance_m, instruction="")` for this actor, with no manual reset.
Only the chosen instruction goes into the native prompt; topology captions and
Planner summaries are not appended. NavGPT VLA retains its separate bounded profile.

The agentic Planner writes self-contained local navigation goals: an action,
necessary robot-relative direction, an identifiable destination and a stopping
location when relevant. It keeps related turns/crossings together and grounds
added landmarks in observations. Repairs describe the remaining route from the
current pose, rather than repeatedly asking for "more steps" without a destination.
Original counts, route constraints and endpoint qualifiers remain mandatory; a
difficult ordinal cannot be replaced with an easier goal. These wording guidelines
draw on the upstream [Prompt Guide](https://github.com/lightorigins/LightNav-0#%EF%B8%8F-prompt-guide).
They are Planner guidance, not schema restrictions or changes to LightNav's native
prompt. Existing plans and traces stay frozen. Navigation improvement from this
wording change has not yet been measured.

The [pinned upstream Prompt Guide](https://github.com/lightorigins/LightNav-0/blob/3015508b70fb6e30bbd66701e94c9b916dcb8df8/README.md#%EF%B8%8F-prompt-guide)
recommends action, optional robot-relative direction, a disambiguated object phrase
and an optional stop. Prefer direction plus object or one salient relational anchor;
ordinals are less reliable and need a counting direction. This is wording guidance,
not an exclusive grammar for every supported task. The official Habitat policy also
accepts full R2R/RxR instructions through the VLN trajectory prompt.

For this harness, the Planner now writes concise R2R-style route legs: ordered movement,
landmarks and spatial relations, then one final stopping location. Related corridor
travel, turns and doorway crossings can stay together. An authored wording example
(not a route in the scan) is: “Follow the hallway past the shelves. Turn left through
the open doorway and stop beside the table inside the room.” The
[R2R paper](https://arxiv.org/html/1711.07280v3) describes natural-language routes
typically spanning multiple rooms; it does not establish a universal one-object or
one-metre instruction limit. No benchmark reference paths or held-out instructions
are copied into the Planner's examples.

For open room-finding tasks, the Planner selects branches/candidate rooms and judges
identity. It can delegate a grounded route into a candidate when the user's task
allows entry, then inspect the returned views. Merely saying “find a meeting room”
is still insufficient as a route. Splitting every transition into metre-sized nudges
and repeated stops outside the same door is also discouraged. Resume an unfinished
leg after a runtime handoff while preserving its destination and endpoint qualifier.

The 20-action local-delegation trial exhausted 60 Planner rounds after 159 environment
steps without STOP. The next development configuration uses an 80-action per-call
bound to permit route continuity. This bound is our experimental orchestration choice,
not an upstream recommendation or a change to LightNav's trained prompt, velocity
controller, continuous history or decode contract. Wording and handoff cadence change
together, so a new run cannot isolate their individual effects.

An opt-in harness profile, `harness.vla_instruction_style: r2r`, now requires the
Planner's `execute_subtask` call to provide `instruction` (the route) and `stop_at`
(a concrete final location) separately. The adapter records both fields and the
compiled text, then uses the same navigation execution implementation. LightNav
receives ordinary English: the route followed by “Stop <stop_at>.” Missing fields
are refused before dispatch, and an empty endpoint is refused before movement.
This is our Planner/tool contract, not a requirement in the upstream LightNav prompt
grammar. It checks structure, not whether a landmark or room classification is true.
The default `natural` profile and existing frozen attempts retain their schemas.

The Planner prompt and eight active tool descriptions were subsequently condensed
from 875/566 to 356/323 whitespace-delimited words (using a fixed instruction
placeholder). Tool schemas and the explicit map-read prerequisite are unchanged.
Each camera label now appears once beside its image, with the originating call ID;
structured execution status remains in the tool response. Process images are labelled
as observed frames rather than predicted waypoints. The context window retains only
the latest requested map-memory summary; earlier map images and geometry remain
historical evidence, while full records remain available through `read_memory` and
the audit trace. Current plan/memory state is injected once without repeated prose
guidance. These are text/context changes, not measured navigation gains.

The subsequent [end-to-end development check](evidence/prompt-compact-e2e-20261109.json)
ran episodes 456 and 7006 with the compact prompts and completion cropping, once
each on the same execution lanes. Both succeeded, as in the preceding baseline.
NE changed from 1.7874/0.2670 m to 0.8932/0.3922 m; mean episode time fell from
191.55 to 143.45 seconds and total Planner input tokens from 469,993 to 264,843.
However, episode 456 made seven zero-motion retries versus two previously, and
neither episode wrote memory (previously five/four calls). Prompt compression,
context presentation, generated plans and tool usage changed together; two runs
do not establish a general performance gain. Initial image hashes match the
baseline. The deployed source digest is
`aaf7fb2b8c75211475689d527978a7730672ef28a35005a661a03c73f00382db`.

An opt-in development ablation sets `harness.actor_history_policy` to
`completed_subtask` (default: `episode`; agentic LightNav only). When `update_plan`
marks the last executed task completed, the harness records the current frame as
its completion boundary. On the next different task's execution it resets only
that actor session and replays the contiguous suffix starting at the boundary.
Passive movement after completion is retained. Task switches without completion,
blocked returns and same-task repairs do not crop; reopening/removing the completed
task before the next execution cancels the pending crop. Repeated completion plans
do not move the boundary. A consumed crop does not restore old images when a task
is subsequently reopened. All episode images remain in the Planner evidence store.

This is an intentional departure from official full-episode history, not a rolling
16-frame window. Native segment indices restart at zero over consecutive frames;
original frame IDs remain in traces. `actor_history_crop` records the completed and
next task IDs, boundary, discarded and retained frame counts. `policy_context`
records the available contiguous segment, not the smaller native sampled subset.
Both experimental arms share runtime-neutral Planner/tool wording, with no model
reset parameter; differences from earlier smoke runs therefore require a fresh
paired baseline. Success and runtime changes need to be interpreted alongside
actual crop activations, Planner decisions and inference concurrency.

The first [paired development run](evidence/lightnav-completion-crop-paired-20261109.json)
used routes 0, 456, 7006 and 7648, one rollout per arm, with the same frozen source
and runtime-neutral Planner prompt. Full history succeeded on 2/4 and completion
cropping on 3/4; mean wall time was 268.8 s vs 208.5 s, with 347 vs 261 actor
predictions. Three treatment episodes actually cropped (six boundaries); route 0
never did. Route 7006 failed in both arms. Route 7648 changed from failure to
success, but also received different local instructions. Even route 456's VLA
tokens diverged before cropping under different inference batches, despite matching
recorded images and instructions. These are preliminary end-to-end observations,
not a causal or statistically established benefit. The default remains `episode`.
[Fixed-input boundary replay](evidence/lightnav-completion-crop-boundary-replay-20261109.json)
separately tests the history effect without changing observations or instruction;
it does not measure navigation success or warm-cache performance.

The checkpoint retains the full episode for SlowFast selection; its configured
`num_history_frames=64` is not a 64-frame ring-buffer limit. The generation allowance
is the official Habitat runner's 64 tokens, not the old eight-token cap. The
VLA-only runner uses the last budgeted action for a separately recorded forced STOP,
as the official evaluator does. Planner-assisted termination remains a Planner
choice. Only malformed trajectory decoding falls back to zero velocity, matching
the official Habitat policy; it is logged as `decode_error`, not a successful
arrival. Transport failures and malformed protocol replies remain errors.

`reproduction/verify_lightnav_official.py` is an explicit GPU parity probe: it
compares the official Habitat policy and our owned batched adapter on recorded
images using one shared model. It checks native samples, token outputs, waypoints,
velocity conversion, continuous history, endpoint reuse and concurrent sessions.
It never starts the formal benchmark. Planner-generated subtasks are an experimental
extension; identical internals do not make those runs an official benchmark score.

The owned worker path is now available through `python -m navgpt.vla.worker --config
<worker.json>`. It imports LightNav inference directly with the `vllm_local` backend;
it does not start or connect to the upstream WebSocket server. It shares the same
context/controller implementation below. See [architecture](ARCHITECTURE.md#lightweight-execution-path)
for direct tools, remote tool execution and MCP compatibility. The bridge described
below remains available for parity checks and existing deployments.

This local extension retains the Planner's eight tools, observed map, place memory,
recovery and final-stop authority, and replaces NavGPT VLA with an existing LightNav-0
service. All observations and motion belong to **one NavGPT Environment**. The existing
sim-engine service remains a reference for parity checks; it does not hold a second
robot state for a harness episode.

```text
Planner + MCP tools ──HTTP── NavGPT Environment / Habitat-Sim
          │                  dedicated policy camera + velocity execution
          └──HTTP── LightNav bridge ──WebSocket── existing lightnav-serve
```

## Components and contract

- [Bridge](../navgpt/vla/lightnav.py): stdlib HTTP service with a private upstream
  WebSocket session. It loads no weights and takes no additional inference GPU.
- [Client](../navgpt/planner/tools/vla_client.py): explicit `lightnav` backend. Only
  the unannotated policy RGB, instruction and sequence index reach LightNav. The
  Planner's side views, pose, map and scoring data do not enter that request.
- [Environment](../navgpt/environment/r2r.py): additional 480×270, 120° HFOV,
  0.88 m forward camera and `step_velocity`. Planner observations retain their own
  camera rig. CameraSpec setup supports Habitat-Sim 0.1.7 and 0.3.x; the A100 smoke
  runs used 0.3.1.
- [MCP tools](../navgpt/planner/tools/mcp_tools.py): policy STOP returns control to
  the Planner. It never calls environment STOP implicitly. `terminate_episode`
  still ends the task.

The 10 decoded SE(2) waypoints are preserved. The Habitat reference controller's
first row with nonzero forward or yaw determines a unicycle command: forward/dt
and yaw/dt, bounded to 0–2.5 m/s and ±300°/s. Lateral displacement is not a
holonomic command. The default execution period is 0.1 s; it is **not a time label
attached to the predicted trajectory points**. Habitat integrates the local
velocity and applies its navmesh collision/sliding filter. This path does not
execute NavGPT's second-waypoint TELEPORT conversion or its two-hit arrival rule.

New episodes reset the bridge's private session. Revised instructions and
delegation pauses retain its history. Corrective movement and retracing feed
causal policy-camera observations to LightNav without generation. An endpoint is
held until the next active prediction, preventing a repeated endpoint image; a
capped delegation's endpoint is retained if the Planner moves next. Captured
motion-frame IDs are distinct from the macro tool's charged step count.

Malformed wire replies and failed requests are errors, not an arrival signal. A
failed upstream session requires an explicit reset. `stop=true` and the reference
controller's zero/low-speed condition are recorded as policy arrival. LightNav does
not expose upstream seeded reset, so controlled-study interfaces are rejected;
this backend currently supports `harness.interface: standard`.

## Configuration

Start with [VLA-only](../configs/experiments/lightnav/vla_only_r2r.json) and
[Planner-assisted](../configs/experiments/lightnav/planner_r2r.json). Machine paths
are in [paths.lightnav.example.json](../configs/paths.lightnav.example.json).
Both experiments select only the first three R2R validation indices by default.
The Planner config uses `gpt-5.6-sol`, the account-available model used in this
integration check; change it to a model advertised by your own account as needed.

The new `vla` keys are:

```json
{
  "backend": "lightnav",
  "url": "http://127.0.0.1:18050",
  "policy_url": "ws://127.0.0.1:8050",
  "control_dt_s": 0.1,
  "max_linear_mps": 2.5,
  "max_angular_dps": 300.0
}
```

`vla.url` is the harness bridge, not the upstream WebSocket address. No NavGPT
checkpoint entry is needed. Preflight verifies backend, camera and control
contracts before running the Planner. Existing configs default to `backend:
navgpt`. JSON configs allow the Habitat service to run without adding PyYAML to
its existing environment.

## Run with the existing A100 installation

The isolated checkout is `/data/nav_storage/lightnav/navgpt-harness-lightnav`.
The upstream model remains at `/data/nav_storage/lightnav/checkpoints/LightNav-0`
and is served on `ws://127.0.0.1:8050`. From that checkout, in separate terminals:

```bash
/data/nav_storage/lightnav/envs/sim-engine-031/bin/python -m navgpt.environment \
  --config configs/experiments/lightnav/planner_r2r.json \
  --paths configs/paths.lightnav.example.json --gpu 3

/data/nav_storage/lightnav/venv/bin/python -m navgpt.vla \
  --config configs/experiments/lightnav/planner_r2r.json \
  --paths configs/paths.lightnav.example.json
```

If a separate minimal bridge environment is needed,
[requirements-lightnav-bridge.txt](../requirements-lightnav-bridge.txt) declares
the WebSocket dependency. The existing LightNav environment already has it.

The successful integration used the logged-in **holo Planner**, with A100
services reached over a LAN-bound SSH tunnel. No credentials were copied to A100.
On holo, create the loopback forwards (adjust the interface and SSH host for a
different network):

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -o 'ProxyCommand=/usr/bin/python3 /home/holo/robots/NavGPT-3/reproduction/direct_lan_socket.py --interface wlp0s20f3 %h %p' \
  -L 127.0.0.1:19250:127.0.0.1:19250 \
  -L 127.0.0.1:18050:127.0.0.1:18050 A100
```

The helper binds to the specified physical interface and fails if it cannot use
that interface. It never falls back to the TUN route. It needs ordinary SSH
public-key access; it does not change routes or authentication configuration.

Then run from the holo repository, using its existing Planner environment:

```bash
.envs/planner/bin/python -m navgpt preflight \
  --config configs/experiments/lightnav/planner_r2r.json
.envs/planner/bin/python -m navgpt run \
  --config configs/experiments/lightnav/planner_r2r.json --episodes 0
```

The VLA-only profile can run directly on A100 with its existing Planner Python;
that mode needs no Planner login. Register the environment and actor through
Workbench; use an execution pool when multiple lanes share an inference server.

The integration services were left available as transient user units:

| Host | Unit | Endpoint |
|---|---|---|
| A100 | `navgpt-lightnav-env-20261106` | `127.0.0.1:19250` |
| A100 | `navgpt-lightnav-bridge-20261106` | `127.0.0.1:18050` → policy `8050` |
| holo | `navgpt-lightnav-lan-tunnel-20261106` | local forwards for both endpoints |

These units are not boot-time installations. Stop them with `systemctl --user
stop UNIT` on the corresponding host, or restart them using the commands above.
The A100 environment renders on GPU 3; the bridge shares the already loaded model
on GPU 2. No model or dataset download was performed.

## Verification and artifacts

```bash
.envs/planner/bin/python -m unittest discover -s tests -v
```

Ten contract tests cover velocity conversion, clipping, model/low-speed STOP,
invalid output, upstream session isolation, instruction changes, frame-only
history, no pose/panorama in policy requests, capped-endpoint deduplication,
external-policy settings and the original NavGPT two-hit stopping behavior.

Current CPU contract tests cover corrective control and causal history without
starting model services. Use the explicit parity probe above for GPU checks.
Historical integration reports remain in [evidence](evidence/); scripts and local
paths from the retired November 6 checkout are not shipped here.

The A100 native smoke artifacts are in
`/data/nav_storage/lightnav/runs/navgpt_lightnav_harness_20261106/outputs/lightnav_vla_only_first3`.
The three routes all succeeded, with aggregate SPL 0.9377 and NE 0.8568 m,
matching the prior sim-engine/LightNav three-route reference within rounding.
There were no teleport actions or runtime errors.

The actual Planner-assisted episode is under
`outputs/lightnav_harness_20261106/lightnav_planner_first1` on holo. It called
`navigate_by_instruction`, `observe_map`, then `terminate_episode`, succeeded at
NE 1.0543 m / SPL 1.0, and used 50 velocity steps plus the final STOP. Four Planner
turns took about 57 s. Live artifacts include all 10 predicted waypoints,
velocity commands and raw model output. These few smoke routes establish
integration, not a benchmark score or a quality improvement over the model alone.
