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
`execute_subtask(task_id, instruction="")` for this actor, with no manual reset.
Only the chosen instruction goes into the native prompt; topology captions and
Planner summaries are not appended. NavGPT VLA retains its separate bounded profile.

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
that mode needs no Planner login. `navgpt launch` also understands the bridge and
can create one bridge/session per shard against a shared upstream policy server.

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

The real corrective-control diagnostic can be replayed against an isolated
environment with:

```bash
.envs/planner/bin/python reproduction/probe_lightnav_repair_resume.py \
  --output outputs/lightnav-repair-resume-new.json
```

It caps each delegation at two movements, moves 0.5 m, resumes with a revised
instruction, verifies causal history and active episode status, then explicitly
stops. This is a plumbing diagnostic, not a scored route.

Integration evidence is under
[`reproduction/lightnav_harness_20261106/`](../reproduction/lightnav_harness_20261106/):

- `integration_report.json`: source hashes, identities, native parity and Planner
  results.
- `repair_resume_probe.json`: history grows 2 → 4 → 6 observations and the
  episode ends only on explicit STOP.
- `planner_smoke.json`, `local_paths.json`, `local_preflight.json`: exact local
  Planner setup. `planner_smoke.log` records the run.

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
