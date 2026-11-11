# Habitat-GS scan development

The 2026-11-10 M-floor probe passed on A100 GPU 2: RGB/depth rendering,
observation without movement, eight shared-tool calls, a 1 m relative move,
1.24 m retrace to the starting position, and STOP at step 13. The baked navmesh
covers about 150.64 m². See the [evidence receipt](evidence/habitat-gs-m-floor-20261110.json)
and [forward view](evidence/habitat-gs-m-floor-forward-20261110.png).
This was a deterministic tool probe without model or actor inference.

The scan backend loads a prepared XGRIDS scan into Habitat-GS, using the existing
Environment HTTP API, `NavigationSession`, movement execution, observed map and
walked-path retracing. It does not use Habitat-GS's separate LLM agent or training
pipelines. The original R2R/RxR backend and benchmark remain separate.

## Runtime and source

Use a separate Python environment and deployment directory. The source commit,
Imath dependency and compatibility patch are recorded in
[`deployment/habitat-gs.lock.json`](../deployment/habitat-gs.lock.json). CUDA is
enabled, Bullet and GUI viewers are disabled. Static scenes do not require torch
or the avatar/SMPL dependencies.

The CUDA architecture patch makes the rasterizer respect
`CMAKE_CUDA_ARCHITECTURES=80` for A100. Upstream hardcodes older architectures that
CUDA 13 no longer compiles. Additional patches use the supported CUDA surface-object
header and query device attributes removed from `cudaDeviceProp`. Preserve the
upstream licenses and record these patches
alongside the source commit.

For an offline machine, transfer the checked-out source **including submodules,
`src/deps/zstd/build` and `data/pbr`**, Imath source, Python wheels, and EGL/OpenGL
development headers, including GLM headers for the CUDA rasterizer. Do not exclude every directory named `build`: zstd contains
source CMake files there. `deployment/build_habitat_gs.sh` prepares a new owned
deployment; `deployment/compile_habitat_gs.sh` runs the isolated build. Both require
the generated identity in the deployment's `owner` file. Run builds through an
owned systemd unit or a terminal that supports large build outputs; the terminal
connector's background runner imposes a file-size limit unsuitable for this build.

## Prepare M-floor

The CPU-only importer selects explicit assets from the archive, rotates positions,
normals and Gaussian quaternions from `(x,y,z)` to `(x,z,-y)`, and preserves scale.
It currently accepts degree-zero, binary little-endian GS PLY. It excludes semantic
annotations and external PCD registration. Default LOD 3 contains about 580,000
Gaussians; LOD 0 selects the full asset.

```bash
uv run python -m navgpt.environment.scan_assets \
  --archive /home/holo/robots/M-floor-exam.zip \
  --output state/habitat-gs/scenes/m-floor-lod3-v1 --lod 3
```

The output directory must be new. `scene.json` hashes the source archive and every
prepared asset. Assets are verified before baking or serving. Never modify a
completed scene package; prepare another version instead.

After baking, `uv run navgpt check-data --config
configs/experiments/scans/m_floor_development.json` verifies the frozen files
without loading CUDA. This check does not certify geometric accuracy.

## Bake and probe

Run from the repository/source-release directory using the Habitat-GS interpreter:

```bash
HGS_PY=/absolute/path/to/independent/habitat-gs/venv/bin/python
"$HGS_PY" -m navgpt.environment.scan_probe bake \
  --manifest state/habitat-gs/scenes/m-floor-lod3-v1/scene.json
"$HGS_PY" -m navgpt.environment.scan_probe probe \
  --manifest state/habitat-gs/scenes/m-floor-lod3-v1/scene.json \
  --output outputs/habitat-gs/probe-001 --gpu 2
```

The bake uses the transformed scan mesh, a 0.18 m radius and 1.5 m agent height.
It freezes all navmesh settings and creates a development start by snapping a
supported recorded scan position. Recorded device poses are coverage diagnostics,
not robot trajectories or evaluation references. Baking creates `navigation/`
once; failed or completed artifacts must be retained when preparing a new version.

The explicit probe checks non-blank RGB, valid native GS depth, observation without
movement, and the actual HTTP/`NavigationSession` tool chain: observation, map,
place annotation, primitive motion, relative movement, retrace and STOP. It uses
no model and cannot start the formal benchmark. Movement submissions are flushed
to `tools.jsonl` before dispatch and are never automatically retried. Each probe
requires a new output directory, preserving uncertain and failed attempts.

## Environment service

```bash
"$HGS_PY" -m navgpt.environment \
  --config configs/experiments/scans/m_floor_development.json --gpu 2
```

Use an ownership-identified managed unit for a persistent service. Keep the
example's loopback binding and choose an unused port. The transport retains the
`r2rce` verb prefix so existing tools and actors use the same protocol. Backend
identity is reported separately as `habitat_gs` in `/health`.

For Planner runs, set `dataset.name=scan`, `dataset.split=development`, and
`environment.backend=habitat_gs`. Preflight permits absent ground truth only when
these settings match the backend's explicit development health contract. Platform
experiments must use `purpose=development`; evaluation and collection are refused.
Add an existing Planner profile and tool set to the example when running a model.

## Reception development task

Create a new task package instead of editing the baked scene's episode file:

```bash
PYTHONPATH=. uv run python deployment/prepare_scan_task.py \
  --source state/habitat-gs/scenes/m-floor-lod3-v1/scene.json \
  --destination state/habitat-gs/scenes/m-floor-reception-v1 \
  --episode-id m-floor-reception-001 \
  --instruction 'Walk forward along the corridor. Find the open entrance on your left with a light-colored reception counter. Stop in the corridor just outside that entrance; do not enter.'
```

The start is the supported corridor position next to the yellow caution barrier,
facing down the corridor: Habitat foot position
`[0.005825, -1.004783, 0.003025]`, xyzw rotation `[0, 0, 0, 1]`.
The task contains only the start and instruction. Visual inspection of the stopping
location is separate from the model input; benchmark metrics remain unavailable.

For a real agentic run, use an isolated managed environment, LightNav inference
instance and execution worker, with a single queue lane. Preserve the actor's
480×270, 0.88 m high, 120° policy camera. Set `panorama_view_px=512` for the
Planner: the standard direction labels occupy most of a 128 px panorama.
Reuse a Planner endpoint through a registered read-only profile; do not adopt or
restart services owned by another deployment. The execution worker and inference
service must have the same implementation digest as the frozen harness source.

Copy the new package's episode file to `<state>/catalog/scan_development.json.gz`.
Register the environment with metadata `dataset=scan`, `split=development`,
`scene_id=m_floor`, `backend=habitat_gs`, `scene_manifest=<absolute path>`, and
`environment_config` containing its rendering settings and step budget. The
Harness composer binds this configuration and the matching execution worker
when it freezes a development experiment.

The [2026-11-10 model validation receipt](evidence/habitat-gs-harness-20261110.json)
records an actual browser-submitted queue run using Qwen3.6-27B-NVFP4,
LightNav-0 and Habitat-GS. With 512 px Planner panoramas it completed nine
model/tool rounds and 58 environment steps in 92.2 seconds, called STOP, closed
the actor session and released its leases. The browser replay also passed.
The earlier 128 px commissioning attempt is retained separately: its Planner
views were obscured by labels and it exhausted its turn budget without STOP.

The integration loop passed, but the reception task did **not** fully pass visual
review. The actor [entered the reception area](evidence/habitat-gs-harness-counter-20261110.png)
to inspect the counter, then the Planner retraced to the
[corridor stopping position](evidence/habitat-gs-harness-stop-20261110.png) and
terminated. This violates “do not enter”; the model's completion claim is not a
task-success measurement. Two collision events were recorded. The task start,
instruction and failed commissioning attempt were preserved, and no navigation
metrics or goal/reference annotations were sent to either model.

The [camera-display follow-up](evidence/habitat-gs-camera-display-20261110.json)
also checks the actual LAN browser replay. Selecting an `observe` event now
updates the camera panel with that event's own images and direction captions,
rather than leaving the previous movement camera or an empty panel visible.
Images open at full size without cropping. Retained low-resolution commissioning
images are identified when enlarged; their covered pixels cannot be recovered
from the recorded image. This display change does not rerun movements or alter
the model's original inputs and completed attempt evidence.

Agentic runs now default to seven tools: `observe`, `observe_map`,
`navigate_to_node`, `terminate_episode`, `update_plan`, `execute_subtask`, and
`update_memory`. Exploration and local corrections use the VLA through
`execute_subtask`; `navigate_to_node` preserves walked-path retracing to visited
nodes. `navigate_relative` and primitive movement are absent from this default.
The API freezes the resolved tool list even when a publisher omits it. Both the
RPC client and execution worker reject unavailable tools before dispatch.
Historical direct-movement ablations require an explicit legacy tool list.

## Record a walkthrough

```bash
"$HGS_PY" deployment/record_habitat_gs.py \
  --manifest state/habitat-gs/scenes/m-floor-lod3-v1/scene.json \
  --output outputs/habitat-gs/walkthrough-001 --gpu 2
```

This records a silent 1920×1080, 30 fps H.264 MP4: walk along a supported
6–10 m corridor route, pause and look to both sides, turn around, and return.
The camera follows the actual robot state and uses the existing environment's
velocity execution and navmesh filtering. A blocked command stops the recording
without retry. This is a scripted scene walkthrough; it does not run a model.

Each output directory must be new. It contains `walkthrough.mp4`, `poster.jpg`,
the frozen recording/source configuration, route, frame-indexed command/pose
journal and completion receipt. Playback uses simulation time; rendering and
encoding may take longer than the video's duration. The Habitat-GS static runtime
includes the `imageio-ffmpeg` encoder, so a system FFmpeg install is unnecessary.

## Validation limits

The current depth source is Habitat-GS's Gaussian renderer, reported as
`gaussian_unvalidated`. A separate mesh-depth renderer is not implemented. Navmesh
generation does not establish that doors, glass, floor levels and robot clearance
are correct. Both geometry and depth remain marked for review.

There are no task goal coordinates, reference paths or scored navigation metrics
in the development episodes. Success, SPL, nDTW and goal distance are `null` and
`evaluation_available=false`; execution telemetry remains available. Scene
annotations must be reviewed before adding a separate evaluation dataset.
