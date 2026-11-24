# LightNav actor service

The LightNav-0 policy runs as a separate HTTP service, used by the `vla_v1` tool set through
`VlaActor` (`open_nav_harness/kernel/executors.py`). The control plane never imports the model.

- `open_nav_harness/actors/lightnav.py`: the service contract (FastAPI). It imports neither
  `lightnav` nor `torch`, and is tested with a fake backend.
- `open_nav_harness/actors/lightnav_engine.py`: the model backend and the entry point. It needs
  `lightnav`, torch and vLLM, so it runs only in its own CUDA environment.
- `requirements-lightnav.txt`: the upstream revision, pinned to commit
  `3015508b70fb6e30bbd66701e94c9b916dcb8df8`. Install it in the actor environment, not in the
  control-plane environment.

## Input boundary

The actor receives policy RGB frames and a language instruction, nothing else. It is given no
poses, goals, maps or Planner photos. Frame `i` is the simulator's event `i`. History is the whole
episode: a session never uses a ring buffer and is cleared only when it is deleted. The
checkpoint's `num_history_frames` is the SlowFast sampling configuration, not a buffer limit.

## HTTP contract

| Request | Meaning |
|---|---|
| `GET /v1/health` | `status`, `startup_id`, `model` identity, `source_sha256`, `control`, `camera`, `history` (`full_episode`), `sessions`, `max_sessions` |
| `POST /v1/sessions` | Open a session; 429 when `max_sessions` is reached |
| `POST /v1/sessions/{id}/frames` | `{first_index, frames: [base64 PNG, at most 32]}`. Appends must be contiguous; a repeated index must have identical content (else 409) |
| `POST /v1/sessions/{id}/act` | `{request_id, instruction}` predicts on the latest frame. A repeated `request_id` with the same content returns the first reply; different content is 409 |
| `DELETE /v1/sessions/{id}` | Close; 409 while busy |

An `act` reply holds `waypoints` (10 SE(2) rows), `raw_text`, `timings_ms`, `history_frames` and
the velocity command: `linear_mps`, `angular_rps`, `stop`, `stop_reason`, `decode_error`.
A session that is busy or has failed answers 409 and is never replayed; a failed session must be
reopened. An invalid waypoint shape is 502.

## Control conversion

The first waypoint row with nonzero forward or yaw motion becomes a unicycle command:
forward/`dt_s` clipped to 0 to `max_linear_mps`, yaw/`dt_s` clipped to +-`max_angular_dps`.
Lateral displacement is dropped. `stop_reason` is:

- `model_stop`: the model's own STOP.
- `low_speed`: speed below 0.01 m/s and 0.5 deg/s, the Habitat STOP rule. Both are arrival.
- `decode_error`: the tokens could not be decoded. This is not arrival; the kernel ends the
  leg as `decode_error`.

`dt_s` is the control period of the velocity step, not a time label on the predicted points.
Generation uses the official Habitat runner's 64-token budget (more when the RVQ decoder needs it).

## Configuration and run

`deployment/actor-lightnav.example.json`:

```json
{
  "model_path": "/srv/models/LightNav-0", "host": "127.0.0.1", "port": 18271, "max_sessions": 4,
  "model": {"gpu_memory_utilization": 0.3, "max_num_seqs": 4, "max_batch_size": 4, "max_wait_ms": 8},
  "control": {"dt_s": 0.1, "max_linear_mps": 2.5, "max_angular_dps": 300.0}
}
```

`model_path`, `port` are required; `host` defaults to `127.0.0.1`, `max_sessions` to 8. Run it in
the actor environment:

```bash
python -m open_nav_harness.actors.lightnav_engine --config deployment/actor-lightnav.json
```

Start it through a trusted template (`owned-lightnav` in `deployment/platform.example.json`)
or by hand. The platform's service health probe for kind `vla` requests `/info`, which this service
does not serve (see [OPERATIONS](OPERATIONS.md#services)); check `GET /v1/health` directly. The run spec points at it with
`"executor": {"kind": "vla", "url": ACTOR_URL}` and `"tool_set": "vla_v1"`.

## Preflight and the control period

Once per run, `VlaActor.preflight` refuses the run when:

- the actor is not `ok` or its history is not `full_episode`;
- the simulator dataset's `velocity.time_step_s` differs from the actor's `control.dt_s`;
- the simulator camera differs from the actor's policy camera (480x270, 120 degree HFOV, 0.88 m,
  no pitch).

The identity it returns (`startup_id`, `source_sha256`, model, control) is stored in the run's
contract. A simulator dataset whose time step equals `dt_s` is therefore required. The tests use
`r2r_train_vla`, a variant of `r2r_train` with its own time step, and set `catalog: "r2r_train"` on
the navigation target so the console lists the source dataset's episodes. The dataset itself is
defined in sim-engine, not here.

## Tests

`tests/test_vla.py` covers the service contract, control conversion and `VlaActor` against fakes.
It does not run the model. Whether a deployed actor reproduces upstream outputs has not been checked
in this repository; earlier measurements remain as history in [evidence](evidence/) and are not
benchmark scores.
