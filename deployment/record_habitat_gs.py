"""Record a scripted scan walkthrough through the existing environment executor.

No model is called. Commands are journalled before execution, never retried, and
all recording outputs go into a fresh directory. Video time uses the fixed
simulation control interval rather than the wall time spent rendering frames.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navgpt.environment.scan_assets import asset_path, sha256, source_fingerprint, write_json  # noqa: E402


def angle_difference(target, current):
    return (target - current + math.pi) % (2 * math.pi) - math.pi


def record(manifest, output, width=1920, height=1080, fps=30):
    import habitat_sim
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image
    from navgpt.environment.scan import ScanEnv

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    owner = os.environ.get("OPEN_NAV_HARNESS_OWNER") or uuid.uuid4().hex
    os.environ["OPEN_NAV_HARNESS_OWNER"] = owner
    dt = 1 / fps
    config = {"rgb_size": 256, "pano_size": 64, "max_steps": 6000,
              "policy_camera": {"width": width, "height": height, "height_m": 1.25, "hfov": 90.0},
              "velocity_control": {"control_dt_s": dt, "max_linear_mps": 0.4, "max_angular_dps": 45.0}}
    write_json(output / "recording.json", {
        "owner": owner, "purpose": "development", "mode": "scripted_walkthrough",
        "model_inference": False, "formal_benchmark": False, "width": width, "height": height,
        "fps": fps, "timebase": "fixed_simulation_dt", "config": config,
        "scene_manifest_sha256": sha256(manifest), "recorder_sha256": sha256(__file__),
        **source_fingerprint(),
    })
    env, writer = None, None
    started = time.monotonic()
    frames, travelled = 0, 0.0
    try:
        env = ScanEnv(manifest, config=config)
        reset = env.set_episode_by_index(0)
        if "error" in reset:
            raise RuntimeError(reset["error"])
        start = env.agent_state()["position"]

        # Pick an already scanned section in the forward corridor and validate
        # its route against this exact navmesh before recording any movement.
        def plan():
            positions = json.loads(asset_path(env.package_root,
                env.manifest["assets"]["scan_poses"]["path"]).read_text())["positions"]
            for pos in positions[::10]:
                if pos[2] > start[2] - 6 or abs(pos[0] - start[0]) > 3:
                    continue
                snapped = np.asarray(env._pathfinder.snap_point(pos))
                if (not np.isfinite(snapped).all()
                        or math.hypot(snapped[0] - pos[0], snapped[2] - pos[2]) > .2):
                    continue
                path = habitat_sim.ShortestPath()
                path.requested_start, path.requested_end = start, snapped
                if env._pathfinder.find_path(path) and 6 <= path.geodesic_distance <= 10:
                    return [list(map(float, p)) for p in path.points]
            raise RuntimeError("No supported corridor route for the walkthrough")

        route = env._submit(plan)
        write_json(output / "route.json", {"role": "scripted_camera_route_not_evaluation_reference",
                                            "waypoints": route})
        writer = imageio_ffmpeg.write_frames(str(output / "walkthrough.mp4"), (width, height),
            fps=fps, codec="libx264", quality=None, macro_block_size=1, pix_fmt_out="yuv420p",
            output_params=["-crf", "20", "-preset", "medium", "-threads", "4",
                           "-movflags", "+faststart", "-metadata", "title=M-floor Habitat-GS walkthrough",
                           "-metadata", "comment=Scripted development walkthrough; no model inference"])
        writer.send(None)
        with (output / "trajectory.jsonl").open("x") as journal:
            def capture(phase):
                nonlocal frames

                def render():
                    obs = env._sim.get_sensor_observations()
                    return np.ascontiguousarray(obs["policy_rgb"][..., :3], dtype=np.uint8)

                rgb = env._submit(render)
                if frames == 0:
                    Image.fromarray(rgb).save(output / "poster.jpg", quality=95)
                writer.send(rgb)
                frames += 1
                if frames % (fps * 5) == 0:
                    print(json.dumps({"phase": phase, "video_seconds": frames / fps,
                                      "distance_m": round(travelled, 2)}), flush=True)

            def command(linear, angular, phase):
                nonlocal travelled
                if frames >= fps * 150:
                    raise RuntimeError("Recording exceeded its bounded duration")
                before = env.agent_state()["position"]
                journal.write(json.dumps({"frame": frames, "phase": phase, "status": "submitted",
                    "linear_mps": linear, "angular_rps": angular, "dt_s": dt}) + "\n")
                journal.flush()
                result = env.step_velocity(float(linear), float(angular), dt)
                journal.write(json.dumps({"frame": frames, "phase": phase, "result": result}) + "\n")
                journal.flush()
                after = result["pose"]["position"]
                moved = math.hypot(after[0] - before[0], after[2] - before[2])
                travelled += moved
                if result["truncated"] or (linear * dt > .0015 and moved < .0001):
                    raise RuntimeError("Motion stopped making progress; recording retained without retry")
                capture(phase)

            def hold(seconds, phase):
                for _ in range(round(seconds * fps)):
                    capture(phase)

            def turn_by(angle, phase):
                # Smooth acceleration/deceleration, bounded to 45 degrees/s.
                count = max(1, math.ceil(abs(angle) / math.radians(45) * 1.875 * fps))
                previous = 0.0
                for i in range(count):
                    t = (i + 1) / count
                    eased = 10*t**3 - 15*t**4 + 6*t**5
                    current = angle * eased
                    command(0, (current - previous) / dt, phase)
                    previous = current

            def follow(points, phase):
                for point in points:
                    while True:
                        state = env.agent_state()
                        dx = point[0] - state["position"][0]
                        dz = point[2] - state["position"][2]
                        distance = math.hypot(dx, dz)
                        if distance <= .04:
                            break
                        error = angle_difference(math.atan2(-dx, -dz), state["heading_rad"])
                        angular = max(-math.radians(45), min(math.radians(45), error * 2.5))
                        linear = min(.4, distance * 1.5) if abs(error) < math.radians(10) else 0.0
                        command(linear, angular, phase)

            hold(1.5, "opening")
            follow(route[1:], "walk_out")
            hold(.75, "arrival")
            turn_by(math.radians(60), "look_left")
            hold(.75, "left_view")
            turn_by(math.radians(-120), "look_right")
            hold(.75, "right_view")
            turn_by(math.radians(60), "face_corridor")
            back = list(reversed(route))[1:]
            state = env.agent_state()
            dx, dz = back[0][0] - state["position"][0], back[0][2] - state["position"][2]
            turn_by(angle_difference(math.atan2(-dx, -dz), state["heading_rad"]), "turn_back")
            follow(back, "return")
            journal.write(json.dumps({"action": "STOP", "status": "submitted"}) + "\n")
            journal.flush()
            stop = env.step(0)
            journal.write(json.dumps({"action": "STOP", "result": stop}) + "\n")
            journal.flush()
            hold(1.5, "ending")
        writer.close()
        writer = None
        result = {"status": "complete", "owner": owner, "frames": frames,
                  "duration_s": frames / fps, "fps": fps, "width": width, "height": height,
                  "distance_m": travelled, "final_state": env.agent_state(),
                  "wall_s": time.monotonic() - started, "video_sha256": sha256(output / "walkthrough.mp4"),
                  "trajectory_sha256": sha256(output / "trajectory.jsonl"),
                  "model_inference": False, "formal_benchmark": False}
        write_json(output / "result.json", result)
        print(json.dumps(result), flush=True)
        return result
    except Exception as exc:
        write_json(output / "failure.json", {"error": str(exc), "traceback": traceback.format_exc(),
                                             "frames": frames, "movement_retry_allowed": False})
        raise
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    os.environ["NAVGPT_GPU_DEVICE_ID"] = str(args.gpu)
    record(args.manifest, args.output)


if __name__ == "__main__":
    main()
