"""Explicit development probes; never starts a model or the formal benchmark."""

import argparse
import base64
import json
import math
import os
from pathlib import Path
import time
import threading
import traceback
import uuid

from .scan_assets import asset_path, sha256, source_fingerprint, verify_manifest, write_json


def bake(manifest_path):
    import habitat_sim
    import numpy as np

    path = Path(manifest_path).resolve()
    manifest = verify_manifest(path)
    destination = path.parent / "navigation"
    destination.mkdir(exist_ok=False)
    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = str(asset_path(path.parent, manifest["assets"]["geometry_stage"]["path"]))
    config.create_renderer = False
    config.enable_physics = False
    config.navmesh_settings = None
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    for key, value in {"agent_height": 1.5, "agent_radius": 0.18, "agent_max_climb": 0.2,
                       "agent_max_slope": 45.0, "cell_size": 0.05, "cell_height": 0.025}.items():
        setattr(settings, key, value)
    settings_path = destination / "settings.json"
    settings.write_to_json(str(settings_path))
    frozen = json.loads(settings_path.read_text())
    with habitat_sim.Simulator(habitat_sim.Configuration(config, [habitat_sim.AgentConfiguration()])) as sim:
        if not sim.recompute_navmesh(sim.pathfinder, settings):
            raise RuntimeError("Cannot build a navigable surface from the scan mesh")
        navmesh = destination / "scene.navmesh"
        sim.pathfinder.save_nav_mesh(str(navmesh))
        poses = json.loads(asset_path(path.parent, manifest["assets"]["scan_poses"]["path"]).read_text())["positions"]
        candidates = []
        distances = []
        for position in poses[::10]:
            snapped = np.asarray(sim.pathfinder.snap_point(position), dtype=float)
            if np.isfinite(snapped).all():
                horizontal = math.hypot(snapped[0] - position[0], snapped[2] - position[2])
                height = position[1] - snapped[1]
                distances.append(horizontal)
                if horizontal <= 0.2 and 0.3 <= height <= 2.0:
                    candidates.append(snapped.tolist())
        if not candidates:
            raise RuntimeError("No recorded scan position has a supported development start")
        start = candidates[0]
        episodes = {"purpose": "development", "episodes": [
            {"episode_id": "scan-dev-0", "scene_id": manifest["scene_id"],
             "start_position": start, "start_rotation": [0, 0, 0, 1],
             "instruction": {"instruction_text": "Observe the surroundings, explore a short clear section of the corridor, then stop."}}]}
        write_json(destination / "episodes.json", episodes)
        receipt = {"schema": "open_nav_scan_navigation_v1", "source_manifest_sha256": sha256(path),
                   "owner": uuid.uuid4().hex, "purpose": "development", "geometry_review": "required",
                   "settings": frozen, "habitat_version": habitat_sim.__version__,
                   "navigable_area_m2": float(sim.pathfinder.navigable_area),
                   "scan_positions_sampled": len(poses[::10]), "supported_start_candidates": len(candidates),
                   "horizontal_snap_p90_m": float(np.quantile(distances, 0.9)) if distances else None,
                   "assets": {key: {"path": name, "sha256": sha256(destination / name)}
                              for key, name in (("navmesh", "scene.navmesh"), ("episodes", "episodes.json"))}}
        write_json(destination / "receipt.json", receipt)
    return receipt


def probe(manifest_path, output):
    import habitat_sim
    import numpy as np
    from .scan import ScanEnv

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    owner = uuid.uuid4().hex
    os.environ["OPEN_NAV_HARNESS_OWNER"] = owner
    write_json(output / "ownership.json", {"owner": owner, "purpose": "development", "pid": os.getpid()})
    write_json(output / "source.json", {
        "habitat_version": habitat_sim.__version__, "cuda_enabled": habitat_sim.cuda_enabled,
        "scene_manifest_sha256": sha256(manifest_path), **source_fingerprint(),
    })
    env = None
    started = time.monotonic()
    try:
        env = ScanEnv(manifest_path, config={"rgb_size": 256, "pano_size": 128, "max_steps": 100})
        reset = env.set_episode_by_index(0)
        if "error" in reset:
            raise RuntimeError(reset["error"])
        forward = env.observe()
        (output / "forward.png").write_bytes(base64.b64decode(forward["rgb"]))
        # An observation must leave the actual pose and movement budget unchanged.
        before = env.agent_state()
        panorama = env.observe_pano()
        after = env.agent_state()
        if before != after:
            raise RuntimeError("Observation changed the navigation state")
        # Sample raw frames on the simulator thread to reject blank renderers.
        def diagnostics():
            observations = env._sim.get_sensor_observations()
            rgb = np.asarray(observations["rgb"])[..., :3]
            depth = np.asarray(observations["pano_depth_0"])
            valid = np.isfinite(depth) & (depth > 0)
            if float(rgb.std()) < 3 or float(valid.mean()) < 0.05:
                raise RuntimeError("Blank RGB or invalid depth rendering")
            return {"rgb_std": float(rgb.std()), "depth_valid_fraction": float(valid.mean()),
                    "depth_median_m": float(np.median(depth[valid]))}
        diagnostic = env._submit(diagnostics)
        # Calls go through the actual HTTP handlers and shared NavigationSession.
        # Each movement is submitted once, with evidence flushed before dispatch.
        events = tool_probe(env, output)
        result = {"purpose": "development", "formal_benchmark": False, "owner": owner,
                  "scene_manifest_sha256": sha256(manifest_path), "reset": reset,
                  "health": env.health(), "render": diagnostic, "observation_pose_unchanged": True,
                  "panorama_keys": sorted(panorama), "motion_events": events,
                  "evaluation": env.evaluate(), "elapsed_s": time.monotonic() - started}
        write_json(output / "result.json", result)
        return result
    except Exception as exc:
        write_json(output / "failure.json", {"error": str(exc), "traceback": traceback.format_exc(),
                                             "movement_retry_allowed": False})
        raise
    finally:
        if env is not None:
            env.close()


def tool_probe(env, output):
    from http.server import ThreadingHTTPServer
    from . import api
    from navgpt.planner.tools.session import NavigationSession

    api.ENV = env
    server = ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    session = NavigationSession(env={"NAVGPT_SERVER_URL": "http://127.0.0.1:" + str(server.server_port),
                                     "NAVGPT_CONDITION": "planner_memory", "NAVGPT_STEP_BUDGET": "100",
                                     "NAVGPT_LIVE_DIR": str(output / "live")})
    calls = [("observe_forward", {}), ("observe_panorama", {}), ("observe_map", {}),
             ("annotate_node", {"name": "probe_start", "caption": "Development probe start"}),
             ("navigate_primitive", {"actions": [2, 1, 3]}),
             ("navigate_relative", {"turn_deg": 0.0, "distance_m": 1.0}),
             ("navigate_to_node", {"target": "probe_start"}),
             ("terminate_episode", {"where": "here"})]
    events = []
    try:
        with (output / "tools.jsonl").open("x") as stream:
            for name, arguments in calls:
                stream.write(json.dumps({"tool": name, "arguments": arguments, "status": "submitted"}) + "\n")
                stream.flush()
                result = session.tools.invoke(name, arguments)
                feedback = [item.text for item in result.content if item.type == "text"]
                record = {"tool": name, "is_error": result.isError, "feedback": feedback,
                          "images": sum(item.type == "image" for item in result.content),
                          "agent_state": env.agent_state()}
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                events.append(record)
                if result.isError:
                    raise RuntimeError("Navigation tool failed: " + name)
                if name == "navigate_to_node" and not any('"arrived": true' in text for text in feedback):
                    raise RuntimeError("Development retrace did not arrive; retain the attempt without retry")
        if not env.agent_state()["done"]:
            raise RuntimeError("STOP did not terminate the development episode")
        return events
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bake", "probe"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    os.environ["NAVGPT_GPU_DEVICE_ID"] = str(args.gpu)
    if args.command == "probe" and args.output is None:
        parser.error("probe requires a fresh --output directory")
    result = bake(args.manifest) if args.command == "bake" else probe(args.manifest, args.output)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
