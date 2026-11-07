"""One isolated episode. Never automatically reruns a navigation attempt."""

import asyncio
import json
import os
import signal
import sys
import traceback
from pathlib import Path
from nav_harness.core.artifacts import atomic_json
from .worker import start_ticks


async def execute(path):
    data = json.loads(path.read_text())
    root = path.parent
    atomic_json(root / "process.json", {"pid": os.getpid(), "start_ticks": start_ticks(os.getpid())})
    spec = data["spec"]
    cfgdata = spec["config"]
    cfgdata["dataset"] = {**cfgdata.get("dataset", {}), "episodes": str(data["episode"])}
    cfgdata["name"] = "episode"
    cfgdata["output"] = {"dir": str(root), "save_frames": True}
    config = root / "config.json"
    atomic_json(config, cfgdata)
    from navgpt.settings import load

    settings = load(config)
    cfg = settings.run_config()
    os.environ.update(settings.provider_env())
    from navgpt.planner.config import VLA_ONLY

    if cfg.condition == VLA_ONLY:
        from navgpt.planner.vla_only import run

        await asyncio.to_thread(run, cfg)
        summary = json.loads((root / "episode" / "summary.json").read_text())
        result = summary["episodes"][0]
    else:
        from navgpt.planner.runner import preflight, run_episode
        from navgpt.planner.runtimes.registry import create_runtime

        await asyncio.to_thread(preflight, cfg)
        runtime = create_runtime(cfg)
        runtime.prepare(cfg)
        result = await run_episode(runtime, cfg, data["episode"], root / "episode")
    from .results import read_jsonl

    task_events = read_jsonl(
        root / "episode" / "live" / f"ep{data['episode']:04d}" / "subtasks" / "events.jsonl"
    )
    from nav_harness.core.recovery import resource_state

    trace_path = root / "episode" / "work" / f"ep{data['episode']:04d}" / "trace" / "requests.jsonl"
    wire = trace_path.exists()
    events = read_jsonl(trace_path if wire else root / "episode" / f"episode_{data['episode']}.jsonl")
    result["resource_state"] = resource_state(events, wire=wire)
    result["vla_calls"] = sum(e.get("kind") == "policy_context" for e in task_events) if task_events else None
    result["implementation_digest"] = spec["implementation_digest"]
    result["attempt_id"] = data["id"]
    result["purpose"] = spec["purpose"]
    # Complete execution is not the same as correct behavior or training admission.
    result["training_status"] = "evaluation_only" if spec["purpose"] != "collection" else "pending_review"
    atomic_json(root / "result.json", result)


async def main(path):
    task = asyncio.current_task()
    assert task is not None
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await execute(path)
    except asyncio.CancelledError:
        atomic_json(path.parent / "cancelled.json", {"cancelled": True, "cleanup_completed": True})
        # No claim of safe remote state. Scheduler retains uncertain leases.
        raise
    except Exception as exc:
        atomic_json(
            path.parent / "failure.json",
            {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve()))
