"""One kernel episode from open to bundle, and a lane-bounded batch runner.

An episode is never replayed. A lost motion reply ends it as ``failed_or_uncertain``
with everything recorded so far; the simulator session state is read back, not redone.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path


from .environment import MotionUncertain, SimEngineEnvironment
from .evidence import Evidence
from .executors import EXECUTORS
from .loop import AgentLoop
from .prompt import FIRST, briefing
from .recorder import Recorder
from .spec import KernelSpec
from .tools import EpisodeState

PLANNER_FAILURES = ("planner_ended_without_stop", "planner_protocol_failure")


class PreflightError(RuntimeError):
    pass


async def preflight(spec: KernelSpec, sim_transport=None) -> dict:
    """Once per run: the simulator must offer legs and must not leak training rewards."""
    env = SimEngineEnvironment(spec.environment, transport=sim_transport)
    try:
        health = await env.health()
    finally:
        await env.close()
    protocol, capabilities = health["protocol"], health["capabilities"]
    problems = [message for failed, message in (
        (protocol.get("reward_mode") != "none", "simulator reward_mode must be none"),
        (not {"point_goal", "retrace"} <= set(capabilities.get("commands", [])),
         "simulator lacks point_goal/retrace commands"),
        (not capabilities.get("observe"), "simulator lacks observe"),
        (spec.environment.camera not in protocol.get("cameras", {}), "unknown simulator camera"),
    ) if failed]
    if problems:
        raise PreflightError("; ".join(problems))
    files = json.dumps(protocol.get("implementation_sha256"), sort_keys=True).encode()
    return {"max_actions": protocol["max_actions"], "workers": health["workers"],
            "implementation_sha256": hashlib.sha256(files).hexdigest(),
            "startup_id": health.get("startup_id")}


async def run_episode(spec: KernelSpec, episode_id: str, root: Path, max_actions: int, *,
                      sim_transport=None, planner_transport=None) -> dict:
    """Open failures (including ``SimulatorBusy``) raise before anything is written."""
    env = SimEngineEnvironment(spec.environment, transport=sim_transport)
    started = time.time()
    try:
        opened = await env.open(episode_id)
        recorder = Recorder(root)
        # Written first so an interrupted owner can release the session after a restart.
        recorder.write("session.json", {"url": spec.environment.url, "session_id": env.session_id,
                                        "episode_id": episode_id, "instruction": opened["instruction"]})
    except BaseException:
        await env.close()
        raise
    result: dict = {"episode_id": episode_id, "started": started}
    try:
        recorder.write("spec.json", {"spec": spec.model_dump(exclude={"planner": {"credential_env"}}),
                                     "episode_id": episode_id})
        instruction = opened["instruction"]
        evidence = Evidence(env, recorder, spec.map)
        await evidence.start()
        state = EpisodeState(instruction, env, EXECUTORS[spec.executor.kind], spec.budgets, recorder,
                             evidence)
        loop = AgentLoop(spec, recorder.journal, transport=planner_transport)
        system = briefing(instruction, spec.budgets.max_turns, max_actions)
        try:
            error = await asyncio.wait_for(loop.run(state, system, FIRST), spec.budgets.timeout_s)
        except asyncio.TimeoutError:
            error = "episode_timeout"
        if state.end_reason is None:
            await env.finish("invalid_output" if error in PLANNER_FAILURES else "cancelled")
        evaluation = await env.metrics()
        recorder.write("evaluation.json", evaluation)
        result.update(status="completed", instruction=instruction, error=error,
                      end_reason=state.end_reason, stop=state.stop, turns=loop.turns,
                      malformed_replies=loop.malformed, usage=loop.usage, legs=len(state.legs),
                      actions=env.event_index, elapsed_s=time.time() - started,
                      training_admission="pending_review")
        recorder.write("result.json", result)
    except MotionUncertain as error:
        result.update(status="failed_or_uncertain", error=str(error), event_index=env.event_index,
                      elapsed_s=time.time() - started)
        try:
            result["simulator_state"] = await env.metrics()
        except Exception as read_error:  # the reply is evidence only; never retry motion
            result["simulator_state_error"] = repr(read_error)
        recorder.write("failed_or_uncertain.json", result)
    except Exception as error:
        # Planner or read-only simulator failures: motion state is known, nothing is retried.
        result.update(status="failed", error=f"{type(error).__name__}: {error}",
                      elapsed_s=time.time() - started)
        recorder.write("failure.json", result)
    finally:
        await env.close()
    return result


async def run_batch(spec: KernelSpec, episodes: list[str], output: Path, lanes: int) -> list[dict]:
    """At most ``lanes`` concurrent episodes; an uncertain episode stops its lane."""
    output.mkdir(parents=True, exist_ok=False)
    contract = await preflight(spec)
    queue: asyncio.Queue[str] = asyncio.Queue()
    for episode in episodes:
        queue.put_nowait(episode)
    results: list[dict] = []

    async def lane():
        while not queue.empty():
            episode = queue.get_nowait()
            result = await run_episode(spec, episode, output / episode, contract["max_actions"])
            results.append(result)
            print(episode, result["status"], result.get("end_reason"), result.get("error"), flush=True)
            if result["status"] != "completed":
                return

    await asyncio.gather(*(lane() for _ in range(lanes)))
    return results
