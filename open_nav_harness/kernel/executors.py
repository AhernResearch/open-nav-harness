"""Who walks a delegated leg. The Planner sees the same delegate tool for every executor."""

from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .actor import ActorClient, ActorError
from .environment import SimEngineEnvironment, SimulatorError
from .evidence import Evidence
from .spec import ExecutorSpec, VlaSpec


@dataclass(frozen=True)
class LegRequest:
    instruction: str
    max_actions: int
    observation_id: str | None = None  # point legs: the photo, view and pixel to walk to
    view: str | None = None
    point: list[int] | None = None


class LegAborted(RuntimeError):
    """An executor failed after moving; ``reply`` is the leg so far, which must be recorded."""

    def __init__(self, reply: dict, cause: Exception):
        super().__init__(str(cause))
        self.reply, self.cause = reply, cause


class Executor(ABC):
    name: str
    commands: tuple[str, ...] = ()  # simulator commands this executor needs

    @classmethod
    @abstractmethod
    def create(cls, spec: ExecutorSpec, transport=None) -> Executor: ...

    async def preflight(self, protocol: dict, dataset: str, camera: str) -> tuple[list[str], dict]:
        """Problems that make this executor unusable on the simulator, and its identity."""
        return [], {}

    async def start(self, env: SimEngineEnvironment, evidence) -> None:
        """Before the first frame is recorded."""

    @abstractmethod
    async def run_leg(self, env: SimEngineEnvironment, leg: LegRequest) -> dict:
        """Execute one leg and return the simulator's step reply (with its public ``leg``)."""

    async def close(self) -> None:
        """Release episode resources; always called, also after a failure. Must not raise."""


class PointExpert(Executor):
    """sim-engine grounds the pixel, plans a clearance path and tracks it."""

    name = "point_expert"
    commands = ("point_goal",)

    @classmethod
    def create(cls, spec, transport=None):
        return cls()

    async def run_leg(self, env, leg):
        return await env.command({"kind": "point_goal", "observation_id": leg.observation_id,
                                  "view": leg.view, "point": leg.point,
                                  "max_actions": leg.max_actions})


class VlaActor(Executor):
    """A language-conditioned policy walks the leg: act, one velocity step, next frame.

    The actor session holds the whole episode. Every simulator frame is appended in event order
    before an action is requested, so frames of other motion (retraces) are included lazily.
    A velocity step is never retried; a lost reply is ``MotionUncertain``.
    """

    name = "vla"
    commands = ("velocity",)

    def __init__(self, spec: VlaSpec, transport=None):
        self.spec = spec
        self.client = ActorClient(spec.url, transport=transport)
        self.evidence: Evidence
        self.appended = 0  # frames [0, appended) are in the actor session

    @classmethod
    def create(cls, spec, transport=None):
        return cls(spec, transport)

    async def preflight(self, protocol, dataset, camera):
        try:
            health = await self.client.health()
        except ActorError as error:
            return [str(error)], {}
        velocity = protocol["dataset_physics"].get(dataset, {}).get("velocity") or {}
        want = health["camera"]
        have = protocol["cameras"].get(camera, {})
        problems = [message for failed, message in (
            (health["status"] != "ok", "actor service is not ok"),
            (health["history"] != "full_episode", "actor history must be the full episode"),
            (velocity.get("time_step_s") != health["control"]["dt_s"],
             f"simulator time step {velocity.get('time_step_s')} s differs from the actor control "
             f"period {health['control']['dt_s']} s"),
            (any(have.get(key) != value for key, value in want.items()),
             "simulator camera differs from the actor's policy camera"),
        ) if failed]
        return problems, {"url": self.spec.url, "startup_id": health["startup_id"],
                          "source_sha256": health["source_sha256"], "model": health["model"],
                          "control": health["control"]}

    async def start(self, env, evidence):
        self.evidence = evidence
        evidence.frames.subscribe()
        session = await self.client.open()
        evidence.recorder.write("actor.json", {"url": self.spec.url, "session_id": session,
                                               "health": await self.client.health()})
        await self.sync(env.event_index)

    async def sync(self, event_index: int):
        """Append the frames of events ``appended..event_index`` that the actor has not seen."""
        cache = self.evidence.frames
        indices = range(self.appended, event_index + 1)
        if not indices:
            return
        frames = [await cache.get(i) for i in indices]
        await self.client.append(self.appended, [frame.png for frame in frames])
        for index in indices:
            cache.release(index)
        self.appended = event_index + 1

    async def run_leg(self, env, leg):
        start, collisions, status, reason = env.event_index, 0, "leg_budget", None
        reply: dict = {"event_index": env.event_index, "done": False, "termination": None}
        try:
            for _ in range(leg.max_actions):
                await self.sync(env.event_index)
                action = await self.client.act(uuid.uuid4().hex, leg.instruction)
                self.evidence.recorder.journal.emit(
                    "actor_act", call_id=self.evidence.call, event_index=env.event_index,
                    **{key: action[key] for key in (
                        "request_id", "history_frames", "waypoints", "raw_text", "linear_mps",
                        "angular_rps", "stop", "stop_reason", "decode_error", "timings_ms")})
                if action["stop_reason"] == "decode_error":
                    status, reason = "decode_error", "the walking policy produced no usable move"
                    break
                if action["stop"]:
                    status = "reached"
                    break
                try:
                    reply = await env.command({"kind": "velocity", "linear_mps": action["linear_mps"],
                                               "angular_rps": action["angular_rps"]})
                except SimulatorError as error:  # a rejection never moved the robot
                    status, reason = "command_rejected", str(error)
                    break
                collisions += reply["collisions"]
                if reply["done"]:
                    status = "episode_budget"
                    break
            await self.sync(env.event_index)
        except ActorError as error:
            leg_so_far = self.leg(env, start, collisions, "actor_error", str(error), reply)
            raise LegAborted(leg_so_far, error) from error
        return self.leg(env, start, collisions, status, reason, reply)

    def leg(self, env, start, collisions, status, reason, reply) -> dict:
        positions = self.evidence.frames.positions
        path = [positions[i] for i in range(start, env.event_index + 1) if i in positions]
        return {**reply, "event_index": env.event_index, "leg": {
            "leg_id": 0, "kind": "vla", "status": status, "reason": reason,
            "start_event_index": start, "end_event_index": env.event_index,
            "distance_m": round(sum(map(math.dist, path, path[1:])), 3),
            "net_displacement_m": round(math.dist(path[0], path[-1]), 3) if path else 0.0,
            "collision_steps": collisions}}

    async def close(self):
        try:
            await self.client.close()
        except ActorError as error:  # the session is released with the actor; keep the evidence
            self.evidence.recorder.journal.emit("actor_close_error", error=str(error))


EXECUTOR_TYPES: dict[str, type[Executor]] = {cls.name: cls for cls in (PointExpert, VlaActor)}
# Stateless executors are shared; the conversation adapter binds one without a run spec.
EXECUTORS = {"point_expert": PointExpert()}


def build_executor(spec: ExecutorSpec, transport=None) -> Executor:
    return EXECUTOR_TYPES[spec.kind].create(spec, transport)
