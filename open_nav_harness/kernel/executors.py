"""Who walks a delegated leg. The Planner sees the same delegate tool for every executor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .environment import SimEngineEnvironment


@dataclass(frozen=True)
class LegRequest:
    observation_id: str
    view: str
    point: list[int]
    instruction: str
    max_actions: int


class Executor(ABC):
    name: str

    @abstractmethod
    async def run_leg(self, env: SimEngineEnvironment, leg: LegRequest) -> dict:
        """Execute one leg and return the simulator's step reply (with its public ``leg``)."""


class PointExpert(Executor):
    """sim-engine grounds the pixel, plans a clearance path and tracks it."""

    name = "point_expert"

    async def run_leg(self, env, leg):
        return await env.command({"kind": "point_goal", "observation_id": leg.observation_id,
                                  "view": leg.view, "point": leg.point,
                                  "max_actions": leg.max_actions})


EXECUTORS = {executor.name: executor for executor in (PointExpert(),)}
