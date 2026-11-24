"""Evidence the Planner never sees: per-step frames and chassis poses, and the observed map.

Everything here is recorded into the episode bundle and its journal only. Poses, depth and
the map are simulator-side facts for operators and training export; no part of them is
returned to a tool, so none can reach a model request. A failed read stops the episode:
an episode whose evidence is incomplete is not silently kept.
"""

from __future__ import annotations

import asyncio
import io
import math
from dataclasses import dataclass

import httpx
import numpy as np
from PIL import Image

from open_nav_harness.mapping.observed import DepthView, MapSpec, ObservedMap

from .environment import VIEWS, Observation, SimEngineEnvironment, SimulatorError
from .recorder import Recorder


class EvidenceError(RuntimeError):
    """Recording failed; the episode ends instead of continuing with missing evidence."""


def heading(rotation_xyzw) -> float:
    """Habitat yaw: 0 faces -z and turns counter-clockwise about +y."""
    x, y, z, w = rotation_xyzw
    forward_x, forward_z = -2 * (x * z + w * y), -(1 - 2 * (x * x + y * y))
    return math.atan2(-forward_x, -forward_z)


def pose_record(event_index: int, pose: dict) -> dict:
    return {"event_index": event_index, "position": pose["position"],
            "heading_rad": heading(pose["rotation_xyzw"])}


@dataclass(frozen=True)
class Frame:
    index: int
    png: bytes
    meta: dict


class FrameCache:
    """Each step frame is read from the simulator once for every consumer.

    Evidence records every frame; an executor that feeds a model also needs them. A frame is
    dropped after all consumers released it. Poses stay (three floats per step) so an executor
    can report distances without a second read.
    """

    def __init__(self, env: SimEngineEnvironment):
        self.env, self.consumers = env, 1
        self.entries: dict[int, list] = {}  # index -> [Frame, consumers still to release]
        self.positions: dict[int, list[float]] = {}

    def subscribe(self):
        self.consumers += 1

    async def get(self, index: int) -> Frame:
        if index not in self.entries:
            raw, meta = await self.env.frame(index)
            stream = io.BytesIO()
            Image.frombytes("RGB", (meta["width"], meta["height"]), raw).save(stream, format="PNG")
            self.entries.setdefault(index, [Frame(index, stream.getvalue(), meta), self.consumers])
            self.positions[index] = meta["pose"]["position"]
        return self.entries[index][0]

    def release(self, index: int):
        entry = self.entries[index]
        entry[1] -= 1
        if entry[1] <= 0:
            del self.entries[index]


class Evidence:
    def __init__(self, env: SimEngineEnvironment, recorder: Recorder, spec: MapSpec):
        self.env, self.recorder = env, recorder
        self.frames = FrameCache(env)
        self.map = ObservedMap(spec) if spec.enabled else None
        self.route: list[dict] = []
        self.call: str | None = None  # the tool call whose effect is being recorded

    async def start(self):
        await self._guard(self._start())

    async def leg(self, leg: dict):
        await self._guard(self._leg(leg))

    async def observation(self, observation: Observation):
        if self.map is not None:
            await self._guard(self._observation(observation))

    async def _guard(self, work):
        try:
            await work
        except (SimulatorError, httpx.HTTPError, KeyError, ValueError) as error:
            raise EvidenceError(f"recording failed: {type(error).__name__}: {error}") from error

    async def _start(self):
        frame = await self.frames.get(self.env.event_index)
        self.frames.release(frame.index)
        self.route.append(pose_record(frame.index, frame.meta["pose"]))

    async def _leg(self, leg: dict):
        indices = range(leg["start_event_index"] + 1, leg["end_event_index"] + 1)
        if not indices:
            return
        frames, poses = [], []
        for frame in await asyncio.gather(*map(self.frames.get, indices)):
            pose = pose_record(frame.index, frame.meta["pose"])
            frames.append({**pose, "rgb": self.recorder.image(frame.png), "raw_sha256": frame.meta["sha256"]})
            poses.append(pose)
            self.frames.release(frame.index)
        self.route += poses
        self.recorder.write(f"legs/{leg['leg_id']:04d}-frames.json", frames)
        self.recorder.journal.emit("leg_evidence", call_id=self.call, leg_id=leg["leg_id"],
                                   frames=[frame["rgb"] for frame in frames], poses=poses)

    async def _observation(self, observation: Observation):
        assert self.map is not None
        depths = await asyncio.gather(*(self.env.depth(observation.observation_id, view) for view in VIEWS))
        views = [self._view(observation.images[view], depth, meta)
                 for view, (depth, meta) in zip(VIEWS, depths)]
        floor_y = float(self.route[-1]["position"][1])
        png, frame = await asyncio.to_thread(self._render, views, floor_y)
        self.recorder.journal.emit("map_rendered", call_id=self.call, event_index=observation.event_index,
                                   start=self.route[0]["position"],
                                   artifact=self.recorder.image(png), **frame)

    def _render(self, views: list[DepthView], floor_y: float) -> tuple[bytes, dict]:
        assert self.map is not None
        for view in views:
            self.map.add(view, floor_y)
        return self.map.render([pose["position"] for pose in self.route])

    @staticmethod
    def _view(png: bytes, depth: np.ndarray, meta: dict) -> DepthView:
        rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        projection = np.asarray(meta["calibration"]["projections"]["depth"], float)
        height, width = depth.shape
        k = np.array([[projection[0, 0] * width / 2, 0, (width - 1) / 2],
                      [0, projection[1, 1] * height / 2, (height - 1) / 2], [0, 0, 1]])
        if rgb.shape[:2] != depth.shape:
            raise ValueError("observed RGB and depth sizes differ")
        return DepthView(rgb, depth, k, np.asarray(meta["pose"]["position"], float),
                         np.asarray(meta["pose"]["rotation_xyzw"], float))
