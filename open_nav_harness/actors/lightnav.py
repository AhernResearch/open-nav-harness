"""LightNav actor service: session-isolated episode history over one shared model.

Only policy RGB frames and a language instruction reach this process; it is given no poses,
goals or maps. A caller appends the frames of its episode in order (frame ``i`` is the
simulator's event ``i``) and asks for one action on the latest frame. The model never sees a
ring buffer: the whole episode stays in the session until it is deleted.

This module imports neither lightnav nor torch; ``lightnav_engine`` supplies the model backend.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

WAYPOINTS = 10
FRAMES_PER_REQUEST = 32


@dataclass(frozen=True)
class Control:
    """The Habitat VLN-CE velocity settings the waypoints are converted for."""

    dt_s: float = 0.1
    max_linear_mps: float = 2.5
    max_angular_dps: float = 300.0

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in (self.dt_s, self.max_linear_mps, self.max_angular_dps)):
            raise ValueError("control settings must be positive and finite")


@dataclass(frozen=True)
class Prediction:
    waypoints: list[list[float]]
    stop: bool  # the model's own STOP signal
    raw_text: str
    decode_error: str | None = None  # the model's tokens could not be decoded
    timings_ms: dict[str, float] = field(default_factory=dict)


class Backend(Protocol):
    """The shared model; one handle per actor session."""

    identity: dict[str, Any]
    camera: dict[str, Any]

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def open(self) -> Any: ...
    async def append(self, handle: Any, png: bytes) -> None: ...
    async def predict(self, handle: Any, instruction: str) -> Prediction: ...


def velocity_action(prediction: Prediction, control: Control) -> dict:
    """The official rule: the first row that moves, as a unicycle velocity.

    Lateral motion is dropped for the nonholonomic base. The model's STOP or a velocity below
    the Habitat STOP thresholds is arrival; undecodable tokens are a ``decode_error``, which
    is not arrival (the official policy would send zero velocity).
    """
    waypoints = prediction.waypoints
    if (len(waypoints) != WAYPOINTS or any(len(row) != 3 for row in waypoints)
            or not all(math.isfinite(v) for row in waypoints for v in row)):
        raise ValueError(f"LightNav must return {WAYPOINTS} finite SE(2) waypoints")
    row = next((r for r in waypoints if abs(r[0]) > 1e-6 or abs(r[2]) > 1e-6), waypoints[0])
    linear = min(max(row[0] / control.dt_s, 0.0), control.max_linear_mps)
    angular_dps = min(max(math.degrees(row[2]) / control.dt_s, -control.max_angular_dps),
                      control.max_angular_dps)
    low_speed = abs(linear) < 0.01 and abs(angular_dps) < 0.5
    reason = ("decode_error" if prediction.decode_error else "model_stop" if prediction.stop
              else "low_speed" if low_speed else None)
    return {"linear_mps": linear, "angular_rps": math.radians(angular_dps),
            "stop": reason in ("model_stop", "low_speed"), "stop_reason": reason,
            "decode_error": prediction.decode_error}


class Frames(BaseModel):
    model_config = ConfigDict(extra="forbid")
    first_index: int = Field(ge=0)
    frames: list[str] = Field(min_length=1, max_length=FRAMES_PER_REQUEST)  # base64 PNG


class Act(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1, max_length=128)
    instruction: str = Field(min_length=1, max_length=2000)


@dataclass
class Session:
    handle: Any
    digests: list[str] = field(default_factory=list)  # sha256 of every appended PNG, by index
    replies: dict[str, tuple[str, dict]] = field(default_factory=dict)
    busy: bool = False
    failed: bool = False


class LightNavService:
    def __init__(self, backend: Backend, control: Control | None = None, max_sessions: int = 8):
        self.backend, self.control, self.max_sessions = backend, control or Control(), max_sessions
        self.sessions: dict[str, Session] = {}
        self.startup_id = uuid.uuid4().hex
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def health(self) -> dict:
        return {"status": "ok", "startup_id": self.startup_id, "model": self.backend.identity,
                "source_sha256": self.source_sha256,
                "control": {"dt_s": self.control.dt_s, "max_linear_mps": self.control.max_linear_mps,
                            "max_angular_dps": self.control.max_angular_dps, "waypoints": WAYPOINTS},
                "camera": self.backend.camera, "history": "full_episode",
                "sessions": len(self.sessions), "max_sessions": self.max_sessions}

    def open(self) -> dict:
        if len(self.sessions) >= self.max_sessions:
            raise HTTPException(429, "actor session capacity exhausted")
        ident = uuid.uuid4().hex
        self.sessions[ident] = Session(self.backend.open())
        return {"session_id": ident}

    def session(self, ident: str) -> Session:
        if ident not in self.sessions:
            raise HTTPException(404, "unknown actor session")
        return self.sessions[ident]

    @asynccontextmanager
    async def exclusive(self, ident: str):
        session = self.session(ident)
        if session.failed or session.busy:
            raise HTTPException(409, "actor session is failed or busy; it is never replayed")
        session.busy = True
        try:
            yield session
        except HTTPException:
            raise
        except BaseException:
            session.failed = True  # a half-applied history must not be extended
            raise
        finally:
            session.busy = False

    async def append(self, ident: str, request: Frames) -> dict:
        async with self.exclusive(ident) as session:
            if request.first_index > len(session.digests):
                raise HTTPException(409, f"frames must be contiguous: have {len(session.digests)}, "
                                         f"got index {request.first_index}")
            for offset, text in enumerate(request.frames):
                try:
                    png = base64.b64decode(text, validate=True)
                except binascii.Error as error:
                    raise HTTPException(422, "frame is not base64") from error
                digest, index = hashlib.sha256(png).hexdigest(), request.first_index + offset
                if index < len(session.digests):  # a repeat must be identical
                    if session.digests[index] != digest:
                        raise HTTPException(409, f"frame {index} was already appended with other content")
                    continue
                await self.backend.append(session.handle, png)
                session.digests.append(digest)
            return {"history_frames": len(session.digests)}

    async def act(self, ident: str, request: Act) -> dict:
        async with self.exclusive(ident) as session:
            digest = hashlib.sha256(
                f"{request.instruction}\0{len(session.digests)}".encode()).hexdigest()
            if request.request_id in session.replies:
                previous, reply = session.replies[request.request_id]
                if previous != digest:
                    raise HTTPException(409, "request_id was already used with different content")
                return reply
            if not session.digests:
                raise HTTPException(409, "append at least one frame before acting")
            prediction = await self.backend.predict(session.handle, request.instruction)
            try:
                action = velocity_action(prediction, self.control)
            except ValueError as error:
                raise HTTPException(502, str(error)) from error
            reply = {"request_id": request.request_id, "history_frames": len(session.digests),
                     "waypoints": prediction.waypoints, "raw_text": prediction.raw_text,
                     "timings_ms": prediction.timings_ms, **action}
            session.replies[request.request_id] = digest, reply
            return reply

    async def close(self, ident: str) -> dict:
        session = self.session(ident)
        if session.busy:
            raise HTTPException(409, "actor session is busy")
        del self.sessions[ident]
        return {"closed": True, "history_frames": len(session.digests)}


def create_app(service: LightNavService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        await service.backend.start()
        try:
            yield
        finally:
            await service.backend.stop()

    app = FastAPI(title="LightNav actor", lifespan=lifespan)
    app.get("/v1/health")(service.health)
    app.post("/v1/sessions")(service.open)
    app.post("/v1/sessions/{ident}/frames")(service.append)
    app.post("/v1/sessions/{ident}/act")(service.act)
    app.delete("/v1/sessions/{ident}")(service.close)
    return app

