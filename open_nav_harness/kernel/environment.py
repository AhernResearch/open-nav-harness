"""sim-engine session client. Motion is never retried; a lost reply is uncertain."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

import httpx

from .spec import EnvironmentSpec

VIEWS = ("front", "right", "back", "left")


class MotionUncertain(RuntimeError):
    """A motion request may or may not have executed; the episode cannot continue."""


class SimulatorError(RuntimeError):
    """sim-engine refused or failed a non-motion operation."""


class SimulatorBusy(SimulatorError):
    """No free simulator worker (HTTP 429); nothing was opened, so the caller may requeue."""


@dataclass(frozen=True)
class Observation:
    observation_id: str
    event_index: int
    images: dict[str, bytes]  # view -> PNG


class SimEngineEnvironment:
    """One episode on one sim-engine session; owns its CAS event index."""

    def __init__(self, spec: EnvironmentSpec, transport=None, timeout_s: float = 120):
        self.spec = spec
        self.http = httpx.AsyncClient(base_url=spec.url.rstrip("/"), timeout=timeout_s,
                                      trust_env=False, transport=transport)
        self.session_id: str | None = None
        self.event_index = 0
        self.done = False

    async def _call(self, method, path, body=None):
        response = await self.http.request(method, path, json=body)
        if response.status_code == 429:
            raise SimulatorBusy(f"{method} {path}: {response.text[:200]}")
        if response.is_error:
            raise SimulatorError(f"{method} {path}: HTTP {response.status_code} {response.text[:500]}")
        return response

    async def health(self) -> dict:
        return (await self._call("GET", "/v1/health")).json()

    async def open(self, episode_id: str) -> dict:
        reply = (await self._call("POST", "/v1/sessions", {
            "dataset": self.spec.dataset, "episode_id": episode_id,
            "camera": self.spec.camera, "seed": self.spec.seed})).json()
        self.session_id, self.event_index = reply["session_id"], reply["event_index"]
        return reply

    async def observe(self) -> Observation:
        base = f"/v1/sessions/{self.session_id}"
        reply = (await self._call("POST", base + "/observe", {"request_id": uuid.uuid4().hex})).json()
        oid = reply["observation_id"]
        pages = await asyncio.gather(*(self._call("GET", f"{base}/observations/{oid}/{view}")
                                       for view in VIEWS))
        return Observation(oid, reply["event_index"], {v: p.content for v, p in zip(VIEWS, pages)})

    async def command(self, command: dict) -> dict:
        """One step request. Transport loss is uncertain and must not be retried."""
        body = {"request_id": uuid.uuid4().hex, "expected_event_index": self.event_index,
                "command": command}
        try:
            response = await self.http.post(f"/v1/sessions/{self.session_id}/step", json=body)
        except httpx.TransportError as error:
            raise MotionUncertain(f"step reply lost: {type(error).__name__}: {error}") from error
        if response.status_code >= 500:
            raise MotionUncertain(f"step failed on the simulator: HTTP {response.status_code}")
        if response.is_error:
            raise SimulatorError(f"step rejected: HTTP {response.status_code} {response.text[:500]}")
        reply = response.json()
        self.event_index, self.done = reply["event_index"], reply["done"]
        return reply

    async def frame(self, event_index: int) -> tuple[bytes, dict]:
        response = await self._call("GET", f"/v1/sessions/{self.session_id}/frames/{event_index}")
        return response.content, {"width": int(response.headers["X-Width"]),
                                  "height": int(response.headers["X-Height"]),
                                  "sha256": response.headers["X-Content-SHA256"]}

    async def finish(self, reason: str) -> dict:
        return (await self._call("POST", f"/v1/sessions/{self.session_id}/finish",
                                 {"reason": reason})).json()

    async def metrics(self) -> dict:
        return (await self._call("GET", f"/v1/sessions/{self.session_id}/metrics")).json()

    async def close(self):
        try:
            if self.session_id is not None:
                await self._call("DELETE", f"/v1/sessions/{self.session_id}")
        finally:
            await self.http.aclose()
