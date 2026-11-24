"""Client of an actor service. Only policy RGB and an instruction cross this boundary."""

from __future__ import annotations

import base64

import httpx

from .environment import SERVICE_LIMITS

FRAMES_PER_REQUEST = 32


class ActorError(RuntimeError):
    """The actor refused or failed. It never moves the robot; a failed session is not reused."""


class ActorClient:
    def __init__(self, url: str, transport=None, timeout_s: float = 300):
        self.http = httpx.AsyncClient(base_url=url.rstrip("/"), timeout=timeout_s, trust_env=False,
                                      transport=transport, limits=SERVICE_LIMITS)
        self.session_id: str | None = None

    async def _call(self, method: str, path: str, body=None) -> dict:
        try:
            response = await self.http.request(method, path, json=body)
        except httpx.HTTPError as error:
            raise ActorError(f"{method} {path}: {type(error).__name__}: {error}") from error
        if response.is_error:
            raise ActorError(f"{method} {path}: HTTP {response.status_code} {response.text[:300]}")
        return response.json()

    async def health(self) -> dict:
        return await self._call("GET", "/v1/health")

    async def open(self) -> str:
        self.session_id = (await self._call("POST", "/v1/sessions"))["session_id"]
        return self.session_id

    async def append(self, first_index: int, pngs: list[bytes]) -> int:
        """Append frames ``first_index`` onward; returns the history length."""
        for start in range(0, len(pngs), FRAMES_PER_REQUEST):
            chunk = pngs[start:start + FRAMES_PER_REQUEST]
            reply = await self._call("POST", f"/v1/sessions/{self.session_id}/frames", {
                "first_index": first_index + start,
                "frames": [base64.b64encode(png).decode() for png in chunk]})
        return reply["history_frames"]

    async def act(self, request_id: str, instruction: str) -> dict:
        return await self._call("POST", f"/v1/sessions/{self.session_id}/act",
                                {"request_id": request_id, "instruction": instruction})

    async def close(self):
        try:
            if self.session_id is not None:
                await self._call("DELETE", f"/v1/sessions/{self.session_id}")
        finally:
            await self.http.aclose()
