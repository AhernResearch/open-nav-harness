"""Bounded asynchronous actor sessions backed by one shared model instance."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from open_nav_harness.core.version import implementation_digest


class OpenSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    backend: str


class Operation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1, max_length=128)
    operation: str
    data: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ActorSession:
    actor: Any
    seen: set[str] = field(default_factory=set)
    failed: bool = False
    cancelled: bool = False
    task: asyncio.Task | None = None


class InferenceService:
    def __init__(self, factory, *, max_sessions=8, max_pending=8, journal=None):
        if max_sessions < 1 or max_pending < 1:
            raise ValueError("Service capacities must be positive")
        self.factory = factory
        self.max_sessions, self.max_pending = max_sessions, max_pending
        self.sessions: dict[str, ActorSession] = {}
        self.opening: set[str] = set()
        self.retired: set[str] = set()
        self.pending = 0
        self.completed = 0
        self.journal = Path(journal) if journal else None
        self.stopping = False
        self.digest = implementation_digest()
        self.factory.record = self.record

    def record(self, event):
        if self.journal:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open("a") as out:
                out.write(json.dumps({"time": time.time(), **event}) + "\n")

    def info(self):
        return {
            **getattr(self.factory, "metadata", {}),
            "backend": self.factory.backend,
            "protocol_version": 2,
            "implementation_digest": self.digest,
            "shared_inference": True,
            "max_sessions": self.max_sessions,
            "max_pending": self.max_pending,
            "sessions": len(self.sessions),
            "pending_requests": self.pending,
            "completed_requests": self.completed,
            "accepting": not self.stopping,
            **getattr(self.factory, "stats", lambda: {})(),
        }

    async def open(self, request):
        ident = request.session_id
        if request.backend != self.factory.backend:
            raise HTTPException(400, "Backend mismatch")
        if ident in self.sessions or ident in self.opening or ident in self.retired:
            raise HTTPException(409, "Session ID already used")
        if self.stopping or len(self.sessions) + len(self.opening) >= self.max_sessions:
            raise HTTPException(429, "Session capacity exhausted")
        self.opening.add(ident)
        try:
            actor = await self.factory.run(self.factory.make)
            self.sessions[ident] = ActorSession(actor)
        finally:
            self.opening.remove(ident)
        return {"session_id": ident}

    def session(self, ident):
        if ident not in self.sessions:
            raise HTTPException(404, "Unknown session")
        return self.sessions[ident]

    def cancel(self, ident):
        session = self.session(ident)
        session.cancelled = session.failed = True
        if hasattr(self.factory, "cancel"):
            self.factory.cancel(session.actor)
        self.record({"kind": "cancel", "session_id": ident, "in_flight": session.task is not None})
        return {"cancelled": True, "in_flight": session.task is not None}

    async def close(self, ident):
        session = self.session(ident)
        if session.task is not None:
            raise HTTPException(409, "Operation has not drained")
        session.failed = True  # Exclude concurrent calls while close yields.
        await self.factory.run(session.actor.close)
        del self.sessions[ident]
        self.retired.add(ident)
        return {"closed": True}

    async def call(self, ident, request):
        session = self.session(ident)
        if request.operation not in ("reset", "replace_context", "observe", "act", "state", "info"):
            raise HTTPException(400, "Unknown operation")
        if session.failed or session.task is not None or request.request_id in session.seen:
            raise HTTPException(409, "Failed, busy session or duplicate request; do not replay")
        if self.stopping or self.pending >= self.max_pending:
            raise HTTPException(429, "Inference queue full; state unchanged")
        session.seen.add(request.request_id)
        self.pending += 1
        submitted = time.monotonic()

        async def execute():
            try:

                def invoke():
                    started = time.monotonic()
                    if session.cancelled:
                        raise RuntimeError("Cancelled before execution")
                    method = getattr(session.actor, request.operation)
                    result = method() if request.operation in ("info", "state") else method(request.data)
                    return result, (started - submitted) * 1000, (time.monotonic() - started) * 1000

                result, queue_ms, execution_ms = await self.factory.run(invoke)
                if session.cancelled:
                    raise RuntimeError("Cancelled result discarded")
                if (
                    request.operation == "act"
                    and isinstance(result, list)
                    and result
                    and isinstance(result[0], dict)
                ):
                    result[0]["inference_metrics"] = {
                        **result[0].get("inference_metrics", {}),
                        "session_id": ident,
                        "request_id": request.request_id,
                        "queue_ms": queue_ms,
                        "execution_ms": execution_ms,
                    }
                self.completed += 1
                self.record(
                    {
                        "kind": "result",
                        "session_id": ident,
                        "request_id": request.request_id,
                        "operation": request.operation,
                        "queue_ms": queue_ms,
                        "execution_ms": execution_ms,
                        "inference_metrics": result[0].get("inference_metrics", {})
                        if request.operation == "act"
                        and isinstance(result, list)
                        and result
                        and isinstance(result[0], dict)
                        else {},
                    }
                )
                return result
            except BaseException as exc:
                session.failed = True
                if any(
                    marker in str(exc).lower()
                    for marker in (
                        "cuda error",
                        "cuda out of memory",
                        "enginecore encountered",
                        "engine core failed",
                    )
                ):
                    self.stopping = True
                    for other in self.sessions.values():
                        other.cancelled = other.failed = True
                self.record(
                    {
                        "kind": "error",
                        "session_id": ident,
                        "request_id": request.request_id,
                        "operation": request.operation,
                        "error": str(exc),
                    }
                )
                raise
            finally:
                self.pending -= 1
                session.task = None

        task = asyncio.create_task(execute())
        session.task = task
        # Keep the owner alive until the blocking CUDA operation really completes.
        # HTTP cancellation never makes a still-running session reusable.
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self.cancel(ident)
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            raise

    async def shutdown(self):
        self.stopping = True
        for ident in list(self.sessions):
            self.cancel(ident)
        await asyncio.gather(*(s.task for s in self.sessions.values() if s.task), return_exceptions=True)
        for ident in list(self.sessions):
            await self.close(ident)
        await self.factory.close()


def create_app(service):
    @asynccontextmanager
    async def lifespan(app):
        await service.factory.start()
        try:
            yield
        finally:
            await service.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.get("/info")
    async def info():
        return service.info()

    @app.post("/sessions")
    async def open_session(request: OpenSession):
        return await service.open(request)

    @app.post("/sessions/{ident}/call")
    async def call(ident: str, request: Operation):
        return {"result": await service.call(ident, request)}

    @app.get("/sessions/{ident}")
    async def state(ident: str):
        s = service.session(ident)
        return {"failed": s.failed, "cancelled": s.cancelled, "in_flight": s.task is not None}

    @app.post("/sessions/{ident}/cancel")
    async def cancel(ident: str):
        return service.cancel(ident)

    @app.post("/sessions/{ident}/close")
    async def close(ident: str):
        return await service.close(ident)

    return app


def main():
    import uvicorn
    from .shared import BackendFactory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    config = json.loads(Path(parser.parse_args().config).read_text())
    service = InferenceService(
        BackendFactory(config),
        max_sessions=config.get("max_sessions", 8),
        max_pending=config.get("max_pending", 8),
        journal=config.get("journal"),
    )
    uvicorn.run(create_app(service), host="127.0.0.1", port=config["port"])


if __name__ == "__main__":
    main()
