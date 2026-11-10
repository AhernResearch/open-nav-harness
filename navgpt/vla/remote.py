"""Synchronous actor adapter for an isolated execution worker.

Only the worker's own tool thread waits here. The shared server accepts all
sessions asynchronously; no model or GPU is imported by this adapter.
"""

from __future__ import annotations

import uuid
import httpx

from open_nav_harness.core.version import implementation_digest


class RemoteActor:
    def __init__(self, url, backend):
        self.url, self.backend = url.rstrip("/"), backend
        self.http = httpx.Client(trust_env=False, timeout=300)
        self.session_id = None
        self.cancelled = False

    def request(self, method, path, data=None):
        response = self.http.request(method, self.url + path, json=data)
        response.raise_for_status()
        return response.json()

    def open(self, ident):
        info = self.info()
        if (
            info.get("protocol_version") != 2
            or info.get("backend") != self.backend
            or info.get("implementation_digest") != implementation_digest()
        ):
            raise RuntimeError("Shared inference implementation or backend mismatch")
        self.session_id = ident  # Retain identity even if the open response is lost.
        self.cancelled = False
        self.request("POST", "/sessions", {"session_id": ident, "backend": self.backend})

    def info(self):
        return self.request("GET", "/info")

    def call(self, operation, data=None):
        if self.session_id is None or self.cancelled:
            raise RuntimeError("Actor session is absent or cancelled")
        result = self.request(
            "POST",
            "/sessions/" + self.session_id + "/call",
            {
                "request_id": uuid.uuid4().hex,
                "operation": operation,
                "data": data or {},
            },
        )["result"]
        if self.cancelled:
            raise RuntimeError("Cancelled actor result discarded")
        return result

    def reset(self, data):
        return self.call("reset", data)

    def replace_context(self, data):
        return self.call("replace_context", data)

    def observe(self, data):
        return self.call("observe", data)

    def act(self, data):
        return self.call("act", data)

    def state(self):
        return self.call("state")

    def cancel(self):
        self.cancelled = True
        if self.session_id:
            return self.request("POST", "/sessions/" + self.session_id + "/cancel", {})

    def close(self):
        if self.session_id:
            self.request("POST", "/sessions/" + self.session_id + "/close", {})
            self.session_id = None
