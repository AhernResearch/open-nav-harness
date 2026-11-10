"""A leased actor and navigation-tool worker. Bind to loopback behind SSH.

Only one episode owns the worker. An exception fences it until explicit close
or process restart. Movement request IDs cannot be replayed. Completed evidence
is persisted before a response is sent, including a response lost in transit.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from navgpt.planner.tools.session import NavigationSession
from navgpt.planner.tools.vla_client import NavGPTVLAClient


class BusyError(RuntimeError):
    pass


class ActorWorker:
    def __init__(self, actor, backend, directory, environment_url):
        self.actor, self.backend = actor, backend
        self.directory = Path(directory).resolve()
        self.environment_url = environment_url
        self.lock = threading.Lock()
        self.session = None
        self.session_id = None
        self.failed = False
        self.cancelled = threading.Event()
        self.seen = set()
        from open_nav_harness.core.version import implementation_digest

        self.implementation_digest = implementation_digest()

    def actor_call(self, method, path, data=None):
        if self.cancelled.is_set():
            raise BusyError("Execution cancelled; retain the environment lease")
        if method == "GET":
            return {"/info": self.actor.info, "/state": self.actor.state}[path]()
        result = {
            "/reset": self.actor.reset,
            "/context": self.actor.replace_context,
            "/act": self.actor.act,
            "/observe": self.actor.observe,
        }[path](data)
        if self.cancelled.is_set():
            raise BusyError("Cancelled inference result discarded")
        return result

    def handle(self, method, path, data=None):
        # Cancellation cannot wait behind the in-flight movement's worker lock.
        if method == "POST" and path == f"/sessions/{self.session_id}/cancel" and self.session_id:
            self.cancelled.set()
            self.failed = True
            from .remote import RemoteActor
            if isinstance(self.actor, RemoteActor):
                self.actor.cancel()
            return {"cancelled": True, "resource_state": "unknown"}
        if not self.lock.acquire(blocking=False):
            raise BusyError("Worker has an in-flight operation; do not retry motion")
        try:
            return self._handle(method, path, data or {})
        finally:
            self.lock.release()

    def _handle(self, method, path, data):
        if method == "GET" and path == "/info":
            return {
                **self.actor.info(),
                "tool_execution": "in_process",
                "protocol_version": 1,
                "implementation_digest": self.implementation_digest,
                "environment_url": self.environment_url,
                "inference_url": self.actor.url if isinstance(getattr(self.actor, "url", None), str) else None,
                "session_id": self.session_id,
                "failed": self.failed,
            }
        if path == "/sessions" and method == "POST":
            if self.session_id is not None:
                raise BusyError("Previous session must be explicitly closed or worker restarted")
            env = data["env"]
            if not isinstance(env, dict) or not all(
                isinstance(k, str) and isinstance(v, str) and k.startswith("NAVGPT_") for k, v in env.items()
            ):
                raise ValueError("Expected navigation session settings")
            if env.get("NAVGPT_VLA_BACKEND") != self.backend:
                raise ValueError("Actor backend mismatch")
            # Deployment owns network and artifact destinations, never model arguments.
            ident = data.get("session_id", uuid.uuid4().hex)
            if not isinstance(ident, str) or len(ident) != 32 or any(c not in "0123456789abcdef" for c in ident):
                raise ValueError("Invalid session identity")
            folder = self.directory / ident
            folder.mkdir(parents=True)
            env = {
                **env,
                "NAVGPT_SERVER_URL": self.environment_url,
                "NAVGPT_LIVE_DIR": str(folder / "live"),
                "NAVGPT_SUBTASK_DIR": str(folder / "live" / "subtasks"),
                "NAVGPT_PLACE_LABELS": str(folder / "labels.json"),
            }
            env.pop("NAVGPT_EXECUTION_URL", None)
            client = NavGPTVLAClient("", backend=self.backend, transport=self.actor_call)
            self.session_id, self.failed, self.seen = ident, False, set()
            self.cancelled.clear()
            try:
                from .remote import RemoteActor
                if isinstance(self.actor, RemoteActor):
                    self.actor.open(ident)
                seed = env.get("NAVGPT_ACTOR_SEED")
                self.actor.reset({"seed": int(seed)} if seed is not None else {})
                self.session = NavigationSession(env=env, actor=client, cancelled=self.cancelled)
            except BaseException:
                self.failed = True
                raise
            self.sent = {}
            return {
                "session_id": ident,
                "root": str(folder),
                "tools": [t.model_dump() for t in self.session.tools.definitions.values()],
            }
        if path.startswith("/sessions/"):
            parts = path.strip("/").split("/")
            if len(parts) != 3 or parts[1] != self.session_id:
                raise ValueError("Unknown or closed session")
            folder = self.directory / self.session_id
            if parts[2] == "close" and method == "POST":
                if self.failed:
                    raise BusyError("Failed execution requires worker restart; retain the lease")
                self.actor.close()
                self.session = None
                self.session_id = None
                return {"closed": True}
            if parts[2] == "artifacts" and method == "GET":
                changed = self.artifacts(folder)
                return {"root": str(folder), "artifacts": changed, "cursor": self.cursor}
            if parts[2] == "ack" and method == "POST":
                if data.get("cursor") != getattr(self, "cursor", None):
                    raise ValueError("Artifact cursor mismatch")
                self.sent.update(self.pending)
                return {"acknowledged": True}
            if parts[2] != "call" or method != "POST":
                raise ValueError("Unknown session operation")
            if self.failed:
                raise BusyError("Session failed; retain the lease and restart owned services")
            ident = data["request_id"]
            if not isinstance(ident, str) or not ident or len(ident) > 128 or ident in self.seen:
                raise ValueError("Invalid or repeated request ID; refusing duplicate execution")
            name, arguments = data["name"], data["arguments"]
            # Validate before recording a dispatch that might have side effects.
            self.session.tools.arguments[name].model_validate(arguments, strict=True)
            self.seen.add(ident)
            receipt = folder / "calls.jsonl"

            def record(value):
                with receipt.open("a") as out:
                    out.write(json.dumps({"request_id": ident, **value}) + "\n")
                    out.flush()
                    os.fsync(out.fileno())

            record({"kind": "dispatch", "name": name, "arguments": arguments})
            try:
                result = self.session.tools.invoke(name, arguments)
            except BaseException as exc:
                self.failed = True
                record({"kind": "error", "error": str(exc)})
                raise
            value = result.model_dump()
            record({"kind": "result", "result": value})
            return {"result": value}
        # Legacy VLA endpoints remain available for rollout parity checks only.
        if self.session_id is not None:
            raise BusyError("An episode owns the actor; use its session tools")
        return self.actor_call(method, path, data)

    def artifacts(self, folder):
        changed = []
        self.pending = {}
        self.cursor = uuid.uuid4().hex
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.name.endswith(".tmp"):
                continue
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            key = str(path.relative_to(folder))
            if self.sent.get(key) != signature:
                changed.append({"path": key, "data": base64.b64encode(path.read_bytes()).decode()})
                self.pending[key] = signature
                # Client acknowledgment commits this cursor through a later request.
        return changed


def serve(worker, host, port):
    class Handler(BaseHTTPRequestHandler):
        def dispatch(self, method):
            try:
                data = None
                if method == "POST":
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 64 * 1024 * 1024:
                        raise ValueError("Invalid request size")
                    data = json.loads(self.rfile.read(size))
                result = worker.handle(method, self.path, data)
                status = 200
            except BusyError as exc:
                result, status = {"error": str(exc)}, 409
            except (ValueError, TypeError, KeyError) as exc:
                result, status = {"error": str(exc)}, 400
            except Exception as exc:
                result, status = {"error": str(exc)}, 500
            body = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.dispatch("GET")

        def do_POST(self):
            self.dispatch("POST")

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        worker.actor.close()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Trusted worker deployment JSON")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    backend = config["backend"]
    if config.get("inference_url"):
        from .remote import RemoteActor
        actor = RemoteActor(config["inference_url"], backend)
    elif backend == "lightnav":
        from .owned import LightNavEngine
        from .lightnav import LightNavBridge

        engine = LightNavEngine(config["model_path"], **config.get("model_options", {}))
        actor = LightNavBridge(inference=engine, **config.get("control", {}))
    elif backend == "navgpt":
        from .owned import NativeEngine
        actor = NativeEngine(config["model_path"], **config.get("model_options", {}))
    else:
        raise ValueError("Unsupported actor backend")
    worker = ActorWorker(actor, backend, config["artifact_dir"], config["environment_url"])
    serve(worker, "127.0.0.1", config["port"])


if __name__ == "__main__":
    main()
