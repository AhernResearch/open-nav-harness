"""Direct tools or one RPC per tool invocation, independent of model providers."""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
import uuid
import requests

from .registry import ToolResult
from .session import NavigationSession


class RemoteTools:
    """Lazy episode ownership; read-only MCP preflight never opens an episode."""

    def __init__(self, env):
        self.env = dict(env)
        registry = NavigationSession(env=env).tools
        self.definitions, self.functions = registry.definitions, registry.functions
        self.url = env["NAVGPT_EXECUTION_URL"].rstrip("/")
        self.http = requests.Session()
        self.http.trust_env = False
        self.session_id = None
        self.failed = False

    def request(self, method, path, data=None):
        # No retries. A lost motion response is an uncertain resource, not a retryable inference.
        response = self.http.request(method, self.url + path, json=data, timeout=1800)
        response.raise_for_status()
        return response.json()

    def start(self):
        reply = self.request("POST", "/sessions", {"env": self.env})
        self.session_id = reply["session_id"]
        self.root = reply["root"]
        actual = {t["name"]: t for t in reply["tools"]}
        expected = {k: v.model_dump() for k, v in self.definitions.items()}
        if actual != expected:
            raise RuntimeError("Remote navigation tool schema differs from this frozen implementation")

    def sync(self):
        reply = self.request("GET", "/sessions/" + self.session_id + "/artifacts")
        root = Path(self.env["NAVGPT_LIVE_DIR"])
        labels = (
            Path(self.env["NAVGPT_PLACE_LABELS"])
            if self.env.get("NAVGPT_PLACE_LABELS")
            else root / "labels.json"
        )
        destinations = {"live": root, "labels.json": labels, "calls.jsonl": root / "remote_calls.jsonl"}
        for item in reply["artifacts"]:
            relative = PurePosixPath(item["path"])
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in destinations:
                raise ValueError("Invalid artifact path from worker")
            path = destinations[relative.parts[0]].joinpath(*relative.parts[1:])
            # Never follow a preexisting symlink in the local artifact tree.
            base = destinations[relative.parts[0]] if relative.parts[0] == "live" else path.parent
            if not path.resolve().is_relative_to(base.resolve()) or any(
                p.is_symlink() for p in [path, *path.parents]
            ):
                raise ValueError("Artifact destination escapes its episode directory")
            raw = base64.b64decode(item["data"], validate=True)
            if path.suffix in (".json", ".jsonl"):
                raw = raw.replace((self.root + "/live").encode(), str(root).encode())
                raw = raw.replace((self.root + "/labels.json").encode(), str(labels).encode())
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".sync-tmp")
            temporary.write_bytes(raw)
            temporary.replace(path)
        self.request("POST", "/sessions/" + self.session_id + "/ack", {"cursor": reply["cursor"]})

    def invoke(self, name, arguments):
        try:
            if self.failed:
                raise RuntimeError("Remote tool outcome uncertain; do not continue this episode")
            if self.session_id is None:
                self.start()
            reply = self.request(
                "POST",
                "/sessions/" + self.session_id + "/call",
                {"request_id": uuid.uuid4().hex, "name": name, "arguments": arguments},
            )
            result = ToolResult.model_validate(reply["result"])
            self.sync()
            return result
        except BaseException:
            self.failed = True
            raise

    async def call_tool(self, name, arguments):
        try:
            return await asyncio.to_thread(self.invoke, name, arguments)
        except BaseException:
            self.failed = True
            raise

    async def list_tools(self):
        return list(self.definitions.values())

    def close(self):
        try:
            # A cancelled/in-flight call retains remote ownership until explicit recovery.
            if self.session_id and not self.failed:
                self.sync()
                self.request("POST", "/sessions/" + self.session_id + "/close", {})
        finally:
            if not self.failed:
                self.http.close()


@asynccontextmanager
async def open_tools(env):
    if env.get("NAVGPT_EXECUTION_URL"):
        tools = RemoteTools(env)
        try:
            yield tools
        finally:
            await asyncio.to_thread(tools.close)
    else:
        yield NavigationSession(env=env).tools
