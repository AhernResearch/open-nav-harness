"""Direct tools or one RPC per tool invocation, independent of model providers."""

from __future__ import annotations

import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
import uuid
import httpx

from .registry import ToolResult
from .session import NavigationSession


class RemoteTools:
    """Lazy episode ownership; read-only MCP preflight never opens an episode."""

    def __init__(self, env):
        self.env = dict(env)
        registry = NavigationSession(env=env).tools
        self.definitions, self.functions = registry.definitions, registry.functions
        self.url = env["NAVGPT_EXECUTION_URL"].rstrip("/")
        self.http = httpx.AsyncClient(trust_env=False, timeout=1800)
        self.session_id = None
        self.failed = False

    async def request(self, method, path, data=None):
        # No retries. A lost motion response is an uncertain resource, not a retryable inference.
        response = await self.http.request(method, self.url + path, json=data)
        response.raise_for_status()
        return response.json()

    async def start(self):
        self.session_id = uuid.uuid4().hex
        receipt = Path(self.env["NAVGPT_LIVE_DIR"]) / "execution_session.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"session_id": self.session_id, "execution_url": self.url}))
        reply = await self.request("POST", "/sessions", {"env": self.env, "session_id": self.session_id})
        if reply["session_id"] != self.session_id:
            raise RuntimeError("Execution session identity mismatch")
        self.root = reply["root"]
        actual = {t["name"]: t for t in reply["tools"]}
        expected = {k: v.model_dump() for k, v in self.definitions.items()}
        if actual != expected:
            raise RuntimeError("Remote navigation tool schema differs from this frozen implementation")

    async def sync(self):
        reply = await self.request("GET", "/sessions/" + self.session_id + "/artifacts")
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
        await self.request("POST", "/sessions/" + self.session_id + "/ack", {"cursor": reply["cursor"]})

    async def invoke(self, name, arguments):
        try:
            if self.failed:
                raise RuntimeError("Remote tool outcome uncertain; do not continue this episode")
            if self.session_id is None:
                await self.start()
            reply = await self.request(
                "POST",
                "/sessions/" + self.session_id + "/call",
                {"request_id": uuid.uuid4().hex, "name": name, "arguments": arguments},
            )
            result = ToolResult.model_validate(reply["result"])
            await self.sync()
            return result
        except BaseException:
            self.failed = True
            raise

    async def call_tool(self, name, arguments):
        try:
            return await self.invoke(name, arguments)
        except BaseException:
            self.failed = True
            if self.session_id:
                try:
                    await self.http.post(self.url + "/sessions/" + self.session_id + "/cancel", timeout=5)
                except httpx.HTTPError:
                    pass  # Retain ownership; cancellation acknowledgement is not proof of idle.
            raise

    async def list_tools(self):
        return list(self.definitions.values())

    async def close(self):
        try:
            # A cancelled/in-flight call retains remote ownership until explicit recovery.
            if self.session_id and not self.failed:
                await self.sync()
                await self.request("POST", "/sessions/" + self.session_id + "/close", {})
        finally:
            await self.http.aclose()


@asynccontextmanager
async def open_tools(env):
    if env.get("NAVGPT_EXECUTION_URL"):
        tools = RemoteTools(env)
        try:
            yield tools
        finally:
            await tools.close()
    else:
        yield NavigationSession(env=env).tools
