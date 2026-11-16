import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from open_nav_harness.platform.database import Database
from open_nav_harness.platform.conversations import Conversations, CreateConversation, Message


@pytest.mark.asyncio
async def test_dialogue_history_and_idempotency():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = root / "config.json"
        config.write_text(
            '{"profiles":{"test":{"model":"test","token_count_mode":"estimate","context_policy":"full"}},"default_model":"test"}'
        )
        db = Database(root / "state")
        db.migrate()
        manager = Conversations(db, config)
        cid = manager.create(CreateConversation())["id"]
        seen = []

        async def generate(self, messages, tools, journal):
            seen.append(messages)
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "你好", "tool_calls": []},
                        "finish_reason": "stop",
                    }
                ]
            }

        with patch("open_nav_harness.providers.compatible.CompatibleClient.generate", generate):
            manager.submit(cid, Message(request_id="1", content="你好"))
            manager.submit(cid, Message(request_id="1", content="你好"))
            for _ in range(100):
                if any(e["kind"] == "assistant_text" for e in manager.events(cid)):
                    break
                await asyncio.sleep(0.01)
            assert len(seen) == 1, manager.events(cid)
            manager.submit(cid, Message(request_id="2", content="继续"))
            for _ in range(100):
                if len(seen) == 2:
                    break
                await asyncio.sleep(0.01)
            assert len(seen) == 2
            assert any(m.get("content") == "你好" and m["role"] == "assistant" for m in seen[-1])
            assert len([e for e in manager.events(cid) if e["kind"] == "user_text"]) == 2
            with pytest.raises(ValueError):
                manager.submit(cid, Message(request_id="1", content="不同"))
            await manager.stop(cid)
            assert manager.get(cid)["status"] == "idle"
        await manager.close()
        db.engine.dispose()


@pytest.mark.asyncio
async def test_mcp_and_navigation_composition_and_blocking():
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from navgpt.planner.tools.registry import ToolDefinition, ToolResult, TextContent
    from open_nav_harness.platform.conversations import CombinedTools, DialogueSettings, DialogueSink

    calls = []

    class Tools:
        async def list_tools(self):
            return [ToolDefinition(name="move", description="fixture", inputSchema={"type": "object"})]

        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return ToolResult(content=[TextContent(text="done")])

    @asynccontextmanager
    async def nav(env):
        yield Tools()

    with tempfile.TemporaryDirectory() as tmp:
        settings = DialogueSettings(profiles={}, default_model="test", navigation={"sim": {}})
        tools = CombinedTools(settings, CreateConversation(navigation="sim"), Path(tmp))
        with patch("open_nav_harness.platform.conversations.open_tools", nav):
            await tools.start()
            assert [t.name for t in await tools.list_tools()] == ["move"]
            assert not (await tools.call_tool("move", {})).isError
            tools.blocked = True
            assert (await tools.call_tool("move", {})).isError
            assert len(calls) == 1
            assert DialogueSink(None, None).skipped().isError
            await tools.close()


@pytest.mark.asyncio
async def test_new_message_supersedes_undispatched_actions():
    import json
    from navgpt.planner.tools.registry import ToolDefinition, ToolResult, TextContent

    entered, release = asyncio.Event(), asyncio.Event()
    executed, generated = [], []

    class Tools:
        def __init__(self, *args):
            self.connection_errors = {}
            self.blocked = False

        async def start(self):
            pass

        async def list_tools(self):
            return [ToolDefinition(name="move", description="fixture", inputSchema={"type": "object"})]

        async def call_tool(self, name, arguments):
            executed.append(name)
            entered.set()
            await release.wait()
            return ToolResult(content=[TextContent(text="action completed")])

        async def close(self):
            pass

        async def stop(self):
            pass

    async def generate(self, messages, tools, journal):
        generated.append(messages)
        calls = (
            [
                {"id": str(i), "type": "function", "function": {"name": "move", "arguments": "{}"}}
                for i in (1, 2)
            ]
            if len(generated) == 1
            else []
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None if calls else "已改口",
                        "tool_calls": calls,
                    },
                    "finish_reason": "tool_calls" if calls else "stop",
                }
            ]
        }

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = root / "config.json"
        config.write_text(
            json.dumps(
                dict(profiles={"test": dict(model="test", token_count_mode="estimate")}, default_model="test")
            )
        )
        db = Database(root / "state")
        db.migrate()
        manager = Conversations(db, config)
        cid = manager.create(CreateConversation())["id"]
        with (
            patch("open_nav_harness.platform.conversations.CombinedTools", Tools),
            patch("open_nav_harness.providers.compatible.CompatibleClient.generate", generate),
        ):
            manager.submit(cid, Message(request_id="1", content="向前"))
            await asyncio.wait_for(entered.wait(), 2)
            manager.submit(cid, Message(request_id="2", content="换一个目标"))
            release.set()
            for _ in range(100):
                if len(generated) == 2:
                    break
                await asyncio.sleep(0.01)
            assert executed == ["move"]
            assert any(m.get("content") == "换一个目标" for m in generated[-1])
            assert any("Not executed" in m.get("content", "") for m in generated[-1] if m["role"] == "tool")
            await manager.stop(cid)
        await manager.close()
        db.engine.dispose()


@pytest.mark.asyncio
async def test_stop_cancels_inference_and_restores_history():
    import json

    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def generate(self, messages, tools, journal):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = root / "config.json"
        config.write_text(
            json.dumps(
                dict(profiles={"test": dict(model="test", token_count_mode="estimate")}, default_model="test")
            )
        )
        db = Database(root / "state")
        db.migrate()
        manager = Conversations(db, config)
        cid = manager.create(CreateConversation())["id"]
        with patch("open_nav_harness.providers.compatible.CompatibleClient.generate", generate):
            manager.submit(cid, Message(request_id="1", content="等待"))
            await asyncio.wait_for(entered.wait(), 2)
            await manager.stop(cid)
            assert cancelled.is_set()
            assert cid not in manager.tasks
            assert manager.get(cid)["status"] == "idle"
            restored = Conversations(db, config)
            assert restored.get(cid)["events"][0]["payload"]["text"] == "等待"
            assert not restored.tasks
        await manager.close()
        db.engine.dispose()


@pytest.mark.asyncio
async def test_namespaced_mcp_coexists_with_navigation():
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from navgpt.planner.tools.registry import ToolDefinition, ToolResult, TextContent
    from open_nav_harness.platform.conversations import CombinedTools, DialogueSettings

    calls = []

    class Nav:
        async def list_tools(self):
            return [ToolDefinition(name="move", description="nav", inputSchema={"type": "object"})]

        async def call_tool(self, name, args):
            calls.append("nav")
            return ToolResult(content=[TextContent(text="nav")])

    class MCP:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def list_tools(self):
            return SimpleNamespace(
                tools=[ToolDefinition(name="move", description="map", inputSchema={"type": "object"})]
            )

        async def call_tool(self, name, args):
            calls.append("mcp")
            return ToolResult(content=[TextContent(text="map")])

    @asynccontextmanager
    async def stream(*args, **kwargs):
        yield (None, None, None)

    @asynccontextmanager
    async def nav(env):
        yield Nav()

    with tempfile.TemporaryDirectory() as tmp:
        settings = DialogueSettings(
            profiles={},
            default_model="test",
            navigation={"sim": {}},
            mcp={"map": dict(url="http://fixture.invalid/mcp")},
        )
        tools = CombinedTools(settings, CreateConversation(navigation="sim", mcp=["map"]), Path(tmp))
        with (
            patch("open_nav_harness.platform.conversations.open_tools", nav),
            patch("mcp.ClientSession", MCP),
            patch("mcp.client.streamable_http.streamablehttp_client", stream),
        ):
            await tools.start()
            assert {t.name for t in await tools.list_tools()} == {"move", "map__move"}
            await tools.call_tool("move", {})
            await tools.call_tool("map__move", {})
            assert calls == ["nav", "mcp"]
            await tools.close()
