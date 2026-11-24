import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from open_nav_harness.platform.database import Database
from open_nav_harness.platform.conversations import Conversations, CreateConversation, Message
from test_kernel import FakeSim

TARGET = {"url": "http://sim", "dataset": "r2r_train", "camera": "cam"}


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
async def test_a_bound_dialogue_carries_the_episode_instruction_in_its_prompt():
    from open_nav_harness.platform.conversations import DialogueSettings, CreateConversation, prompt_for

    settings = DialogueSettings(profiles={}, default_model="test", navigation={"sim": TARGET})
    spec = CreateConversation(navigation="sim", episode="898")
    bound = prompt_for(settings, spec, {"instruction": "Walk to the sofa and stop."})
    assert "Walk to the sofa and stop." in bound and "delegate" in bound
    assert "operator may talk to you" in bound
    assert prompt_for(settings, spec, None) != bound


@pytest.mark.asyncio
async def test_a_bound_episode_is_exposed_as_the_kernel_tool_set_and_records_evidence():
    from open_nav_harness.platform.conversations import CombinedTools, DialogueSettings, CreateConversation

    sim = FakeSim()

    with tempfile.TemporaryDirectory() as tmp:
        settings = DialogueSettings(profiles={}, default_model="test", navigation={"sim": TARGET})
        spec = CreateConversation(navigation="sim", episode="898")
        tools = CombinedTools(settings, spec, Path(tmp), transport=httpx.MockTransport(sim))
        await tools.start()
        assert tools.session["instruction"] == "Walk to the sofa and stop."
        names = [tool.name for tool in await tools.list_tools()]
        assert names == ["observe", "delegate", "return_to", "update_plan", "stop"]
        result = await tools.call_tool("observe", {"views": ["front"]})
        assert not result.isError and [c.type for c in result.content] == ["text", "text", "image"]
        refused = await tools.call_tool("delegate", {"view": "back", "point": [500, 800], "instruction": "go"})
        assert refused.isError and "Observe the back view" in refused.content[0].text
        assert sim.steps == []
        # Every binding writes its own evidence folder and never touches another one.
        root = next(Path(tmp, "navigation").iterdir())
        assert {p.name for p in root.iterdir()} >= {"session.json", "spec.json", "trace"}
        records = [json.loads(line) for line in (root / "trace" / "requests.jsonl").read_text().splitlines()]
        rendered = [r for r in records if r["kind"] == "map_rendered"]
        assert rendered and all(len(r["call_id"]) == 32 for r in rendered)
        await tools.close()


@pytest.mark.asyncio
async def test_a_lost_motion_reply_ends_the_binding_and_blocks_further_motion():
    from open_nav_harness.platform.conversations import CombinedTools, DialogueSettings, CreateConversation

    sim = FakeSim(lose_step=True)
    with tempfile.TemporaryDirectory() as tmp:
        settings = DialogueSettings(profiles={}, default_model="test", navigation={"sim": TARGET})
        tools = CombinedTools(settings, CreateConversation(navigation="sim", episode="898"), Path(tmp),
                              transport=httpx.MockTransport(sim))
        await tools.start()
        await tools.call_tool("observe", {"views": ["front"]})
        with pytest.raises(Exception):
            await tools.call_tool("delegate", {"view": "front", "point": [500, 800], "instruction": "go"})
        assert tools.blocked and len(sim.steps) == 1
        assert (await tools.call_tool("observe", {"views": ["front"]})).isError
        root = next(Path(tmp, "navigation").iterdir())
        assert (root / "failed_or_uncertain.json").exists()


@pytest.mark.asyncio
async def test_binding_requires_a_known_environment_and_an_episode():
    from open_nav_harness.platform.conversations import CreateConversation

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = root / "config.json"
        config.write_text(
            '{"profiles":{"test":{"model":"test","token_count_mode":"estimate","context_policy":"full"}},'
            '"default_model":"test","navigation":{"sim":{"url":"http://sim","dataset":"r2r_train",'
            '"camera":"cam"}}}'
        )
        db = Database(root / "state")
        db.migrate()
        manager = Conversations(db, config)
        with pytest.raises(ValueError, match="episode"):
            manager.create(CreateConversation(navigation="sim"))
        with pytest.raises(ValueError, match="Unknown"):
            manager.create(CreateConversation(navigation="other", episode="1"))
        created = manager.create(CreateConversation(navigation="sim", episode="898"))
        assert created["spec"]["episode"] == "898"
        await manager.close()
        db.engine.dispose()


@pytest.mark.asyncio
async def test_new_message_supersedes_undispatched_actions():
    import json
    from open_nav_harness.providers.tools import ToolDefinition, ToolResult, TextContent

    entered, release = asyncio.Event(), asyncio.Event()
    executed, generated = [], []

    class Tools:
        def __init__(self, *args):
            self.connection_errors = {}
            self.blocked = False
            self.session = None

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
    from open_nav_harness.providers.tools import ToolDefinition, ToolResult, TextContent
    from open_nav_harness.platform.conversations import CombinedTools, DialogueSettings

    calls = []

    class Nav:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self, episode):
            return {"instruction": "Walk to the sofa."}

        def definitions(self):
            return [ToolDefinition(name="move", description="nav", inputSchema={"type": "object"})]

        async def call(self, name, args):
            calls.append("nav")
            return ToolResult(content=[TextContent(text="nav")])

        async def close(self):
            pass

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

    with tempfile.TemporaryDirectory() as tmp:
        settings = DialogueSettings(
            profiles={},
            default_model="test",
            navigation={"sim": {"url": "http://sim", "dataset": "r2r_train", "camera": "cam"}},
            mcp={"map": dict(url="http://fixture.invalid/mcp")},
        )
        tools = CombinedTools(settings, CreateConversation(navigation="sim", episode="1", mcp=["map"]), Path(tmp))
        with (
            patch("open_nav_harness.platform.conversations.KernelNavigation", Nav),
            patch("mcp.ClientSession", MCP),
            patch("mcp.client.streamable_http.streamablehttp_client", stream),
        ):
            await tools.start()
            assert {t.name for t in await tools.list_tools()} == {"move", "map__move"}
            await tools.call_tool("move", {})
            await tools.call_tool("map__move", {})
            assert calls == ["nav", "mcp"]
            await tools.close()


@pytest.mark.asyncio
async def test_a_vla_environment_binds_the_vla_tool_set_and_releases_the_actor():
    from open_nav_harness.platform.conversations import (CombinedTools, CreateConversation, DialogueSettings,
                                                         prompt_for)
    from test_vla import FakeModel, actor, forward

    model = FakeModel(forward(), forward(stop=True))
    service, actor_transport = actor(model)
    target = {**TARGET, "tool_set": "vla_v1", "executor": {"kind": "vla", "url": "http://actor"}}
    settings = DialogueSettings(profiles={}, default_model="test", navigation={"vla": target})
    spec = CreateConversation(navigation="vla", episode="898")
    with tempfile.TemporaryDirectory() as tmp:
        tools = CombinedTools(settings, spec, Path(tmp), transport=httpx.MockTransport(FakeSim()),
                              actor_transport=actor_transport)
        await tools.start()
        delegate = next(t for t in await tools.list_tools() if t.name == "delegate")
        assert set(delegate.inputSchema["properties"]) == {"instruction"}
        assert "delegate(instruction)" in prompt_for(settings, spec, tools.session)
        result = await tools.call_tool("delegate", {"instruction": "walk to the sofa"})
        assert not result.isError and '"reached"' in result.content[0].text
        assert model.opened == 1 and len(service.sessions) == 1
        await tools.close()
        assert service.sessions == {}


def test_a_navigation_target_needs_the_executor_its_tool_set_was_written_for():
    from open_nav_harness.core.config import PlannerProfile
    from open_nav_harness.platform.navigation import NavigationTarget

    with pytest.raises(ValueError, match="needs executor vla"):
        NavigationTarget.model_validate({**TARGET, "tool_set": "vla_v1"})
    target = NavigationTarget.model_validate(
        {**TARGET, "tool_set": "vla_v1", "executor": {"kind": "vla", "url": "http://actor"}})
    spec = target.kernel_spec(PlannerProfile(model="m"))
    assert spec.tool_set == "vla_v1" and spec.executor.url == "http://actor"


def test_options_name_the_catalog_a_variant_dataset_reuses():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from open_nav_harness.platform.conversations import DialogueSettings, router

    settings = DialogueSettings(profiles={}, default_model="test", navigation={
        "sim": TARGET, "vla": {**TARGET, "dataset": "r2r_train_vla", "catalog": "r2r_train"}})
    app = FastAPI()
    app.include_router(router(SimpleNamespace(settings=settings)))
    navigation = TestClient(app).get("/api/conversations/options").json()["navigation"]
    assert [(n["dataset"], n["catalog"]) for n in navigation] == [
        ("r2r_train", "r2r_train"), ("r2r_train_vla", "r2r_train")]
