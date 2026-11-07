import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from navgpt.planner.tools.registry import ToolResult as CallToolResult, TextContent, ImageContent
from nav_harness.core.config import PlannerProfile
from nav_harness.providers import runtime as module
from nav_harness.providers.runtime import ManagedRuntime, tool_feedback
from navgpt.planner.contracts import EpisodeContext
from navgpt.planner.runner import EventSink


def call(ident, name):
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": "{}"}}


async def run_fake(tmp_path, monkeypatch, responses, names=None):
    names = names or ["observe_panorama", "terminate_episode"]
    executed = []

    class Client:
        def __init__(self, _, journal=None):
            pass

        async def count(self, *_):
            return 200

        async def close(self):
            pass

        async def generate(self, *_):
            return {"choices": [{"message": {"role": "assistant", "tool_calls": responses.pop(0)}}]}

    class Session:
        def __init__(self, *_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def initialize(self):
            pass

        async def list_tools(self):
            return [
                SimpleNamespace(
                    name=name,
                    description=name,
                    inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
                )
                for name in names
            ]

        async def call_tool(self, name, args):
            executed.append(name)
            value = {
                "steps_taken_total": 0,
                "episode_over": name == "terminate_episode",
                "end_reason": "stop_called" if name == "terminate_episode" else "",
            }
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(value))])

    @asynccontextmanager
    async def transport(*_):
        yield Session()

    monkeypatch.setattr(module, "CompatibleClient", Client)
    monkeypatch.setattr(module, "open_tools", transport)
    monkeypatch.setattr(module, "expected_tools", lambda *_: names)
    ctx = EpisodeContext(
        "system",
        "first",
        "test",
        5,
        None,
        {},
        None,
        [],
        {"NAVGPT_CONDITION": "planner_subtask"},
        tmp_path,
        20,
    )
    sink = EventSink(tmp_path / "events.jsonl")
    outcome = await ManagedRuntime(PlannerProfile(model="test")).run(ctx, sink)
    sink.close()
    return executed, outcome


async def test_multi_call_response_counts_once_and_stops(tmp_path, monkeypatch):
    executed, outcome = await run_fake(
        tmp_path, monkeypatch, [[call("1", "observe_panorama"), call("2", "terminate_episode")]]
    )
    assert executed == ["observe_panorama", "terminate_episode"]
    assert outcome.error is None
    assert outcome.turns == 1


async def test_invalid_batch_never_partially_executes(tmp_path, monkeypatch):
    executed, outcome = await run_fake(
        tmp_path,
        monkeypatch,
        [[call("1", "observe_panorama"), call("2", "unknown")], [call("3", "terminate_episode")]],
    )
    assert executed == ["terminate_episode"]
    assert outcome.error is None
    assert outcome.turns == 2


async def test_repeated_id_never_repeats_side_effect(tmp_path, monkeypatch):
    executed, outcome = await run_fake(
        tmp_path, monkeypatch, [[call("1", "observe_panorama")], [call("1", "observe_panorama")]]
    )
    assert executed == ["observe_panorama"]
    assert "Repeated tool call id" in outcome.error


def test_multimodal_tool_feedback_preserves_call_provenance():
    feedback, images, texts = tool_feedback(
        CallToolResult(
            content=[
                TextContent(type="text", text="observation"),
                ImageContent(type="image", data="aGVsbG8=", mimeType="image/png"),
            ]
        ),
        "call-7",
    )
    assert feedback["tool_call_id"] == "call-7"
    assert "call-7" in images[0]["text"]
    assert images[1]["image_url"]["url"] == "data:image/png;base64,aGVsbG8="


async def test_map_and_memory_cannot_be_precomputed_in_one_model_response(tmp_path, monkeypatch):
    executed, outcome = await run_fake(
        tmp_path,
        monkeypatch,
        [
            [call("1", "observe_map"), call("2", "update_memory")],
            [call("3", "observe_map")],
            [call("4", "update_memory")],
            [call("5", "terminate_episode")],
        ],
        names=["observe_map", "update_memory", "terminate_episode"],
    )
    assert executed == ["observe_map", "update_memory", "terminate_episode"]
    assert outcome.error is None
    events = [json.loads(line) for line in (tmp_path / "trace/requests.jsonl").read_text().splitlines()]
    assert any("separate model turns" in str(e) for e in events if e["kind"] == "tool_validation_error")
