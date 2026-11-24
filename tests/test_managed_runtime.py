"""The dialogue loop: malformed batches never execute, call ids never repeat."""

from types import SimpleNamespace

from open_nav_harness.core.config import PlannerProfile
from open_nav_harness.providers import runtime as module
from open_nav_harness.providers.runtime import ManagedRuntime
from open_nav_harness.providers.tools import TextContent, ToolDefinition, ToolResult

SCHEMA = {"type": "object", "properties": {"distance": {"type": "number"}}, "required": ["distance"],
          "additionalProperties": False}


def call(ident, name, arguments):
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": arguments}}


class Tools:
    def __init__(self):
        self.executed = []

    async def list_tools(self):
        return [ToolDefinition(name="move", description="move", inputSchema=SCHEMA)]

    async def call_tool(self, name, arguments):
        self.executed.append(arguments)
        return ToolResult(content=[TextContent(text="moved")])


class Sink:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload):
        self.events.append((kind, payload))

    def receive(self, window):
        pass

    def pending(self):
        return False


async def converse(tmp_path, monkeypatch, replies):
    class Client:
        def __init__(self, *_, **__):
            pass

        async def count(self, *_):
            return 100

        async def generate(self, *_):
            return {"choices": [{"message": replies.pop(0)}]}

        async def close(self):
            pass

    monkeypatch.setattr(module, "CompatibleClient", Client)
    tools, sink = Tools(), Sink()
    ctx = SimpleNamespace(workdir=tmp_path, briefing="system", first_prompt="", max_turns=5)
    outcome = await ManagedRuntime(PlannerProfile(model="test")).run(ctx, sink, tools)
    return outcome, tools, sink


async def test_a_malformed_call_rejects_its_whole_batch_and_the_model_repairs_it(tmp_path, monkeypatch):
    outcome, tools, sink = await converse(tmp_path, monkeypatch, [
        {"role": "assistant", "tool_calls": [call("1", "move", '{"distance": 1}'), call("2", "move", "{}")]},
        {"role": "assistant", "tool_calls": [call("3", "move", '{"distance": 2}')]},
        {"role": "assistant", "content": "done"},
    ])
    assert outcome.error is None and outcome.turns == 3
    assert tools.executed == [{"distance": 2}]
    rejected = [p for kind, p in sink.events if kind == "tool_result" and p["is_error"]]
    assert [p["tool_use_id"] for p in rejected] == ["1", "2"]


async def test_a_repeated_call_id_ends_the_turn_without_executing(tmp_path, monkeypatch):
    outcome, tools, _ = await converse(tmp_path, monkeypatch, [
        {"role": "assistant", "tool_calls": [call("1", "move", '{"distance": 1}')]},
        {"role": "assistant", "tool_calls": [call("1", "move", '{"distance": 1}')]},
    ])
    assert tools.executed == [{"distance": 1}]
    assert "Repeated tool call id" in outcome.error
