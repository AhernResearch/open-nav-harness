"""One contract across direct calls and the optional SDK transport."""

import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from navgpt.planner.tools.registry import Image, ToolRegistry
from navgpt.planner.tools.mcp_tools import build_mcp
from nav_harness.core.artifacts import Journal
from nav_harness.core.context import ContextWindow


async def test_mcp_preserves_registry_schema_and_multimodal_results():
    registry = ToolRegistry()
    calls = []

    def observe(label: str, count: int = 1) -> list:
        calls.append((label, count))
        return [json.dumps({"label": label}), Image(b"png bytes")]

    registry.register("observe", "Get an observation", observe)
    server = build_mcp(SimpleNamespace(tools=registry))
    listed = await server.list_tools()
    direct = (await registry.list_tools())[0]
    assert listed[0].name == direct.name
    for key in ("properties", "required"):
        assert listed[0].inputSchema[key] == direct.inputSchema[key]
    a = await registry.call_tool("observe", {"label": "test"})
    b = await server.call_tool("observe", {"label": "test"})
    assert [(c.type, getattr(c, "text", None), getattr(c, "data", None)) for c in b.content] == [
        (c.type, getattr(c, "text", None), getattr(c, "data", None)) for c in a.content
    ]
    assert calls == [("test", 1), ("test", 1)]
    with pytest.raises(ValidationError):
        await registry.call_tool("observe", {"label": "test", "count": "1"})
    assert len(calls) == 2


def test_local_tool_transport_does_not_import_mcp():
    code = """
import sys, asyncio
from navgpt.planner.tools.transport import open_tools
async def check():
    async with open_tools({'NAVGPT_CONDITION': 'planner_subtask'}) as tools:
        assert len(await tools.list_tools()) == 10
asyncio.run(check())
assert not any(m == 'mcp' or m.startswith('mcp.') for m in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


async def test_pruned_history_is_not_recounted_or_reintroduced(tmp_path):
    window = ContextWindow("system", "task", "bounded", 1500, 128)
    seen = []

    async def count(messages, tools):
        seen.append(messages)
        return 400 + sum(m.get("role") == "tool" for m in messages) * 400

    for i in range(12):
        window.append(
            [{"role": "assistant", "content": str(i)}, {"role": "tool", "content": str(i)}],
            {"summary": str(i)},
        )
    journal = Journal(tmp_path)
    await window.build(count, [], journal)
    assert len(seen) <= 6
    start = window.start
    seen.clear()
    await window.build(count, [], journal)
    assert len(seen) == 1
    assert window.start == start
    window.append([{"role": "user", "content": "next"}], {"summary": "next"})
    await window.build(count, [], journal)
    assert window.start >= start
    assert all(not any(m.get("content") == "0" for m in messages) for messages in seen)
