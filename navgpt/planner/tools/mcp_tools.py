"""MCP transport entry point; all tool logic lives in NavigationSession."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def build_mcp(session):
    """Expose the same registry to SDK clients; no navigation rules live here."""
    import functools
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult

    registry = getattr(session, "tools", session)
    server = FastMCP("env")
    for name, definition in registry.definitions.items():

        def bind(tool_name):
            @functools.wraps(registry.functions[tool_name])
            async def call(*args, **kwargs):
                result = await registry.call_tool(tool_name, kwargs)
                return CallToolResult.model_validate(result.model_dump())

            return call

        server.tool(name=name, description=definition.description)(bind(name))
    return server


async def main():
    import os
    from navgpt.planner.tools.transport import open_tools

    env = {k: v for k, v in os.environ.items() if k.startswith("NAVGPT_")}
    async with open_tools(env) as registry:
        await build_mcp(registry).run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
