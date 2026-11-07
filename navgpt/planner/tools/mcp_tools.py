"""MCP transport entry point; all tool logic lives in NavigationSession."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from navgpt.planner.tools.session import NavigationSession


def build_mcp(session):
    """Expose the same registry to SDK clients; no navigation rules live here."""
    import functools
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult

    server = FastMCP("env")
    for name, definition in session.tools.definitions.items():

        def bind(tool_name):
            @functools.wraps(session.tools.functions[tool_name])
            async def call(*args, **kwargs):
                result = await session.tools.call_tool(tool_name, kwargs)
                return CallToolResult.model_validate(result.model_dump())

            return call

        server.tool(name=name, description=definition.description)(bind(name))
    return server


if __name__ == "__main__":
    build_mcp(NavigationSession()).run()
