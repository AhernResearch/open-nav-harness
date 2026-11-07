"""MCP transport entry point; all tool logic lives in NavigationSession."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from navgpt.planner.tools.session import NavigationSession

if __name__ == "__main__":
    NavigationSession().mcp.run()
