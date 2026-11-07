"""Tool transport selection. Local agents use direct Python calls by default."""

from contextlib import asynccontextmanager
from .session import NavigationSession


@asynccontextmanager
async def open_tools(env):
    session = NavigationSession(env=env)
    yield session.tools
