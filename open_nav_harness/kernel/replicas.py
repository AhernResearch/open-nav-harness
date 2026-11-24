"""Planner replica slots: an episode is pinned to one server so its prompt prefix stays cached."""

from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from typing import AsyncIterator

from open_nav_harness.core.config import PlannerProfile


class ReplicaPool:
    """Episodes in flight per (server, model), shared by every run that uses this pool.

    ``pin`` waits for a replica below ``max_concurrency`` and picks the least busy one (ties go
    to the listed order). A failing replica is not skipped: its episodes fail and are not replayed.
    """

    def __init__(self):
        self.busy: Counter[tuple[str, str]] = Counter()
        self.changed = asyncio.Condition()

    def choose(self, profile: PlannerProfile) -> str | None:
        free = [url for url in profile.endpoints()
                if self.busy[url, profile.model] < profile.max_concurrency]
        return min(free, key=lambda url: self.busy[url, profile.model], default=None)

    @asynccontextmanager
    async def pin(self, profile: PlannerProfile) -> AsyncIterator[PlannerProfile]:
        async with self.changed:
            await self.changed.wait_for(lambda: self.choose(profile) is not None)
            url = self.choose(profile)
            assert url is not None
            self.busy[url, profile.model] += 1
        try:
            yield profile.model_copy(update={"base_url": url})
        finally:
            async with self.changed:
                self.busy[url, profile.model] -= 1
                self.changed.notify_all()
