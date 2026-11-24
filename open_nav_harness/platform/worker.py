"""Single durable scheduler of kernel runs; one worker owns a state folder."""

import asyncio
import fcntl
from typing import IO

from .database import Database
from .runs import RunScheduler, RunStore


class Worker:
    def __init__(self, db: Database):
        self.db = db
        self.lockfile: IO[str] | None = None
        self.runs = RunScheduler(RunStore(db))

    def lock(self):
        self.lockfile = (self.db.root / "worker.lock").open("a")
        fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)

    async def run(self):
        self.lock()
        await self.runs.recover()
        while True:
            self.runs.tick()
            await asyncio.sleep(1)
