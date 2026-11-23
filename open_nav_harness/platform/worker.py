"""Single durable scheduler; each episode runs in an owned process group."""

import asyncio
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import IO
from sqlalchemy import text
from open_nav_harness.core.artifacts import atomic_json
from open_nav_harness.core.config import PlatformSettings
from .database import Database
from .runs import RunScheduler, RunStore


def start_ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def process_matches(attempt):
    pid = attempt.get("pid")
    if not pid or start_ticks(pid) != attempt.get("start_ticks"):
        return False
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return (
            b"open_nav_harness.platform.episode" in cmd
            and str(Path(attempt["output"]) / "input.json").encode() in cmd
        )
    except OSError:
        return False


class Worker:
    def __init__(self, db: Database, settings: PlatformSettings):
        self.db, self.settings = db, settings
        self.children: dict[str, asyncio.subprocess.Process] = {}
        self.terminating: dict[str, float] = {}
        self.lockfile: IO[str] | None = None
        self.runs = RunScheduler(RunStore(db))

    def lock(self):
        self.lockfile = (self.db.root / "worker.lock").open("a")
        fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def complete(self, a, record):
        if record.get("resource_state") == "unknown":
            with self.db.transaction() as c:
                c.execute(
                    text("UPDATE attempts SET result=:r WHERE id=:id"),
                    {"id": a["id"], "r": json.dumps(record)},
                )
            self.db.block(
                a["id"], "Motion response missing or errored; stop owned motion services and reconcile"
            )
        else:
            self.db.finish(a["id"], "completed", record)

    def reconcile(self):
        for a in self.db.rows("SELECT * FROM attempts WHERE status='running'"):
            root = Path(a["output"])
            receipt = root / "process.json"
            if receipt.exists():
                process = json.loads(receipt.read_text())
                a.update(process)
                with self.db.transaction() as c:
                    c.execute(
                        text("UPDATE attempts SET pid=:p,start_ticks=:s WHERE id=:id"),
                        {"p": a["pid"], "s": a["start_ticks"], "id": a["id"]},
                    )
            if process_matches(a):
                continue
            result = root / "result.json"
            if result.exists():
                self.complete(a, json.loads(result.read_text()))
            else:
                # An unclean death can leave a remote movement in flight. Keep
                # leases until the operator reconciles owned service processes.
                self.db.block(
                    a["id"], "Unclean episode exit; stop owned motion services and reconcile resources"
                )

    async def launch(self, a):
        from open_nav_harness.core.version import implementation_digest

        if a["spec"]["implementation_digest"] != implementation_digest():
            raise ValueError("Implementation changed since experiment freeze; create a new experiment")
        root = Path(a["output"])
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(root / "input.json", a)
        with (root / "worker.log").open("ab") as log:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "open_nav_harness.platform.episode",
                str(root / "input.json"),
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        self.children[a["id"]] = proc
        with self.db.transaction() as c:
            c.execute(
                text("UPDATE attempts SET pid=:p,start_ticks=:s WHERE id=:id"),
                {"p": proc.pid, "s": start_ticks(proc.pid), "id": a["id"]},
            )

    async def tick(self):
        for a in self.db.rows(
            "SELECT a.*,j.cancel_requested FROM attempts a JOIN jobs j ON j.id=a.job_id WHERE a.status='running'"
        ):
            proc = self.children.get(a["id"])
            if proc and proc.returncode is not None:
                self.children.pop(a["id"], None)
            alive = process_matches(a)
            root = Path(a["output"])
            if not alive:
                result = root / "result.json"
                if result.exists():
                    record = json.loads(result.read_text())
                    self.complete(a, record)
                else:
                    self.reconcile()
                continue
            cfg = json.loads((root / "input.json").read_text())["spec"]["config"]
            timeout = cfg.get("planner", {}).get("episode_timeout_s", 1200) + 120
            if a["cancel_requested"] or time.time() - a["created"] > timeout:
                if a["id"] not in self.terminating:
                    os.killpg(a["pid"], signal.SIGTERM)
                    self.terminating[a["id"]] = time.monotonic()
                elif time.monotonic() - self.terminating[a["id"]] > 30:
                    os.killpg(a["pid"], signal.SIGKILL)
                    # Reconciliation deliberately blocks uncertain resources.
        active = len(self.db.rows("SELECT id FROM attempts WHERE status='running'"))
        while active < self.settings.max_workers:
            a = self.db.claim()
            if not a:
                break
            try:
                await self.launch(a)
            except Exception as exc:
                if a["id"] in self.children:
                    self.db.block(a["id"], "Launch state uncertain: " + str(exc))
                else:
                    self.db.finish(a["id"], "failed", error="launch failed: " + str(exc))
            active += 1

    async def run(self):
        self.lock()
        self.reconcile()
        await self.runs.recover()
        while True:
            await self.tick()
            self.runs.tick()
            await asyncio.sleep(1)
