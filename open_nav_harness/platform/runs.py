"""Kernel runs: a durable episode queue, the in-process scheduler and the HTTP API.

A run freezes one ``KernelSpec``, the simulator contract and the kernel source digest.
Episodes execute in the worker process as asyncio tasks. An episode holds one simulator
slot and one planner slot for its whole life, so it stays pinned to one planner replica.
Nothing is ever replayed: an interrupted or lost episode ends as ``failed_or_uncertain``
and only its simulator session is released.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import Field, field_validator
from sqlalchemy import text

from open_nav_harness.core.config import StrictModel
from open_nav_harness.core.version import kernel_digest
from open_nav_harness.kernel.environment import SimulatorBusy
from open_nav_harness.kernel.episode import preflight, run_episode
from open_nav_harness.kernel.spec import KernelSpec

from .database import Database, ident
from .kernel_feed import KernelFeed
from .media import artifact_response, image_file
from .stream import follow

TERMINAL = ("completed", "failed", "failed_or_uncertain", "cancelled")
SUMMARY = ("end_reason", "error", "stop", "turns", "malformed_replies", "legs", "actions", "elapsed_s")
BUSY_BACKOFF_S = 5.0


class RunRequest(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    purpose: Literal["evaluation", "collection"] = "evaluation"
    spec: KernelSpec
    episodes: list[str] = Field(min_length=1, max_length=20000)
    lanes: int = Field(default=1, ge=1, le=64)

    @field_validator("episodes")
    @classmethod
    def unique(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("episodes must be unique")
        return value


class ComposeRun(StrictModel):
    """A run chosen by name from trusted settings; clients never supply endpoints."""

    name: str = Field(min_length=1, max_length=200)
    purpose: Literal["evaluation", "collection"] = "evaluation"
    planner: str
    environment: str
    episodes: list[str] = Field(min_length=1, max_length=20000)
    lanes: int = Field(default=1, ge=1, le=64)

    def request(self, settings) -> RunRequest:
        if settings is None or self.planner not in settings.profiles:
            raise ValueError("Unknown planner " + self.planner)
        if self.environment not in settings.navigation:
            raise ValueError("Unknown environment " + self.environment)
        spec = settings.navigation[self.environment].kernel_spec(settings.profiles[self.planner])
        return RunRequest(name=self.name, purpose=self.purpose, spec=spec, episodes=self.episodes,
                          lanes=self.lanes)


def summarize(result: dict, evaluation: dict | None) -> dict:
    summary = {key: result.get(key) for key in SUMMARY}
    summary["metrics"] = (evaluation or {}).get("metrics")
    return summary


class RunStore:
    """Run and episode rows; every state change is one transaction."""

    def __init__(self, db: Database):
        self.db, self.root = db, db.root / "runs"

    def folder(self, rid: str, episode: str) -> Path:
        return self.root / rid / episode

    def create(self, request: RunRequest, contract: dict) -> str:
        if request.purpose == "collection" and not request.spec.environment.dataset.endswith("_train"):
            raise ValueError("Collection requires a train dataset; held-out data is evaluation-only")
        rid, now = ident(), time.time()
        spec = request.spec.model_dump(mode="json")
        with self.db.transaction() as c:
            c.execute(text(
                "INSERT INTO runs(id,name,purpose,spec,contract,digest,lanes,status,created,updated) "
                "VALUES(:id,:n,:p,:s,:c,:d,:l,'queued',:t,:t)"),
                dict(id=rid, n=request.name, p=request.purpose, s=json.dumps(spec),
                     c=json.dumps(contract), d=kernel_digest(), l=request.lanes, t=now))
            c.execute(text(
                "INSERT INTO run_episodes(run_id,episode,ordinal,status,updated) "
                "VALUES(:r,:e,:o,'queued',:t)"),
                [dict(r=rid, e=episode, o=i, t=now) for i, episode in enumerate(request.episodes)])
            self.db.emit(c, "run_created", rid, {"name": request.name, "episodes": len(request.episodes)})
        return rid

    def runs(self) -> list[dict]:
        rows = self.db.rows(
            "SELECT r.id,r.name,r.purpose,r.lanes,r.status,r.cancel_requested,r.created,r.updated,"
            "e.status AS episode_status,count(*) AS n FROM runs r JOIN run_episodes e ON e.run_id=r.id "
            "GROUP BY r.id,e.status ORDER BY r.created DESC")
        runs: dict[str, dict] = {}
        for row in rows:
            status, n = row.pop("episode_status"), row.pop("n")
            runs.setdefault(row["id"], {**row, "episodes": {}})["episodes"][status] = n
        return list(runs.values())

    def run(self, rid: str) -> dict:
        rows = self.db.rows("SELECT * FROM runs WHERE id=:id", id=rid)
        if not rows:
            raise KeyError("Unknown run")
        run = rows[0]
        for key in ("spec", "contract"):
            run[key] = json.loads(run[key])
        run["episodes"] = [
            {**row, "summary": json.loads(row["summary"]) if row["summary"] else None}
            for row in self.db.rows(
                "SELECT episode,ordinal,status,summary,error,started,updated FROM run_episodes "
                "WHERE run_id=:id ORDER BY ordinal", id=rid)]
        return run

    def episode(self, rid: str, episode: str) -> dict:
        rows = self.db.rows("SELECT * FROM run_episodes WHERE run_id=:r AND episode=:e", r=rid, e=episode)
        if not rows:
            raise KeyError("Unknown run episode")
        return rows[0]

    def recent(self, limit: int) -> list[dict]:
        """Latest episodes for the harness sidebar, with the API base their views read from."""
        rows = self.db.rows(
            "SELECT e.run_id,e.episode,e.status,e.started,e.updated,r.name FROM run_episodes e "
            "JOIN runs r ON r.id=e.run_id WHERE e.status!='queued' "
            "ORDER BY e.status='running' DESC, e.started DESC LIMIT :n", n=limit)
        for row in rows:
            receipt = self.folder(row["run_id"], row["episode"]) / "session.json"
            session = json.loads(receipt.read_text()) if receipt.exists() else {}
            row.update(id=f"{row['run_id']}/{row['episode']}", created=row["started"],
                       base=f"/api/runs/{row['run_id']}/episodes/{row['episode']}",
                       meta={"instruction": session.get("instruction")})
        return rows

    def active(self) -> list[dict]:
        rows = self.db.rows("SELECT * FROM runs WHERE status IN ('queued','running') ORDER BY created")
        for row in rows:
            row["spec"], row["contract"] = json.loads(row["spec"]), json.loads(row["contract"])
        return rows

    def claim(self, rid: str) -> str | None:
        with self.db.transaction() as c:
            if c.execute(text("SELECT cancel_requested FROM runs WHERE id=:id"), dict(id=rid)).scalar():
                return None
            episode = c.execute(text(
                "SELECT episode FROM run_episodes WHERE run_id=:r AND status='queued' "
                "ORDER BY ordinal LIMIT 1"), dict(r=rid)).scalar()
            if episode is None:
                return None
            now = time.time()
            c.execute(text("UPDATE run_episodes SET status='running',started=:t,updated=:t "
                           "WHERE run_id=:r AND episode=:e"), dict(r=rid, e=episode, t=now))
            c.execute(text("UPDATE runs SET status='running',updated=:t WHERE id=:r AND status='queued'"),
                      dict(r=rid, t=now))
        return episode

    def requeue(self, rid: str, episode: str):
        with self.db.transaction() as c:
            c.execute(text("UPDATE run_episodes SET status='queued',started=NULL,updated=:t "
                           "WHERE run_id=:r AND episode=:e"), dict(r=rid, e=episode, t=time.time()))

    def finish(self, rid: str, episode: str, status: str, summary: dict | None = None,
               error: str | None = None):
        assert status in TERMINAL
        with self.db.transaction() as c:
            c.execute(text("UPDATE run_episodes SET status=:s,summary=:m,error=:x,updated=:t "
                           "WHERE run_id=:r AND episode=:e AND status='running'"),
                      dict(r=rid, e=episode, s=status, x=error, t=time.time(),
                           m=json.dumps(summary) if summary is not None else None))
            self.db.emit(c, "run_episode_" + status, rid, {"episode": episode, "error": error})
        self.settle(rid)

    def fail_queued(self, rid: str, error: str):
        with self.db.transaction() as c:
            c.execute(text("UPDATE run_episodes SET status='failed',error=:x,updated=:t "
                           "WHERE run_id=:r AND status='queued'"), dict(r=rid, x=error, t=time.time()))
        self.settle(rid)

    def cancel(self, rid: str):
        self.run(rid)
        with self.db.transaction() as c:
            c.execute(text("UPDATE runs SET cancel_requested=1,updated=:t WHERE id=:r"),
                      dict(r=rid, t=time.time()))
            c.execute(text("UPDATE run_episodes SET status='cancelled',updated=:t "
                           "WHERE run_id=:r AND status='queued'"), dict(r=rid, t=time.time()))
            self.db.emit(c, "run_cancel_requested", rid, {})
        self.settle(rid)

    def settle(self, rid: str):
        """A run ends when none of its episodes is queued or running."""
        with self.db.transaction() as c:
            open_ = c.execute(text("SELECT count(*) FROM run_episodes WHERE run_id=:r "
                                   "AND status IN ('queued','running')"), dict(r=rid)).scalar()
            if open_:
                return
            cancelled = c.execute(text("SELECT cancel_requested FROM runs WHERE id=:r"), dict(r=rid)).scalar()
            c.execute(text("UPDATE runs SET status=:s,updated=:t WHERE id=:r AND status NOT IN "
                           "('completed','cancelled')"),
                      dict(r=rid, s="cancelled" if cancelled else "completed", t=time.time()))

    def interrupted(self) -> list[dict]:
        return self.db.rows("SELECT run_id,episode FROM run_episodes WHERE status='running'")


class RunScheduler:
    """Fills each active run up to its lanes, bounded by shared simulator and planner slots."""

    def __init__(self, store: RunStore, *, sim_transport=None, planner_transport=None):
        self.store = store
        self.sim_transport, self.planner_transport = sim_transport, planner_transport
        self.lanes: dict[str, set[asyncio.Task]] = {}
        self.slots: dict[tuple, asyncio.Semaphore] = {}
        # The code this process imported; the disk may change under a running worker.
        self.loaded = kernel_digest()

    def slot(self, key: tuple, size: int) -> asyncio.Semaphore:
        return self.slots.setdefault(key, asyncio.Semaphore(size))

    async def recover(self):
        """Episodes left running by a dead owner end as uncertain; their sessions are released."""
        for row in self.store.interrupted():
            folder = self.store.folder(row["run_id"], row["episode"])
            released = await self.release(folder / "session.json")
            self.store.finish(row["run_id"], row["episode"], "failed_or_uncertain",
                              error="interrupted by a worker restart; " + released)

    async def release(self, receipt: Path) -> str:
        if not receipt.exists():
            return "no session was recorded"
        session = json.loads(receipt.read_text())
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=30, transport=self.sim_transport) as http:
                response = await http.delete(f"{session['url'].rstrip('/')}/v1/sessions/"
                                             f"{session['session_id']}")
            return f"session release HTTP {response.status_code}"
        except httpx.HTTPError as error:
            return f"session release failed: {type(error).__name__}"

    def tick(self):
        for run in self.store.active():
            lanes = self.lanes.setdefault(run["id"], set())
            lanes -= {task for task in lanes if task.done()}
            if run["cancel_requested"]:
                for task in lanes:
                    task.cancel()
                continue
            if run["digest"] != self.loaded:
                if kernel_digest() != run["digest"]:
                    self.store.fail_queued(run["id"], "kernel source changed since the run was frozen")
                continue  # frozen against newer code than this worker loaded: wait for a restart
            while len(lanes) < run["lanes"]:
                lanes.add(asyncio.create_task(self.lane(run)))

    async def lane(self, run: dict):
        spec = KernelSpec.model_validate(run["spec"])
        while True:
            if kernel_digest() != run["digest"]:  # edited while this lane was running
                self.store.fail_queued(run["id"], "kernel source changed since the run was frozen")
                return
            episode = self.store.claim(run["id"])
            if episode is None:
                return
            await self.execute(run, spec, episode)

    async def execute(self, run: dict, spec: KernelSpec, episode: str):
        rid, planner = run["id"], spec.planner
        environment = self.slot(("sim", spec.environment.url), run["contract"]["workers"])
        model = self.slot(("planner", planner.base_url, planner.model), planner.max_concurrency)
        try:
            async with environment, model:
                result = await run_episode(spec, episode, self.store.folder(rid, episode),
                                           run["contract"]["max_actions"],
                                           sim_transport=self.sim_transport,
                                           planner_transport=self.planner_transport)
        except SimulatorBusy:
            # Nothing was opened or written: the simulator is shared with another client.
            self.store.requeue(rid, episode)
            await asyncio.sleep(BUSY_BACKOFF_S)
            return
        except asyncio.CancelledError:
            self.store.finish(rid, episode, "cancelled", error="cancelled while running")
            raise
        except Exception as error:
            self.store.finish(rid, episode, "failed", error=f"{type(error).__name__}: {error}")
            return
        folder = self.store.folder(rid, episode)
        evaluation = folder / "evaluation.json"
        self.store.finish(rid, episode, result["status"],
                          summarize(result, json.loads(evaluation.read_text())
                                    if evaluation.exists() else None),
                          error=result.get("error") if result["status"] != "completed" else None)


def bundle(folder: Path) -> tuple[Path, str]:
    """A gzip tarball of one episode folder and its sha256."""
    handle = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    with tarfile.open(fileobj=handle, mode="w:gz") as tar:
        tar.add(folder, arcname=folder.name)
    handle.close()
    digest = hashlib.sha256(Path(handle.name).read_bytes()).hexdigest()
    return Path(handle.name), digest


def router(store: RunStore, settings=None) -> APIRouter:
    """``settings`` holds the named planner profiles and environments that ``/compose`` may use."""
    api = APIRouter(prefix="/api/runs")

    async def submit(request: RunRequest) -> dict:
        contract = await preflight(request.spec)
        return {"id": store.create(request, contract)}

    @api.post("", status_code=201)
    async def create(body: RunRequest):
        return await submit(body)

    @api.post("/compose", status_code=201)
    async def compose(body: ComposeRun):
        return await submit(body.request(settings))

    @api.get("")
    def listing():
        return store.runs()

    @api.get("/episodes/recent")
    def recent():
        return store.recent(20)

    @api.get("/{rid}")
    def detail(rid: str):
        return store.run(rid)

    @api.post("/{rid}/cancel")
    def cancel(rid: str):
        store.cancel(rid)
        return store.run(rid)

    @api.get("/{rid}/episodes/{episode}/bundle")
    def download(rid: str, episode: str):
        if store.episode(rid, episode)["status"] not in TERMINAL:
            raise HTTPException(409, "episode is not finished")
        folder = store.folder(rid, episode)
        if not folder.is_dir():
            raise HTTPException(404, "episode wrote no bundle")
        path, digest = bundle(folder)
        return FileResponse(path, media_type="application/gzip", filename=f"{rid}-{episode}.tar.gz",
                            headers={"X-Content-SHA256": digest}, background=BackgroundTask(path.unlink))

    @api.get("/{rid}/episodes/{episode}/stream")
    async def stream(rid: str, episode: str, request: Request, after: str = ""):
        store.episode(rid, episode)
        feed = KernelFeed(store.folder(rid, episode), request.headers.get("last-event-id") or after)

        def status():
            row = store.episode(rid, episode)
            return {"status": row["status"], "error": row["error"], "run_id": rid}

        return follow(request, feed, status, lambda current: current["status"] in TERMINAL)

    @api.get("/{rid}/episodes/{episode}/artifact/{digest}")
    def artifact(rid: str, episode: str, digest: str):
        store.episode(rid, episode)
        return artifact_response(store.folder(rid, episode) / "trace" / "media", digest)

    @api.get("/{rid}/episodes/{episode}/files/{relative:path}")
    def files(rid: str, episode: str, relative: str):
        store.episode(rid, episode)
        folder = store.folder(rid, episode)
        media, _, digest = relative.rpartition("/")
        if media == "trace/media":
            return artifact_response(folder / media, digest)
        return image_file(folder, relative)

    return api
