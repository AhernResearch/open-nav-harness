"""Open-access management API. No navigation logic here."""

import asyncio
import hashlib
import json
import time
import zipfile
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text
from nav_harness.core.config import ExperimentSpec, PlatformSettings, ServiceSpec
from .database import Database, ident
from .services import ServiceManager
from .catalog import Catalog
from .harness import Feed, episode_meta
from .results import attempt_detail, csv_export, metrics, aggregate, read_jsonl


class Action(BaseModel):
    action: str


class Enqueue(BaseModel):
    experiment_id: str


class ImportRequest(BaseModel):
    path: str


def create_app(root: Path, settings: PlatformSettings | None = None):
    settings = settings or PlatformSettings()
    root = root.resolve()
    db = Database(root)
    db.migrate()
    manager = ServiceManager(db, settings)

    @asynccontextmanager
    async def lifespan(app):
        async def monitor():
            while True:
                await manager.reconcile_services()
                await asyncio.sleep(3)

        task = asyncio.create_task(monitor())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            db.engine.dispose()

    app = FastAPI(title="Open Nav Harness", version="0.1.0", lifespan=lifespan)
    app.state.db = db
    app.state.services = manager

    @app.middleware("http")
    async def response_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(ValueError)
    async def bad_value(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(RuntimeError)
    async def operation_failed(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.exception_handler(KeyError)
    async def not_found(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.get("/health")
    def health():
        return {"ok": True, "version": "0.1.0"}

    @app.get("/api/overview")
    def overview():
        counts = db.rows("SELECT status,COUNT(*) AS count FROM jobs GROUP BY status")
        return {
            "jobs": counts,
            "services": len(db.rows("SELECT id FROM services")),
            "attempts": len(db.rows("SELECT id FROM attempts")),
        }

    @app.get("/api/services")
    def services():
        return [
            {**r, "spec": json.loads(r["spec"])} for r in db.rows("SELECT * FROM services ORDER BY created")
        ]

    @app.post("/api/services", status_code=201)
    def register(body: ServiceSpec):
        if body.managed and body.template not in settings.service_templates:
            raise ValueError("Unknown service template")
        return {"id": db.register_service(body)}

    @app.get("/api/inference/status")
    async def inference_status():
        shared = [
            r
            for r in db.rows("SELECT id,spec FROM services")
            if json.loads(r["spec"]).get("metadata", {}).get("shared_inference")
        ]

        async def inspect(row):
            return {
                "service_id": row["id"],
                "name": json.loads(row["spec"])["name"],
                **await manager.health(row["id"]),
            }

        return await asyncio.gather(*(inspect(row) for row in shared))

    @app.get("/api/templates")
    def templates():
        return [{"name": k, "host": v.host} for k, v in settings.service_templates.items()]

    @app.get("/api/services/{sid}/health")
    async def service_health(sid: str):
        return await manager.health(sid)

    @app.post("/api/services/{sid}/action")
    async def service_action(sid: str, body: Action):
        return await manager.operate(sid, body.action)

    @app.get("/api/services/{sid}/logs")
    async def service_logs(sid: str):
        return {"text": await manager.logs(sid)}

    @app.get("/api/gpus")
    async def gpus():
        results = []
        for host in sorted({t.host for t in settings.service_templates.values()}):
            try:
                data = await manager.ssh(
                    host,
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                )
                results.append({"host": host, "csv": data})
            except RuntimeError as exc:
                results.append({"host": host, "error": str(exc)})
        return results

    @app.get("/api/experiments")
    def experiments():
        return [
            {**r, "spec": json.loads(r["spec"])}
            for r in db.rows("SELECT * FROM experiments ORDER BY created DESC")
        ]

    @app.post("/api/experiments", status_code=201)
    def experiment(body: ExperimentSpec):
        deployments = {}
        for sid in [
            body.planner_service,
            body.environment_service,
            body.vla_service,
            *(b.environment_service for b in body.execution_pool),
            *(b.execution_service for b in body.execution_pool),
        ]:
            if not sid:
                continue
            service = db.service(sid)
            template = settings.service_templates.get(service.template)
            if template:
                deployments[sid] = {
                    "host": template.host,
                    "argv": template.argv,
                    "working_directory": template.working_directory,
                    "environment_names": sorted(template.environment),
                }
        return {"id": db.create_experiment(body, deployments)}

    @app.get("/api/jobs")
    def jobs():
        rows = db.rows(
            "SELECT j.*,e.spec FROM jobs j JOIN experiments e ON e.id=j.experiment_id ORDER BY created DESC"
        )
        for row in rows:
            spec = json.loads(row.pop("spec"))
            row["name"] = spec["name"]
            row["total"] = len(spec["episodes"])
            row["completed"] = db.rows(
                "SELECT COUNT(DISTINCT episode) AS n FROM attempts WHERE job_id=:j AND status='completed'",
                j=row["id"],
            )[0]["n"]
            row["active"] = db.rows(
                "SELECT COUNT(*) AS n FROM attempts WHERE job_id=:j AND status='running'", j=row["id"]
            )[0]["n"]
            row["execution_pool"] = spec.get("execution_pool", [])
            row["resources"] = {
                k: spec.get(k) for k in ("planner_service", "environment_service", "vla_service")
            }
        return rows

    @app.post("/api/jobs", status_code=201)
    def enqueue(body: Enqueue):
        return {"id": db.enqueue(body.experiment_id)}

    @app.post("/api/jobs/{jid}/action")
    def job_action(jid: str, body: Action):
        db.control(jid, body.action)
        return {"ok": True}

    @app.get("/api/jobs/{jid}/attempts")
    def attempts(jid: str):
        return [metrics(r) for r in db.rows("SELECT * FROM attempts WHERE job_id=:j ORDER BY created", j=jid)]

    def get_attempt(aid):
        rows = db.rows("SELECT * FROM attempts WHERE id=:id", id=aid)
        if not rows:
            raise KeyError("Unknown attempt")
        return rows[0]

    @app.get("/api/attempts/{aid}")
    def attempt(aid: str):
        return attempt_detail(get_attempt(aid))

    @app.post("/api/attempts/{aid}/reconcile")
    async def reconcile_attempt(aid: str):
        await manager.reconcile_attempt(aid)
        return {"ok": True}

    @app.get("/api/attempts/{aid}/artifact/{digest}")
    def artifact(aid: str, digest: str):
        a = get_attempt(aid)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise HTTPException(404)
        path = Path(a["output"]) / "episode" / "work" / f"ep{a['episode']:04d}" / "trace" / "media" / digest
        if not path.is_file():
            raise HTTPException(404)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise HTTPException(409, "Artifact digest mismatch")
        return Response(
            raw,
            media_type="image/png"
            if raw.startswith(b"\x89PNG")
            else "image/webp"
            if raw.startswith(b"RIFF")
            else "image/jpeg",
            # Content-addressed and verified above, so it can never change under this URL.
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )

    def image_file(directory: Path, relative: str):
        path = (directory / relative).resolve()
        if (
            not path.is_relative_to(directory.resolve())
            or path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp")
            or not path.is_file()
        ):
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/api/attempts/{aid}/files/{relative:path}")
    def attempt_image(aid: str, relative: str):
        return image_file(Path(get_attempt(aid)["output"]), relative)

    @app.get("/api/jobs/{jid}/metrics.csv")
    def export_metrics(jid: str):
        rows = db.rows("SELECT * FROM attempts WHERE job_id=:j ORDER BY created", j=jid)
        return Response(
            "\ufeff" + csv_export(rows),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="metrics.csv"'},
        )

    @app.get("/api/attempts/{aid}/export")
    def export_trace(aid: str):
        a = get_attempt(aid)
        if a["status"] == "running":
            raise HTTPException(409, "Cannot export an active attempt")
        source = Path(a["output"])
        destination = root / "exports" / (aid + ".zip")
        destination.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for path in source.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    z.write(path, path.relative_to(source))
            z.writestr(
                "review.json",
                json.dumps(
                    {
                        "training_status": (json.loads(a["result"]) if a["result"] else {}).get(
                            "training_status", "incomplete"
                        ),
                        "semantic_verified": False,
                        "attempt_id": aid,
                    }
                ),
            )
        return FileResponse(destination, filename=aid + ".zip")

    @app.get("/api/compare")
    def compare(job_ids: str):
        ids = job_ids.split(",")
        if len(ids) > 8:
            raise ValueError("Compare at most eight jobs")
        result = []
        for jid in ids:
            rows = [
                metrics(r) for r in db.rows("SELECT * FROM attempts WHERE job_id=:j ORDER BY created", j=jid)
            ]
            result.append({"job_id": jid, "rows": rows, "aggregate": aggregate(rows)})
        return result

    @app.post("/api/imports")
    def import_result(body: ImportRequest):
        path = Path(body.path).resolve()
        allowed = [Path(p).resolve() for p in settings.import_roots]
        if not any(path.is_relative_to(p) for p in allowed) or not (path / "summary.json").is_file():
            raise ValueError("Import must reference a summary directory inside configured read-only roots")
        iid = ident()
        with db.transaction() as c:
            c.execute(
                text("INSERT INTO imports(id,path,created) VALUES(:id,:p,:t)"),
                {"id": iid, "p": str(path), "t": time.time()},
            )
            db.emit(c, "result_imported", iid, {})
        return {"id": iid}

    @app.get("/api/imports")
    def imports():
        return [
            {**r, "summary": json.loads((Path(r["path"]) / "summary.json").read_text())}
            for r in db.rows("SELECT * FROM imports")
        ]

    def get_import(iid):
        rows = db.rows("SELECT * FROM imports WHERE id=:id", id=iid)
        if not rows:
            raise KeyError("Unknown import")
        directory = Path(rows[0]["path"]).resolve()
        if not any(directory.is_relative_to(Path(p).resolve()) for p in settings.import_roots):
            raise ValueError("Import root is no longer allowed")
        return directory

    @app.get("/api/imports/{iid}/episodes/{episode}")
    def imported_episode(iid: str, episode: int):
        directory = get_import(iid)
        summary = json.loads((directory / "summary.json").read_text())
        record = next((r for r in summary.get("episodes", []) if r.get("index") == episode), None)
        if record is None:
            raise KeyError("Unknown episode")
        images = [
            p.relative_to(directory).as_posix()
            for p in sorted((directory / "live" / f"ep{episode:04d}").rglob("*.png"))
            if not p.is_symlink()
        ]
        return {
            "result": record,
            "events": read_jsonl(directory / f"episode_{episode}.jsonl"),
            "images": images,
            "input_visibility": "legacy_observable_events_only",
            "training_status": "evaluation_only",
        }

    @app.get("/api/imports/{iid}/files/{relative:path}")
    def imported_image(iid: str, relative: str):
        return image_file(get_import(iid), relative)

    @app.get("/api/events")
    async def events(request: Request, after: int = 0):
        last = max(after, int(request.headers.get("last-event-id", "0")))

        async def stream():
            nonlocal last
            while not await request.is_disconnected():
                rows = db.rows("SELECT * FROM events WHERE seq>:n ORDER BY seq LIMIT 100", n=last)
                for row in rows:
                    last = row["seq"]
                    yield f"id: {last}\ndata: {json.dumps(row)}\n\n"
                if not rows:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    @app.get("/api/harness/services")
    async def harness_services():
        """Probe registered choices without adopting processes or changing scheduler state."""
        async def choice(row):
            spec = row["spec"]
            health = await manager.health(row["id"])
            profile = spec.get("planner")
            sdk = profile and profile["runtime"] != "openai_compatible"
            available = health["ok"] is True or bool(sdk and health["ok"] is None)
            reason = "" if available else "服务暂不可用"
            if available and profile and not sdk:
                advertised = health.get("details", {}).get("data", [])
                if not any(m.get("id") == profile["model"] for m in advertised):
                    available, reason = False, "服务未提供此模型"
            if spec["managed"] and row["status"] != "ready":
                available, reason = False, "请先在 Workbench 启动服务"
            return {**row, "available": available, "unavailable_reason": reason}

        return await asyncio.gather(*(choice(row) for row in services()))

    @app.get("/api/harness/jobs/{jid}")
    def harness_job(jid: str):
        """Read-only queue feedback; resource admission stays in the scheduler."""
        rows = db.rows("SELECT * FROM jobs WHERE id=:id", id=jid)
        if not rows:
            raise KeyError("Unknown job")
        job = rows[0]
        spec = json.loads(db.rows("SELECT spec FROM experiments WHERE id=:id", id=job["experiment_id"])[0]["spec"])
        attempts = db.rows("SELECT id FROM attempts WHERE job_id=:id ORDER BY created DESC LIMIT 1", id=jid)
        reason, blocked = "queued", None
        if job["status"] == "queued":
            pool = spec.get("execution_pool") or []
            groups = [[b["environment_service"], b["execution_service"]] for b in pool] if pool else [
                [spec["environment_service"], *([spec["vla_service"]] if spec.get("vla_service") else [])]
            ]
            services = {r["id"]: r for r in db.rows("SELECT id,status,spec FROM services")}
            def ready(sid):
                row = services[sid]
                return not json.loads(row["spec"])["managed"] or row["status"] == "ready"

            shared = [spec[k] for k in ("planner_service", "vla_service") if spec.get(k)]
            ready_groups = [group for group in groups if all(ready(sid) for sid in group)]
            leases = {r["resource"]: r for r in db.rows(
                "SELECT l.resource,a.id,a.status FROM leases l JOIN attempts a ON a.id=l.attempt_id"
            )}
            if not ready_groups or not all(ready(sid) for sid in shared):
                reason = "services_unavailable"
            elif not any(all(sid not in leases for sid in group) for group in ready_groups):
                blocked = next((leases[sid]["id"] for group in ready_groups for sid in group
                                if sid in leases and leases[sid]["status"] == "blocked"), None)
                reason = "needs_reconciliation" if blocked else "resources_busy"
        return {**job, "attempt_id": attempts[0]["id"] if attempts else None,
                "waiting_reason": reason, "blocked_attempt_id": blocked}

    @app.get("/api/harness/attempts")
    def harness_attempts():
        rows = db.rows(
            "SELECT a.id,a.job_id,a.episode,a.ordinal,a.status,a.created,a.updated,a.output,e.spec FROM attempts a "
            "JOIN jobs j ON j.id=a.job_id JOIN experiments e ON e.id=j.experiment_id "
            "ORDER BY a.status='running' DESC, a.created DESC LIMIT 20"
        )
        for row in rows:
            row["name"] = json.loads(row.pop("spec"))["name"]
            row["meta"] = episode_meta(Path(row.pop("output")), row["episode"])
        return rows

    catalog = Catalog(root / "catalog")

    @app.get("/api/catalog")
    def catalogs():
        return catalog.datasets()

    @app.get("/api/catalog/{name}/episodes")
    def catalog_episodes(name: str, scene: str, q: str = "", limit: int = 50):
        return catalog.episodes(name, scene, q, min(limit, 200))

    @app.get("/api/attempts/{aid}/stream")
    async def attempt_stream(aid: str, request: Request, after: str = ""):
        """Item lifecycle events of one attempt; a replay is the same stream read from zero."""
        a = get_attempt(aid)
        feed = Feed(Path(a["output"]), a["episode"], request.headers.get("last-event-id") or after)

        def attempt_status():
            return db.rows(
                "SELECT a.status,a.error,a.job_id,j.status AS job_status FROM attempts a "
                "JOIN jobs j ON j.id=a.job_id WHERE a.id=:id",
                id=aid,
            )[0]

        async def stream():
            yield "retry: 3000\n\n"
            status, quiet = None, 0
            while not await request.is_disconnected():
                events = await asyncio.to_thread(feed.poll)
                if events is None:
                    yield "event: reset\ndata: {}\n\n"
                    return
                quiet = 0 if events else quiet + 1
                current = await asyncio.to_thread(attempt_status)
                if current != status:
                    status = current
                    events.append({"type": "attempt.status", **current})
                if events:
                    yield f"id: {feed.cursor}\ndata: {json.dumps(events, ensure_ascii=False)}\n\n"
                elif current["status"] != "running" and quiet >= 2:
                    yield f"event: end\ndata: {json.dumps(current)}\n\n"
                    return
                else:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    dist = Path(__file__).resolve().parents[1] / "web"

    def page():
        index = dist / "index.html"
        if not index.is_file():
            raise HTTPException(404)
        return FileResponse(index)

    for route in ("/harness", "/workbench"):
        app.get(route, include_in_schema=False)(page)
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app
