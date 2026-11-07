"""Authenticated single-origin management API. No navigation logic here."""

import asyncio
import hashlib
import hmac
import json
import secrets
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
from .results import attempt_detail, csv_export, metrics, aggregate, read_jsonl


class Login(BaseModel):
    password: str


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
    password_path = root / "admin-password"
    if not password_path.exists():
        password_path.write_text(secrets.token_urlsafe(24))
        password_path.chmod(0o600)
    password = password_path.read_text().strip()
    key = hashlib.sha256(password.encode()).digest()
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

    app = FastAPI(title="Nav Harness", version="0.1.0", lifespan=lifespan)
    app.state.db = db
    app.state.services = manager

    def authenticated(request):
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:], password):
            return True
        cookie = request.cookies.get("navh_session", "")
        try:
            stamp, signature = cookie.split(".")
            expected = hmac.new(key, stamp.encode(), "sha256").hexdigest()
            return 0 <= time.time() - int(stamp) < 28800 and hmac.compare_digest(signature, expected)
        except (ValueError, TypeError):
            return False

    @app.middleware("http")
    async def authentication(request: Request, call_next):
        if request.url.path.startswith("/api/") and request.url.path != "/api/login":
            if not authenticated(request):
                return JSONResponse({"detail": "Authentication required"}, status_code=401)
            if request.method not in ("GET", "HEAD") and not request.headers.get("authorization"):
                origin = request.headers.get("origin")
                if origin != str(request.base_url).rstrip("/"):
                    return JSONResponse({"detail": "Same-origin request required"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
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

    @app.post("/api/login")
    async def login(body: Login, request: Request):
        if not hmac.compare_digest(body.password, password):
            await asyncio.sleep(0.5)
            raise HTTPException(401, "Invalid password")
        stamp = str(int(time.time()))
        signed = stamp + "." + hmac.new(key, stamp.encode(), "sha256").hexdigest()
        response = JSONResponse({"ok": True})
        response.set_cookie(
            "navh_session",
            signed,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            max_age=28800,
        )
        return response

    @app.post("/api/logout")
    def logout():
        response = JSONResponse({"ok": True})
        response.delete_cookie("navh_session")
        return response

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
        return {"id": db.create_experiment(body)}

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
    def reconcile_attempt(aid: str):
        db.release_reconciled(aid)
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

    dist = Path(__file__).resolve().parents[1] / "web"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app
