"""Open-access management API. No navigation logic here."""

import asyncio
import json
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from open_nav_harness.core.config import PlatformSettings, ServiceSpec
from .database import Database
from .services import ServiceManager
from .catalog import Catalog
from .runs import RunStore, router as runs_router


class Action(BaseModel):
    action: str


def create_app(root: Path, settings: PlatformSettings | None = None):
    settings = settings or PlatformSettings()
    root = root.resolve()
    db = Database(root)
    db.migrate()
    manager = ServiceManager(db, settings)
    from .conversations import Conversations, router
    conversations = Conversations(db, settings.conversations_config)

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
            await conversations.close()
            db.engine.dispose()

    app = FastAPI(title="Open Nav Harness", version="0.1.0", lifespan=lifespan)
    app.include_router(router(conversations))
    runs = RunStore(db)
    app.include_router(runs_router(runs, conversations.settings))
    app.state.conversations = conversations
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

    catalog = Catalog(root / "catalog")

    @app.get("/api/catalog")
    def catalogs():
        return catalog.datasets()

    @app.get("/api/catalog/{name}/episodes")
    def catalog_episodes(name: str, scene: str, q: str = "", limit: int = 50):
        return catalog.episodes(name, scene, q, min(limit, 200))

    dist = Path(__file__).resolve().parents[1] / "web"

    def page():
        index = dist / "index.html"
        if not index.is_file():
            raise HTTPException(404)
        return FileResponse(index)

    for route in ("/harness", "/workbench", "/chat"):
        app.get(route, include_in_schema=False)(page)
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app
