"""Management CLI; web and CLI share configuration and application services."""

import argparse
import asyncio
import json
from pathlib import Path
from nav_harness.core.config import PlatformSettings
from nav_harness.core.artifacts import atomic_json


def main():
    p = argparse.ArgumentParser(prog="nav-harness")
    p.add_argument("--state", type=Path, default=Path("state"))
    p.add_argument("--settings", type=Path)
    sub = p.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    sub.add_parser("worker")
    sub.add_parser("init")
    sub.add_parser("openapi")
    backup = sub.add_parser("backup")
    backup.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.state.resolve()
    settings = (
        PlatformSettings.model_validate_json(a.settings.read_text()) if a.settings else PlatformSettings()
    )
    if a.command == "worker":
        from nav_harness.platform.database import Database
        from nav_harness.platform.worker import Worker

        db = Database(root)
        db.migrate()
        asyncio.run(Worker(db, settings).run())
    elif a.command == "backup":
        import sqlite3

        a.output.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(root / "platform.sqlite3") as source, sqlite3.connect(a.output) as target:
            source.backup(target)
        print("Metadata backup written; preserve runs/ and private deployment settings separately.")
    else:
        from nav_harness.platform.api import create_app

        app = create_app(root, settings)
        if a.command == "openapi":
            print(json.dumps(app.openapi()))
        elif a.command == "init":
            atomic_json(root / "settings.example.json", settings.model_dump())
            print("Initialized. Console and API are accessible without a password.")
        else:
            import uvicorn

            uvicorn.run(app, host=a.host or settings.host, port=a.port or settings.port)


if __name__ == "__main__":
    main()
