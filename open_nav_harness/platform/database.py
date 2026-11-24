"""Durable control-plane metadata: services, conversations and runs share one SQLite file."""

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from sqlalchemy import create_engine, event, text
from open_nav_harness.core.config import ServiceSpec


def ident():
    return uuid.uuid4().hex


class Database:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.engine = create_engine(
            "sqlite:///" + str(root / "platform.sqlite3"), connect_args={"timeout": 30}
        )

        @event.listens_for(self.engine, "connect")
        def pragmas(connection, _):
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=30000")

    def migrate(self):
        from alembic.config import Config
        from alembic import command

        cfg = Config()
        cfg.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
        cfg.attributes["engine"] = self.engine
        command.upgrade(cfg, "head")

    @contextmanager
    def transaction(self):
        with self.engine.connect() as c:
            c.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                yield c
                c.commit()
            except BaseException:
                c.rollback()
                raise

    def rows(self, sql, **params):
        with self.engine.connect() as c:
            return [dict(r) for r in c.execute(text(sql), params).mappings()]

    @staticmethod
    def emit(c, kind, entity, payload):
        c.execute(
            text("INSERT INTO events(time,kind,entity,payload) VALUES(:t,:k,:e,:p)"),
            {"t": time.time(), "k": kind, "e": entity, "p": json.dumps(payload, ensure_ascii=False)},
        )

    def register_service(self, spec: ServiceSpec):
        sid = ident()
        with self.transaction() as c:
            for old in c.execute(text("SELECT spec FROM services")).scalars():
                d = json.loads(old)
                if d["name"] == spec.name or d["url"].rstrip("/") == spec.url.rstrip("/"):
                    raise ValueError("Service name and endpoint must be unique")
            c.execute(
                text("INSERT INTO services(id,spec,created) VALUES(:id,:s,:t)"),
                {"id": sid, "s": spec.model_dump_json(), "t": time.time()},
            )
            self.emit(c, "service_registered", sid, {"name": spec.name})
        return sid

    def service(self, sid):
        rows = self.rows("SELECT * FROM services WHERE id=:id", id=sid)
        if not rows:
            raise KeyError("Unknown service")
        return ServiceSpec.model_validate_json(rows[0]["spec"])
