"""Durable control-plane metadata; transactions serialize job claims."""

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from sqlalchemy import create_engine, event, text
from nav_harness.core.config import ExperimentSpec, ServiceSpec


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

    def create_experiment(self, spec: ExperimentSpec, deployments: dict | None = None):
        from navgpt.settings import Settings, _validate

        services = {
            k: self.service(getattr(spec, k))
            for k in ["environment_service", "planner_service", "vla_service"]
            if getattr(spec, k)
        }
        for field, expected in [
            ("environment_service", "environment"),
            ("planner_service", "planner"),
            ("vla_service", "vla"),
        ]:
            if field in services and services[field].kind != expected:
                raise ValueError("Service kind mismatch: " + field)
        if spec.purpose == "collection" and spec.config.get("dataset", {}).get("split") != "train":
            raise ValueError("Collection requires the explicit train split; held-out data is evaluation-only")
        cfg = dict(spec.config)
        # Build effective config now: workers use exactly this frozen snapshot.
        cfg["environment"] = {**cfg.get("environment", {}), "url": services["environment_service"].url}
        if "vla_service" in services:
            cfg["vla"] = {**cfg.get("vla", {}), "url": services["vla_service"].url}
            if services["vla_service"].metadata.get("tool_execution"):
                cfg["harness"] = {**cfg.get("harness", {}), "execution_url": services["vla_service"].url}
        if "planner_service" in services:
            profile = services["planner_service"].planner
            cfg["planner"] = {
                **cfg.get("planner", {}),
                "runtime": profile.runtime,
                "model": profile.model,
                "max_budget_usd": None,
            }
            if profile.runtime == "openai_compatible":
                cfg["planner"]["profile"] = profile.model_dump()
            elif profile.effort:
                cfg["planner"]["effort"] = profile.effort
        else:
            cfg.pop("planner", None)
        cfg["output"] = {"save_frames": True}
        _validate(cfg, Path("experiment"))
        run = Settings(Path("experiment"), cfg, {}).run_config()
        from navgpt.planner.config import needs_vla, VLA_ONLY

        if (needs_vla(run.condition) or run.condition == VLA_ONLY) and not spec.vla_service:
            raise ValueError("This mode requires a VLA service")
        frozen = spec.model_dump()
        frozen["config"] = cfg
        from nav_harness.core.version import implementation_digest

        frozen["implementation_digest"] = implementation_digest()
        frozen["deployment_templates"] = deployments or {}
        frozen["services"] = {k: v.model_dump() for k, v in services.items()}
        eid = ident()
        with self.transaction() as c:
            c.execute(
                text("INSERT INTO experiments(id,spec,created) VALUES(:id,:s,:t)"),
                {"id": eid, "s": json.dumps(frozen), "t": time.time()},
            )
            self.emit(c, "experiment_created", eid, {"name": spec.name})
        return eid

    def enqueue(self, eid):
        if not self.rows("SELECT id FROM experiments WHERE id=:id", id=eid):
            raise KeyError("Unknown experiment")
        jid = ident()
        now = time.time()
        with self.transaction() as c:
            c.execute(
                text(
                    "INSERT INTO jobs(id,experiment_id,status,created,updated) VALUES(:id,:e,'queued',:t,:t)"
                ),
                {"id": jid, "e": eid, "t": now},
            )
            self.emit(c, "job_queued", jid, {})
        return jid

    def control(self, jid, action):
        if action not in ("pause", "resume", "cancel"):
            raise ValueError("Unknown job action")
        with self.transaction() as c:
            row = c.execute(text("SELECT * FROM jobs WHERE id=:id"), {"id": jid}).mappings().first()
            if not row:
                raise KeyError("Unknown job")
            if row["status"] in ("completed", "cancelled", "failed"):
                raise ValueError("Terminal jobs are immutable; create another job")
            if action == "resume":
                active = c.execute(
                    text("SELECT id FROM attempts WHERE job_id=:id AND status IN ('running','blocked')"),
                    {"id": jid},
                ).first()
                if active:
                    raise ValueError(
                        "Reconcile blocked resources or wait for the active attempt before resuming"
                    )
                c.execute(
                    text(
                        "UPDATE jobs SET status='queued',pause_requested=0,cancel_requested=0,updated=:t WHERE id=:id"
                    ),
                    {"id": jid, "t": time.time()},
                )
            else:
                key = "pause_requested" if action == "pause" else "cancel_requested"
                c.execute(
                    text(f"UPDATE jobs SET {key}=1,updated=:t WHERE id=:id"), {"id": jid, "t": time.time()}
                )
            if action in ("pause", "cancel"):
                active = c.execute(
                    text("SELECT 1 FROM attempts WHERE job_id=:id AND status='running'"), {"id": jid}
                ).first()
                if not active:
                    status = "cancelled" if action == "cancel" else "paused"
                    c.execute(
                        text("UPDATE jobs SET status=:s,updated=:t WHERE id=:id"),
                        {"s": status, "t": time.time(), "id": jid},
                    )
            self.emit(c, "job_" + action, jid, {})

    def claim(self):
        with self.transaction() as c:
            jobs = (
                c.execute(
                    text(
                        "SELECT j.*,e.spec FROM jobs j JOIN experiments e ON e.id=j.experiment_id WHERE j.status IN ('queued','running') ORDER BY j.created"
                    )
                )
                .mappings()
                .all()
            )
            for job in jobs:
                jid = job["id"]
                spec = json.loads(job["spec"])
                running = c.execute(
                    text("SELECT id FROM attempts WHERE job_id=:j AND status='running'"), {"j": jid}
                ).first()
                if running:
                    continue
                if job["cancel_requested"] or job["pause_requested"]:
                    status = "cancelled" if job["cancel_requested"] else "paused"
                    c.execute(
                        text("UPDATE jobs SET status=:s,updated=:t WHERE id=:j"),
                        {"s": status, "t": time.time(), "j": jid},
                    )
                    self.emit(c, "job_" + status, jid, {})
                    continue
                done = set(
                    c.execute(
                        text("SELECT episode FROM attempts WHERE job_id=:j AND status='completed'"),
                        {"j": jid},
                    ).scalars()
                )
                pending = [ep for ep in spec["episodes"] if ep not in done]
                if not pending:
                    c.execute(
                        text("UPDATE jobs SET status='completed',updated=:t WHERE id=:j"),
                        {"t": time.time(), "j": jid},
                    )
                    self.emit(c, "job_completed", jid, {})
                    continue
                service_ids = [
                    spec[k] for k in ("environment_service", "vla_service", "planner_service") if spec.get(k)
                ]
                unavailable = []
                for sid in service_ids:
                    service = (
                        c.execute(text("SELECT spec,status FROM services WHERE id=:id"), {"id": sid})
                        .mappings()
                        .one()
                    )
                    if json.loads(service["spec"])["managed"] and service["status"] != "ready":
                        unavailable.append(sid)
                if unavailable:
                    c.execute(
                        text("UPDATE jobs SET error='Waiting for managed services to be ready' WHERE id=:j"),
                        {"j": jid},
                    )
                    continue
                resources = [spec["environment_service"]]
                if spec.get("vla_service"):
                    resources.append(spec["vla_service"])
                if any(
                    c.execute(text("SELECT 1 FROM leases WHERE resource=:r"), {"r": r}).first()
                    for r in resources
                ):
                    continue
                planner = spec.get("planner_service")
                if planner:
                    slots = spec["services"]["planner_service"]["planner"]["max_concurrency"]
                    free = next(
                        (
                            f"{planner}:{i}"
                            for i in range(slots)
                            if not c.execute(
                                text("SELECT 1 FROM leases WHERE resource=:r"), {"r": f"{planner}:{i}"}
                            ).first()
                        ),
                        None,
                    )
                    if free is None:
                        continue
                    resources.append(free)
                episode = pending[0]
                aid = ident()
                now = time.time()
                ordinal = (
                    c.execute(
                        text("SELECT COUNT(*) FROM attempts WHERE job_id=:j AND episode=:e"),
                        {"j": jid, "e": episode},
                    ).scalar_one()
                    + 1
                )
                out = str(self.root / "runs" / jid / aid)
                c.execute(
                    text(
                        "INSERT INTO attempts(id,job_id,episode,ordinal,status,output,created,updated) VALUES(:a,:j,:e,:o,'running',:p,:t,:t)"
                    ),
                    {"a": aid, "j": jid, "e": episode, "o": ordinal, "p": out, "t": now},
                )
                for resource in resources:
                    c.execute(
                        text("INSERT INTO leases(resource,attempt_id) VALUES(:r,:a)"),
                        {"r": resource, "a": aid},
                    )
                c.execute(
                    text("UPDATE jobs SET status='running',error=NULL,updated=:t WHERE id=:j"),
                    {"t": now, "j": jid},
                )
                self.emit(c, "attempt_started", aid, {"job_id": jid, "episode": episode, "ordinal": ordinal})
                return {
                    "id": aid,
                    "job_id": jid,
                    "episode": episode,
                    "ordinal": ordinal,
                    "output": out,
                    "spec": spec,
                }
        return None

    def block(self, aid, error):
        with self.transaction() as c:
            row = c.execute(text("SELECT job_id FROM attempts WHERE id=:id"), {"id": aid}).one()
            c.execute(
                text("UPDATE attempts SET status='blocked',error=:e,updated=:t WHERE id=:id"),
                {"id": aid, "e": error, "t": time.time()},
            )
            c.execute(
                text("UPDATE jobs SET status='interrupted',error=:e,updated=:t WHERE id=:id"),
                {"id": row[0], "e": error, "t": time.time()},
            )
            self.emit(c, "attempt_blocked", aid, {"error": error})

    def release_reconciled(self, aid):
        with self.transaction() as c:
            a = (
                c.execute(
                    text(
                        "SELECT a.*,e.spec FROM attempts a JOIN jobs j ON j.id=a.job_id JOIN experiments e ON e.id=j.experiment_id WHERE a.id=:id"
                    ),
                    {"id": aid},
                )
                .mappings()
                .first()
            )
            if not a or a["status"] != "blocked":
                raise ValueError("Only a blocked attempt can be reconciled")
            from .worker import process_matches

            if process_matches(dict(a)):
                raise ValueError("Attempt process is still alive")
            spec = json.loads(a["spec"])
            for field in ("environment_service", "vla_service"):
                if not spec.get(field):
                    continue
                service = (
                    c.execute(text("SELECT spec,status FROM services WHERE id=:id"), {"id": spec[field]})
                    .mappings()
                    .one()
                )
                if not json.loads(service["spec"])["managed"] or service["status"] != "stopped":
                    raise ValueError(
                        "Stop all platform-owned environment and VLA services before reconciliation; external resources require operator recovery"
                    )
            c.execute(
                text("UPDATE attempts SET status=:s,updated=:t WHERE id=:id"),
                {
                    "id": aid,
                    "t": time.time(),
                    # A committed episode result remains a measured outcome,
                    # including a timeout. Cleanup must not silently repeat it.
                    "s": "completed" if a["result"] else "interrupted",
                },
            )
            c.execute(text("DELETE FROM leases WHERE attempt_id=:id"), {"id": aid})
            c.execute(
                text(
                    "UPDATE jobs SET status=CASE WHEN cancel_requested=1 THEN 'cancelled' ELSE 'interrupted' END, updated=:t WHERE id=:id"
                ),
                {"id": a["job_id"], "t": time.time()},
            )
            self.emit(
                c,
                "attempt_reconciled",
                aid,
                {"evidence": "owned motion services stopped; attempt process absent"},
            )

    def finish(self, aid, status, result=None, error=None):
        with self.transaction() as c:
            row = c.execute(text("SELECT job_id FROM attempts WHERE id=:a"), {"a": aid}).first()
            if not row:
                raise KeyError("Unknown attempt")
            c.execute(
                text("UPDATE attempts SET status=:s,result=:r,error=:e,updated=:t WHERE id=:a"),
                {
                    "s": status,
                    "r": json.dumps(result) if result else None,
                    "e": error,
                    "t": time.time(),
                    "a": aid,
                },
            )
            c.execute(text("DELETE FROM leases WHERE attempt_id=:a"), {"a": aid})
            if status != "completed":
                c.execute(
                    text("UPDATE jobs SET status='interrupted',error=:e,updated=:t WHERE id=:j"),
                    {"e": error, "t": time.time(), "j": row[0]},
                )
            self.emit(c, "attempt_" + status, aid, {"error": error})
