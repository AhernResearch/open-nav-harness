import asyncio
import base64
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from nav_harness.core.artifacts import ArtifactStore, Journal
from nav_harness.core.config import (
    ExperimentSpec,
    PlannerProfile,
    PlatformSettings,
    ServiceSpec,
    ServiceTemplate,
)
from nav_harness.core.context import ContextOverflow, ContextWindow
from nav_harness.platform.api import create_app
from nav_harness.platform.database import Database
from nav_harness.platform.results import read_jsonl
from nav_harness.platform.services import ServiceManager
from nav_harness.platform.worker import Worker
from nav_harness.providers.compatible import CompatibleClient


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path)
    d.migrate()
    yield d
    d.engine.dispose()


def experiment(db, *, env=None, managed=False, purpose="development", split="train"):
    env = env or db.register_service(
        ServiceSpec(
            name="env",
            kind="environment",
            url="http://localhost:19200",
            managed=managed,
            template="test" if managed else None,
            gpu_ids=[0] if managed else [],
            reserve_mib=1024 if managed else 0,
        )
    )
    vla = db.register_service(ServiceSpec(name="vla", kind="vla", url="http://localhost:18000"))
    planner = db.register_service(
        ServiceSpec(
            name="planner",
            kind="planner",
            url="http://localhost:18001/v1",
            planner=PlannerProfile(model="test", base_url="http://localhost:18001/v1"),
        )
    )
    return db.create_experiment(
        ExperimentSpec(
            name="smoke",
            environment_service=env,
            vla_service=vla,
            planner_service=planner,
            episodes=[0, 1],
            purpose=purpose,
            config={"dataset": {"name": "r2r", "split": split}, "harness": {"mode": "subtask"}},
        )
    ), env


def test_serialized_claim_and_completed_episodes_never_rerun(db):
    eid, _ = experiment(db)
    jid = db.enqueue(eid)
    with ThreadPoolExecutor(4) as pool:
        claims = list(pool.map(lambda _: db.claim(), range(4)))
    claims = [c for c in claims if c]
    assert len(claims) == 1
    db.finish(claims[0]["id"], "completed", {"task_success": 0})
    second = db.claim()
    assert second["episode"] == 1
    db.finish(second["id"], "completed", {"task_success": 1})
    assert db.claim() is None
    assert db.rows("SELECT status FROM jobs WHERE id=:id", id=jid)[0]["status"] == "completed"


def test_distinct_jobs_cannot_share_environment(db):
    eid, _ = experiment(db)
    db.enqueue(eid)
    db.enqueue(eid)
    assert db.claim()
    assert db.claim() is None


def test_pause_and_resume_are_durable(db):
    eid, _ = experiment(db)
    jid = db.enqueue(eid)
    db.control(jid, "pause")
    assert db.claim() is None
    db.control(jid, "resume")
    assert db.claim()
    with pytest.raises(ValueError):
        db.control(jid, "resume")


def test_unclean_worker_retains_leases(db):
    eid, _ = experiment(db)
    jid = db.enqueue(eid)
    a = db.claim()
    Worker(db, PlatformSettings()).reconcile()
    assert db.rows("SELECT status FROM attempts")[0]["status"] == "blocked"
    assert db.rows("SELECT * FROM leases")
    with pytest.raises(ValueError, match="Reconcile"):
        db.control(jid, "resume")
    with pytest.raises(ValueError, match="platform-owned"):
        db.release_reconciled(a["id"])


def test_worker_recovers_committed_result_without_rerun(db):
    eid, _ = experiment(db)
    db.enqueue(eid)
    a = db.claim()
    root = Path(a["output"])
    root.mkdir(parents=True)
    (root / "result.json").write_text('{"task_success":0}')
    Worker(db, PlatformSettings()).reconcile()
    assert not db.rows("SELECT * FROM leases")
    assert db.claim()["episode"] == 1


def test_managed_service_not_ready_blocks_claim(db):
    eid, env = experiment(db, managed=True)
    db.enqueue(eid)
    assert db.claim() is None
    with db.transaction() as c:
        c.execute(text("UPDATE services SET status='ready' WHERE id=:id"), {"id": env})
    assert db.claim()


def test_collection_refuses_held_out_split(db):
    with pytest.raises(ValueError, match="train split"):
        experiment(db, purpose="collection", split="val_unseen")


async def test_external_services_cannot_be_stopped(db):
    _, env = experiment(db)
    manager = ServiceManager(db, PlatformSettings())
    manager.ssh = AsyncMock()
    with pytest.raises(ValueError, match="External"):
        await manager.operate(env, "stop")
    manager.ssh.assert_not_called()


async def test_active_lease_blocks_service_stop(db):
    eid, env = experiment(db, managed=True)
    with db.transaction() as c:
        c.execute(text("UPDATE services SET status='ready' WHERE id=:id"), {"id": env})
    db.enqueue(eid)
    db.claim()
    manager = ServiceManager(
        db,
        PlatformSettings(
            service_templates={
                "test": ServiceTemplate(host="a100", argv=["python"], working_directory="/tmp")
            }
        ),
    )
    manager.ssh = AsyncMock()
    with pytest.raises(ValueError, match="leased"):
        await manager.operate(env, "stop")
    manager.ssh.assert_not_called()


def test_lossless_media_roundtrip_and_corruption(tmp_path):
    store = ArtifactStore(tmp_path)
    value = {"image_url": {"url": "data:image/png;base64," + base64.b64encode(b"image").decode()}}
    frozen = store.freeze(value)
    assert store.thaw(frozen) == value
    (tmp_path / "media" / frozen["image_url"]["url"]["artifact"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash"):
        store.thaw(frozen)


def test_journal_never_overwrites_and_rejects_corrupt_line(tmp_path):
    journal = Journal(tmp_path)
    journal.emit("request", password="secret")
    assert read_jsonl(journal.path)[0]["password"] == "[REDACTED]"
    with pytest.raises(FileExistsError):
        Journal(tmp_path)
    with journal.path.open("a") as f:
        f.write("{")
    assert len(read_jsonl(journal.path)) == 1
    with journal.path.open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError):
        read_jsonl(journal.path)


async def test_context_prunes_atomic_blocks_preserves_current_state(tmp_path):
    w = ContextWindow("system", "task", "bounded", 1500, 128)
    w.append(
        [{"role": "assistant", "content": "old"}, {"role": "tool", "content": "old result"}], {"current": "a"}
    )
    w.append(
        [{"role": "assistant", "content": "new"}, {"role": "tool", "content": "new result"}], {"current": "b"}
    )

    async def counter(messages, tools):
        return 1300 if any(m.get("content") == "old" for m in messages) else 600

    messages = await w.build(counter, [], Journal(tmp_path))
    assert not any(m.get("content") == "old" for m in messages)
    assert [m["content"] for m in messages[-2:]] == ["new", "new result"]
    assert '"current": "b"' in messages[2]["content"]


async def test_context_overflow_without_safe_summary(tmp_path):
    w = ContextWindow("system", "task", "bounded", 2048, 128)
    w.append([{"role": "user", "content": "history"}])

    async def counter(*_):
        return 2000

    with pytest.raises(ContextOverflow):
        await w.build(counter, [], Journal(tmp_path))


async def test_wire_requests_and_model_retry_are_recorded(tmp_path):
    p = PlannerProfile(model="test")
    client = CompatibleClient(p)
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]},
        )

    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    journal = Journal(tmp_path)
    await client.generate([{"role": "user", "content": "hello"}], [], journal)
    await client.close()
    events = read_jsonl(journal.path)
    assert events[0]["request"] == seen[0]
    assert events[1]["kind"] == "model_response"


def test_api_open_access_and_unknown_templates(tmp_path):
    app = create_app(tmp_path)
    assert not (tmp_path / "admin-password").exists()
    with TestClient(app) as c:
        assert c.get("/api/services").status_code == 200
        spec = {"name": "test-env", "kind": "environment", "url": "http://localhost:19200"}
        response = c.post("/api/services", json=spec)
        assert response.status_code == 201
        assert c.get("/api/services").json()[0]["id"] == response.json()["id"]
        assert c.post(
            "/api/services", json={**spec, "managed": True, "template": "unknown"}
        ).status_code == 422
        assert "/api/login" not in app.openapi()["paths"]
        assert "/api/logout" not in app.openapi()["paths"]


def test_session_state_and_tool_contract_are_isolated(monkeypatch):
    monkeypatch.setenv("NAVGPT_CONDITION", "planner_subtask")
    from navgpt.planner.tools.session import NavigationSession
    from navgpt.planner.runner import expected_tools

    a, b = NavigationSession(), NavigationSession()
    a._steps_taken = 9
    assert b._steps_taken == 0
    assert set(a.tools.definitions) == set(expected_tools("planner_subtask"))


def test_place_annotation_and_map_feedback_use_episode_local_labels(monkeypatch):
    monkeypatch.setenv("NAVGPT_CONDITION", "planner_subtask")
    from navgpt.planner.tools.session import NavigationSession

    a, b = NavigationSession(), NavigationSession()
    payload = {
        "nearest_place": 0,
        "you_are_at_xy": [0, 0],
        "places": [
            {
                "id": i,
                "xy": [distance, 0],
                "straight_m": distance,
                "route_m": distance,
                "visits": 1,
                "links": [],
            }
            for i, distance in [(0, 0), (1, 0.5), (2, 4)]
        ],
    }
    for session in (a, b):
        monkeypatch.setattr(session, "_fetch_places", lambda: payload)
        monkeypatch.setattr(session, "_call", lambda *args: {"position": [0, 0, 0]})
        monkeypatch.setattr(session, "_dump_labels", lambda: None)
    first = a.add_place("kitchen", "by the island")
    assert json.loads(first[0])["recorded_place"] == 0
    assert '"kitchen"' in first[1]
    assert '"kitchen"' in a._places_line()
    assert a._pending_place(payload) == 2
    assert a._label_target(payload) == 1  # Nearby unnamed node, not the farthest/newest one.
    assert b._label_target(payload) == 0
    assert '"kitchen"' not in b._format_places(payload)


def test_lost_motion_response_blocks_even_if_metrics_written(db):
    eid, _ = experiment(db)
    db.enqueue(eid)
    a = db.claim()
    Worker(db, PlatformSettings()).complete(a, {"task_success": 0, "resource_state": "unknown"})
    assert db.rows("SELECT status FROM attempts")[0]["status"] == "blocked"
    assert db.rows("SELECT * FROM leases")


def test_motion_recovery_requires_acknowledged_result():
    from nav_harness.core.recovery import resource_state

    events = [{"kind": "tool_dispatch", "call_id": "a", "name": "execute_subtask"}]
    assert resource_state(events, wire=True) == "unknown"
    assert (
        resource_state(events + [{"kind": "tool_result", "call_id": "a", "is_error": True}], wire=True)
        == "unknown"
    )
    assert (
        resource_state(events + [{"kind": "tool_result", "call_id": "a", "is_error": False}], wire=True)
        == "idle"
    )


def test_paused_job_can_cancel_without_a_free_worker_slot(db):
    eid, _ = experiment(db)
    jid = db.enqueue(eid)
    db.control(jid, "pause")
    assert db.rows("SELECT status FROM jobs WHERE id=:id", id=jid)[0]["status"] == "paused"
    db.control(jid, "cancel")
    assert db.rows("SELECT status FROM jobs WHERE id=:id", id=jid)[0]["status"] == "cancelled"
    assert db.claim() is None


def test_reconciliation_finishes_pending_cancellation(db):
    eid, _ = experiment(db, managed=True)
    with db.transaction() as c:
        rows = c.execute(text("SELECT id,spec FROM services")).mappings().all()
        for row in rows:
            spec = json.loads(row["spec"])
            if spec["kind"] in ("environment", "vla"):
                spec.update(managed=True, template="test")
                c.execute(
                    text("UPDATE services SET spec=:s,status='ready' WHERE id=:id"),
                    {"s": json.dumps(spec), "id": row["id"]},
                )
    jid = db.enqueue(eid)
    a = db.claim()
    db.control(jid, "cancel")
    db.block(a["id"], "uncertain motion")
    with db.transaction() as c:
        c.execute(
            text(
                "UPDATE services SET status='stopped' WHERE json_extract(spec,'$.kind') IN ('environment','vla')"
            )
        )
    db.release_reconciled(a["id"])
    assert db.rows("SELECT status FROM jobs WHERE id=:id", id=jid)[0]["status"] == "cancelled"
    assert not db.rows("SELECT * FROM leases")


def test_comparison_counts_completed_errors_without_metrics_as_failures():
    from nav_harness.platform.results import aggregate

    rows = [
        {"episode": 0, "status": "completed", "success": 1, "error": None},
        {"episode": 1, "status": "completed", "success": None, "error": "protocol error"},
        {"episode": 1, "status": "completed", "success": 1, "error": None},
        {"episode": 2, "status": "interrupted", "success": None, "error": "worker exited"},
    ]
    result = aggregate(rows)
    assert result["routes_completed"] == 2
    assert result["execution_errors"] == 1
    assert result["sr"] == 0.5


def test_reconciliation_preserves_committed_timeout_as_completed(db):
    eid, _ = experiment(db, managed=True)
    with db.transaction() as c:
        rows = c.execute(text("SELECT id,spec FROM services")).mappings().all()
        for row in rows:
            spec = json.loads(row["spec"])
            if spec["kind"] in ("environment", "vla"):
                spec.update(managed=True, template="test")
                c.execute(
                    text("UPDATE services SET spec=:s,status='ready' WHERE id=:id"),
                    {"s": json.dumps(spec), "id": row["id"]},
                )
    jid = db.enqueue(eid)
    a = db.claim()
    from nav_harness.platform.worker import Worker
    from nav_harness.core.config import PlatformSettings

    Worker(db, PlatformSettings()).complete(
        a, {"error": "episode_timeout", "task_success": 0, "resource_state": "unknown"}
    )
    with db.transaction() as c:
        c.execute(
            text(
                "UPDATE services SET status='stopped' WHERE json_extract(spec,'$.kind') IN ('environment','vla')"
            )
        )
    db.release_reconciled(a["id"])
    assert db.rows("SELECT status FROM attempts WHERE id=:id", id=a["id"])[0]["status"] == "completed"
    with db.transaction() as c:
        c.execute(text("UPDATE services SET status='ready'"))
    db.control(jid, "resume")
    next_attempt = db.claim()
    assert next_attempt is None or next_attempt["episode"] != a["episode"]
