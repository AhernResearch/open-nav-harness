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


def test_api_auth_csrf_and_unknown_templates(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as c:
        assert c.get("/api/services").status_code == 401
        password = (tmp_path / "admin-password").read_text()
        assert c.post("/api/login", json={"password": password}).status_code == 200
        assert c.get("/api/services").status_code == 200
        assert c.post("/api/logout").status_code == 403
        assert c.post("/api/logout", headers={"Origin": "http://testserver"}).status_code == 200
        assert c.get("/api/services").status_code == 401
        assert c.get("/api/services", headers={"Authorization": "Bearer " + password}).status_code == 200


def test_session_state_and_tool_contract_are_isolated(monkeypatch):
    monkeypatch.setenv("NAVGPT_CONDITION", "planner_subtask")
    from navgpt.planner.tools.session import NavigationSession
    from navgpt.planner.runner import expected_tools

    a, b = NavigationSession(), NavigationSession()
    a._steps_taken = 9
    assert b._steps_taken == 0
    assert set(a.mcp._tool_manager._tools) == set(expected_tools("planner_subtask"))


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
