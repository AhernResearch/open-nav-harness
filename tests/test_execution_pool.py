"""One job can fill independent lanes, without duplicate episodes or shared motion."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text
from nav_harness.core.config import ExperimentSpec, ServiceSpec, PlannerProfile
from nav_harness.platform.database import Database


@pytest.fixture
def pool(tmp_path):
    db = Database(tmp_path)
    db.migrate()
    planner = db.register_service(
        ServiceSpec(
            name="planner",
            kind="planner",
            url="http://planner:8000/v1",
            planner=PlannerProfile(model="test", base_url="http://planner:8000/v1", max_concurrency=4),
        )
    )
    vla = db.register_service(
        ServiceSpec(
            name="vla",
            kind="vla",
            url="http://vla:8000",
            max_concurrency=4,
            metadata={"shared_inference": True},
        )
    )
    bindings = []
    for i in range(4):
        env = db.register_service(ServiceSpec(name=f"env{i}", kind="environment", url=f"http://env{i}:8000"))
        worker = db.register_service(
            ServiceSpec(
                name=f"worker{i}",
                kind="execution",
                url=f"http://worker{i}:8000",
                metadata={"environment_url": f"http://env{i}:8000", "inference_url": "http://vla:8000"},
            )
        )
        bindings.append({"environment_service": env, "execution_service": worker})
    spec = ExperimentSpec(
        name="parallel",
        planner_service=planner,
        vla_service=vla,
        execution_pool=bindings,
        episodes=list(range(8)),
        config={"dataset": {"name": "r2r", "split": "train"}, "harness": {"mode": "agentic"}},
    )
    eid = db.create_experiment(spec)
    jid = db.enqueue(eid)
    yield db, jid, spec
    db.engine.dispose()


def test_concurrent_claims_fill_one_job_without_duplicates(pool):
    db, jid, spec = pool
    with ThreadPoolExecutor(8) as executor:
        claimed = [a for a in executor.map(lambda _: db.claim(), range(8)) if a]
    assert len(claimed) == 4
    assert {a["episode"] for a in claimed} == set(range(4))
    assert len({a["spec"]["execution_service"] for a in claimed}) == 4
    assert len(db.rows("SELECT * FROM leases")) == 16
    frozen = json.loads(db.rows("SELECT spec FROM experiments")[0]["spec"])
    assert frozen["environment_service"] is None  # Attempt binding never mutates experiment.
    for attempt in claimed:
        binding = json.loads(
            db.rows("SELECT binding FROM attempts WHERE id=:id", id=attempt["id"])[0]["binding"]
        )
        assert binding["execution_service"] == attempt["spec"]["execution_service"]
        assert (
            attempt["spec"]["config"]["harness"]["execution_url"]
            == binding["services"]["execution_service"]["url"]
        )
    db.finish(claimed[0]["id"], "completed", {"success": 1})
    assert db.claim()["episode"] == 4


def test_pause_drains_all_active_attempts_then_resume_excludes_completed(pool):
    db, jid, _ = pool
    active = [db.claim() for _ in range(4)]
    db.control(jid, "pause")
    assert db.claim() is None
    for a in active[:-1]:
        db.finish(a["id"], "completed", {"success": 0})
    assert db.claim() is None
    assert db.rows("SELECT status FROM jobs")[0]["status"] == "running"
    db.finish(active[-1]["id"], "completed", {"success": 0})
    assert db.claim() is None
    assert db.rows("SELECT status FROM jobs")[0]["status"] == "paused"
    db.control(jid, "resume")
    assert db.claim()["episode"] == 4


def test_terminal_job_waits_for_last_attempt_and_cancel_drains(pool):
    db, jid, _ = pool
    first = [db.claim() for _ in range(4)]
    for a in first:
        db.finish(a["id"], "completed", {"success": 1})
    last = [db.claim() for _ in range(4)]
    for a in last[:-1]:
        db.finish(a["id"], "completed", {"success": 1})
    assert db.claim() is None
    assert db.rows("SELECT status FROM jobs")[0]["status"] == "running"
    db.control(jid, "cancel")
    assert db.claim() is None
    db.finish(last[-1]["id"], "completed", {"success": 1})
    assert db.claim() is None
    assert db.rows("SELECT status FROM jobs")[0]["status"] == "cancelled"


def test_one_blocked_lane_preserves_all_its_leases_and_other_attempts_can_finish(pool):
    db, jid, _ = pool
    a, b = db.claim(), db.claim()
    db.block(a["id"], "response lost")
    db.finish(b["id"], "completed", {"success": 0})
    assert len(db.rows("SELECT * FROM leases")) == 4
    assert db.claim() is None
    assert db.rows("SELECT status FROM jobs")[0]["status"] == "interrupted"


def test_cancel_does_not_mark_unknown_motion_terminal(pool):
    db, jid, _ = pool
    attempt = db.claim()
    db.block(attempt['id'], 'unknown motion')
    db.control(jid, 'cancel')
    row = db.rows('SELECT * FROM jobs WHERE id=:id', id=jid)[0]
    assert row['status'] == 'interrupted' and row['cancel_requested']
    assert db.rows('SELECT * FROM leases')


async def test_reconciliation_closes_only_failed_actor_session_and_waits_for_gpu(pool, monkeypatch):
    import httpx
    from pathlib import Path
    from nav_harness.core.config import PlatformSettings
    from nav_harness.platform.services import ServiceManager
    db, jid, spec = pool
    failed, healthy = db.claim(), db.claim()
    db.block(failed['id'], 'lost response')
    binding = json.loads(db.rows('SELECT binding FROM attempts WHERE id=:id', id=failed['id'])[0]['binding'])
    with db.transaction() as connection:
        for key in ('environment_service', 'execution_service'):
            sid = binding[key]
            service = db.service(sid).model_dump()
            service.update(managed=True, template='test')
            connection.execute(text("UPDATE services SET spec=:s,status='stopped' WHERE id=:id"),
                               {'id': sid, 's': json.dumps(service)})
    receipt = Path(failed['output'])/'episode/live/execution_session.json'
    receipt.parent.mkdir(parents=True)
    ident = 'a'*32
    receipt.write_text(json.dumps({'session_id': ident}))
    drained = False
    requests = []
    def handle(request):
        requests.append((request.method, request.url.path))
        assert request.url.path.startswith('/sessions/'+ident)
        return httpx.Response(409 if request.url.path.endswith('/close') and not drained else 200, json={})
    client_type = httpx.AsyncClient
    monkeypatch.setattr('nav_harness.platform.services.httpx.AsyncClient',
        lambda **kwargs: client_type(transport=httpx.MockTransport(handle), **kwargs))
    manager = ServiceManager(db, PlatformSettings())
    with pytest.raises(ValueError, match='still running'):
        await manager.reconcile_attempt(failed['id'])
    assert len(db.rows('SELECT * FROM leases')) == 8
    drained = True
    await manager.reconcile_attempt(failed['id'])
    leases = db.rows('SELECT * FROM leases')
    assert len(leases) == 4 and {r['attempt_id'] for r in leases} == {healthy['id']}
    assert db.rows('SELECT status FROM services WHERE id=:id', id=spec.vla_service)[0]['status'] == 'registered'
