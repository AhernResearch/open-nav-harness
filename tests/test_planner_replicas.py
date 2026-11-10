"""Replicated planner pools: each episode is pinned to one replica, replicas fill evenly."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from open_nav_harness.core.config import ExperimentSpec, PlannerProfile, PlatformSettings, ServiceSpec
from open_nav_harness.platform.database import Database, pick_slot
from open_nav_harness.platform.services import ServiceManager

REPLICAS = ["http://a100:18110/v1", "http://a100:18111/v1", "http://a100:18112/v1"]
LAUNCHER = Path(__file__).resolve().parents[1] / "deployment/vllm_replicas.py"


@pytest.fixture
def pool(tmp_path):
    db = Database(tmp_path)
    db.migrate()
    planner = db.register_service(
        ServiceSpec(
            name="qwen-replicas",
            kind="planner",
            url=REPLICAS[0],
            planner=PlannerProfile(model="qwen", base_url=REPLICAS[0], max_concurrency=12),
            metadata={"replica_urls": REPLICAS},
        )
    )
    vla = db.register_service(
        ServiceSpec(name="vla", kind="vla", url="http://vla:8000", max_concurrency=4,
                    metadata={"shared_inference": True})
    )
    bindings = []
    for i in range(4):
        env = db.register_service(ServiceSpec(name=f"env{i}", kind="environment", url=f"http://env{i}:8000"))
        worker = db.register_service(
            ServiceSpec(name=f"worker{i}", kind="execution", url=f"http://worker{i}:8000",
                        metadata={"environment_url": f"http://env{i}:8000", "inference_url": "http://vla:8000"})
        )
        bindings.append({"environment_service": env, "execution_service": worker})
    spec = ExperimentSpec(
        name="replicas", planner_service=planner, vla_service=vla, execution_pool=bindings,
        episodes=list(range(8)),
        config={"dataset": {"name": "r2r", "split": "train"}, "harness": {"mode": "agentic"}},
    )
    db.enqueue(db.create_experiment(spec))
    yield db, planner
    db.engine.dispose()


def test_pick_slot_prefers_least_occupied_replica():
    assert pick_slot(12, 3, [True] * 12) == 0
    assert pick_slot(12, 3, [False, True, True] + [True] * 9) == 1
    # Replica 1 frees up while 0 and 2 still hold an episode each.
    assert pick_slot(12, 3, [False, True, False, False] + [True] * 8) == 1
    assert pick_slot(4, 0, [False, True, True, True]) == 1
    assert pick_slot(2, 3, [False, False]) is None


def test_claims_pin_each_episode_to_a_balanced_replica(pool):
    db, planner = pool
    claimed = [db.claim() for _ in range(4)]
    urls = [a["spec"]["config"]["planner"]["profile"]["base_url"] for a in claimed]
    assert urls[:3] == REPLICAS and urls[3] == REPLICAS[0]
    events = db.rows("SELECT payload FROM events WHERE kind='attempt_started' ORDER BY seq")
    assert [json.loads(e["payload"])["planner_url"] for e in events] == urls
    frozen = json.loads(db.rows("SELECT spec FROM experiments")[0]["spec"])
    assert frozen["config"]["planner"]["profile"]["base_url"] == REPLICAS[0]  # experiment stays frozen
    db.finish(claimed[1]["id"], "completed", {"success": 1})
    assert db.claim()["spec"]["config"]["planner"]["profile"]["base_url"] == REPLICAS[1]


async def test_replicated_planner_is_healthy_only_when_every_replica_answers(pool, monkeypatch):
    db, planner = pool
    services = ServiceManager(db, PlatformSettings())

    async def get(self, url, headers):
        status = 503 if url.startswith(REPLICAS[2]) else 200
        return httpx.Response(status, json={"data": [{"id": "qwen"}]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    assert (await services.health(planner))["ok"] is False
    ok = AsyncMock(return_value=httpx.Response(200, json={"data": [{"id": "qwen"}]},
                                               request=httpx.Request("GET", REPLICAS[0])))
    monkeypatch.setattr(httpx.AsyncClient, "get", ok)
    assert (await services.health(planner))["ok"] is True
    assert [c.args[0] for c in ok.await_args_list] == [u + "/models" for u in REPLICAS]


def test_launcher_stops_all_replicas_when_one_exits(tmp_path):
    # Each fake replica records its GPU/port; the one on port 2 exits at once.
    child = ("import os,sys,time,pathlib; port=sys.argv[-1]; "
             f"pathlib.Path({str(tmp_path)!r},port).write_text(os.environ['CUDA_VISIBLE_DEVICES']); "
             "time.sleep(0 if port=='2' else 60)")
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), "--ports", "1,2,3", "--", sys.executable, "-c", child],
        env={"CUDA_VISIBLE_DEVICES": "0,3,4"}, timeout=30,
    )
    assert result.returncode == 1
    assert {p.name: p.read_text() for p in tmp_path.iterdir()} == {"1": "0", "2": "3", "3": "4"}


def test_launcher_rejects_mismatched_ports():
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), "--ports", "1,2", "--", "vllm", "serve"],
        env={"CUDA_VISIBLE_DEVICES": "0,3,4"}, capture_output=True, timeout=30,
    )
    assert result.returncode != 0
