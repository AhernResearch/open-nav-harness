"""Replicated planner pools: health needs every replica; the launcher owns them together."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from open_nav_harness.core.config import PlannerProfile, PlatformSettings, ServiceSpec
from open_nav_harness.platform.database import Database
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
    yield db, planner
    db.engine.dispose()


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
