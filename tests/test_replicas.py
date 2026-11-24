"""Planner replica pinning: least busy first, one replica per episode, slots shared across runs."""

import asyncio
import json

import httpx
import pytest
from test_runs import CONTRACT, FakeSim, drain, request, stopper, store  # noqa: F401

from open_nav_harness.core.config import PlannerProfile
from open_nav_harness.kernel.replicas import ReplicaPool
from open_nav_harness.platform.runs import RunRequest, RunScheduler

A, B, C = "http://a/v1", "http://b/v1", "http://c/v1"


def profile(**fields):
    return PlannerProfile(model="m", base_url=A, **fields)


async def test_each_episode_takes_the_least_busy_replica_and_waits_when_all_are_full():
    pool, held = ReplicaPool(), []
    for _ in range(3):
        pin = pool.pin(profile(replicas=[B, C]))
        held.append((pin, await pin.__aenter__()))
    assert [p.base_url for _, p in held] == [A, B, C]
    waiting = asyncio.create_task(pool.pin(profile(replicas=[B, C])).__aenter__())
    await asyncio.sleep(0)
    assert not waiting.done()
    await held[1][0].__aexit__(None, None, None)
    assert (await waiting).base_url == B


async def test_runs_sharing_replicas_share_their_slots_and_a_profile_without_replicas_is_unchanged():
    pool = ReplicaPool()
    async with pool.pin(profile(replicas=[B], max_concurrency=2)) as first:
        async with pool.pin(profile(replicas=[B], max_concurrency=2)) as second:
            assert (first.base_url, second.base_url) == (A, B)
            async with pool.pin(profile(max_concurrency=2)) as single:
                assert single.base_url == A and single.replicas == []
        assert pool.busy == {(A, "m"): 1, (B, "m"): 0}


def test_replicas_must_be_plain_http_endpoints():
    with pytest.raises(ValueError):
        profile(replicas=["http://user:secret@b/v1"])
    assert profile(replicas=[A, B]).endpoints() == [A, B]


async def test_a_run_records_the_replica_each_episode_was_pinned_to(store):  # noqa: F811
    hosts = set()

    def planner(req):
        hosts.add(req.url.host)
        return stopper(req)

    spec = request(["1", "2", "3", "4"]).model_dump()
    spec["spec"]["planner"].update(max_concurrency=1, replicas=["http://planner-b/v1"])
    rid = store.create(RunRequest.model_validate(spec), CONTRACT)
    await drain(RunScheduler(store, sim_transport=httpx.MockTransport(FakeSim()),
                             planner_transport=httpx.MockTransport(planner)), rid)
    episodes = store.run(rid)["episodes"]
    used = [e["summary"]["planner_url"] for e in episodes]
    assert set(used) == {"http://planner/v1", "http://planner-b/v1"} and hosts == {"planner", "planner-b"}
    for e in episodes:
        frozen = json.loads((store.folder(rid, e["episode"]) / "spec.json").read_text())
        assert frozen["spec"]["planner"]["base_url"] == e["summary"]["planner_url"]
