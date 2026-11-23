"""Kernel runs: durable queue, in-process scheduling, recovery and bundle export."""

import asyncio
import hashlib
import io
import json
import tarfile

import httpx
import pytest
from test_kernel import FakeSim, call, reply

from open_nav_harness.platform import runs
from open_nav_harness.platform.api import create_app
from open_nav_harness.platform.database import Database
from open_nav_harness.platform.kernel_feed import KernelFeed, plan_state
from open_nav_harness.platform.runs import RunRequest, RunScheduler, RunStore

CONTRACT = {"max_actions": 500, "workers": 2}


def stopper(request: httpx.Request):
    """A planner that stops at once in every episode."""
    if request.url.path.endswith("/tokenize"):
        return httpx.Response(200, json={"count": 1000})
    return httpx.Response(200, json={
        "choices": [{"message": reply(call("stop", "c1", status="abort", reason="test")),
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


def request(episodes, purpose="evaluation", dataset="r2r_train", lanes=2):
    return RunRequest.model_validate({
        "name": "t", "purpose": purpose, "episodes": episodes, "lanes": lanes,
        "spec": {"planner": {"model": "m", "base_url": "http://planner/v1", "context_limit": 32768,
                             "max_concurrency": 2},
                 "environment": {"url": "http://sim", "dataset": dataset, "camera": "cam"}}})


@pytest.fixture
def store(tmp_path):
    db = Database(tmp_path / "state")
    db.migrate()
    return RunStore(db)


async def drain(scheduler, rid):
    scheduler.tick()
    await asyncio.gather(*scheduler.lanes[rid])


def scheduler(store, sim):
    return RunScheduler(store, sim_transport=httpx.MockTransport(sim),
                        planner_transport=httpx.MockTransport(stopper))


async def test_a_run_executes_every_episode_in_lanes_and_settles(store):
    rid = store.create(request(["1", "2", "3"]), CONTRACT)
    await drain(scheduler(store, FakeSim()), rid)
    run = store.run(rid)
    assert run["status"] == "completed"
    assert [e["status"] for e in run["episodes"]] == ["completed"] * 3
    summary = run["episodes"][0]["summary"]
    assert summary["end_reason"] == "policy_stop" and summary["metrics"] == {"success": 1.0}
    assert json.loads((store.folder(rid, "1") / "session.json").read_text())["session_id"] == "s1"


async def test_a_busy_simulator_requeues_before_anything_is_written(store, monkeypatch):
    monkeypatch.setattr(runs, "BUSY_BACKOFF_S", 0)
    sim, busy = FakeSim(), [True]

    def flaky(request):
        if request.url.path == "/v1/sessions" and busy:
            busy.pop()
            return httpx.Response(429, text="all simulator workers are occupied")
        return sim(request)

    rid = store.create(request(["1"], lanes=1), CONTRACT)
    await drain(scheduler(store, flaky), rid)
    # The lane took the requeued episode again after the backoff, into a fresh folder.
    assert busy == [] and store.run(rid)["episodes"][0]["status"] == "completed"


async def test_an_episode_left_running_ends_uncertain_and_its_session_is_released(store):
    rid = store.create(request(["7"]), CONTRACT)
    assert store.claim(rid) == "7"
    folder = store.folder(rid, "7")
    folder.mkdir(parents=True)
    (folder / "session.json").write_text(json.dumps({"url": "http://sim", "session_id": "s9"}))
    deleted = []

    def sim(request):
        deleted.append((request.method, request.url.path))
        return httpx.Response(200, json={"closed": True})

    await scheduler(store, sim).recover()
    episode = store.run(rid)["episodes"][0]
    assert episode["status"] == "failed_or_uncertain" and "HTTP 200" in episode["error"]
    assert deleted == [("DELETE", "/v1/sessions/s9")] and store.run(rid)["status"] == "completed"


async def test_cancel_drops_queued_episodes_and_settles_the_run(store):
    rid = store.create(request(["1", "2"]), CONTRACT)
    store.cancel(rid)
    run = store.run(rid)
    assert run["status"] == "cancelled" and {e["status"] for e in run["episodes"]} == {"cancelled"}
    assert store.claim(rid) is None


async def test_a_changed_kernel_fails_the_queued_episodes(store, monkeypatch):
    rid = store.create(request(["1"]), CONTRACT)
    monkeypatch.setattr(runs, "kernel_digest", lambda: "changed")
    await drain(scheduler(store, FakeSim()), rid)
    episode = store.run(rid)["episodes"][0]
    assert episode["status"] == "failed" and "kernel source changed" in episode["error"]


def test_collection_requires_a_train_dataset(store):
    with pytest.raises(ValueError, match="train"):
        store.create(request(["1"], purpose="collection", dataset="r2r_val_unseen"), CONTRACT)
    with pytest.raises(ValueError, match="unique"):
        request(["1", "1"])


async def test_a_finished_episode_downloads_as_a_verified_bundle(store, tmp_path):
    rid = store.create(request(["1"]), CONTRACT)
    await drain(scheduler(store, FakeSim()), rid)
    app = create_app(tmp_path / "state")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
        listing = (await http.get("/api/runs")).json()
        assert listing[0]["episodes"] == {"completed": 1}
        response = await http.get(f"/api/runs/{rid}/episodes/1/bundle")
    assert hashlib.sha256(response.content).hexdigest() == response.headers["x-content-sha256"]
    names = tarfile.open(fileobj=io.BytesIO(response.content)).getnames()
    assert {"1/result.json", "1/evaluation.json", "1/session.json"} <= set(names)


async def test_the_kernel_feed_replays_an_episode_bundle_into_item_events(store):
    rid = store.create(request(["1"]), CONTRACT)
    await drain(scheduler(store, FakeSim()), rid)
    feed = KernelFeed(store.folder(rid, "1"))
    events = feed.poll()
    kinds = [e["type"] for e in events]
    assert kinds[0] == "session.started" and events[0]["instruction"] == "Walk to the sofa and stop."
    assert kinds[-1] == "session.ended" and events[-1]["explicit_stop"]
    assert {"turn.completed", "item.started", "item.completed"} <= set(kinds)
    # A reconnect from the cursor repeats only the idempotent end.
    assert [e["type"] for e in KernelFeed(feed.folder, feed.cursor).poll()] == ["session.started",
                                                                                 "session.ended"]


def test_plan_state_marks_the_first_unfinished_step_in_progress():
    state = plan_state([{"text": "a", "done": True}, {"text": "b", "done": False},
                        {"text": "c", "done": False}])
    assert [t["status"] for t in state["tasks"]] == ["completed", "in_progress", "pending"]
