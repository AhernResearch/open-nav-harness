"""The LightNav actor contract and the VLA executor, against a fake model and a fake simulator.

The real actor service runs in-process behind an ASGI transport; only the model is scripted.
"""

import json
import math

import httpx
import pytest
from test_kernel import FakeSim, Planner, call, reply

from open_nav_harness.actors.lightnav import Control, LightNavService, Prediction, create_app, velocity_action
from open_nav_harness.kernel.episode import PreflightError, preflight, run_episode
from open_nav_harness.kernel.spec import KernelSpec

STILL = [[0.0, 0.0, 0.0]] * 10
FORWARD = [[0.2, 0.0, 0.0]] + [[0.4, 0.0, 0.0]] * 9


def forward(stop=False, decode_error=None, rows=None):
    return Prediction(rows or FORWARD, stop, "<act>", decode_error)


class FakeModel:
    """Scripted predictions; records the history length each prediction saw."""

    identity = {"name": "fake"}
    camera = {"width": 480, "height": 270, "hfov": 120.0, "height_m": 0.88, "pitch_down_rad": 0.0}

    def __init__(self, *script, fail=False):
        self.script, self.seen, self.opened, self.frames, self.fail = list(script), [], 0, [], fail

    async def start(self): ...
    async def stop(self): ...

    def open(self):
        self.opened += 1
        return []

    async def append(self, handle, png):
        handle.append(png)

    async def predict(self, handle, instruction):
        if self.fail:
            raise RuntimeError("cuda error")
        self.seen.append((len(handle), instruction))
        return self.script.pop(0) if self.script else forward()


def actor(model, max_sessions=8):
    service = LightNavService(model, Control(), max_sessions)
    return service, httpx.ASGITransport(app=create_app(service), raise_app_exceptions=False)


def vla_spec(**budgets):
    return KernelSpec.model_validate({
        "tool_set": "vla_v1",
        "planner": {"model": "m", "base_url": "http://planner/v1", "context_limit": 32768},
        "environment": {"url": "http://sim", "dataset": "r2r_train", "camera": "cam"},
        "executor": {"kind": "vla", "url": "http://actor"}, "budgets": budgets})


async def run(tmp_path, sim, planner, model, **budgets):
    service, transport = actor(model)
    result = await run_episode(vla_spec(**budgets), "4966", tmp_path / "ep", 500,
                               sim_transport=httpx.MockTransport(sim),
                               planner_transport=httpx.MockTransport(planner), actor_transport=transport)
    return result, service


def delegate(ident, text="walk to the sofa"):
    return reply(call("delegate", ident, instruction=text))


def stop(ident):
    return reply(call("stop", ident, status="complete", reason="done"))


def test_the_first_moving_row_becomes_a_clamped_unicycle_velocity():
    rows = [[0.0, 0.0, 0.0], [0.1, 0.05, math.radians(6)]] + FORWARD[:8]
    action = velocity_action(forward(rows=rows), Control(dt_s=0.1))
    assert action["linear_mps"] == pytest.approx(1.0) and action["stop_reason"] is None
    assert math.degrees(action["angular_rps"]) == pytest.approx(60)
    fast = velocity_action(forward(rows=[[9.0, 0, math.radians(90)]] * 10), Control())
    assert fast["linear_mps"] == 2.5 and math.degrees(fast["angular_rps"]) == pytest.approx(300)
    assert velocity_action(forward(rows=[[-0.2, 0, 0]] + FORWARD[1:]), Control())["stop_reason"] == "low_speed"


def test_stop_is_arrival_but_a_decode_error_is_not():
    assert velocity_action(forward(stop=True), Control())["stop_reason"] == "model_stop"
    assert velocity_action(forward(rows=STILL), Control())["stop"] is True
    bad = velocity_action(forward(rows=STILL, decode_error="no act tokens"), Control())
    assert bad["stop"] is False and bad["stop_reason"] == "decode_error"
    with pytest.raises(ValueError):
        velocity_action(forward(rows=FORWARD[:3]), Control())


async def test_service_history_is_contiguous_idempotent_and_acts_once_per_request_id():
    model = FakeModel()
    _, transport = actor(model, max_sessions=1)
    async with httpx.AsyncClient(transport=transport, base_url="http://a") as http:
        opened = (await http.post("/v1/sessions")).json()["session_id"]
        assert (await http.post("/v1/sessions")).status_code == 429
        base, frame = f"/v1/sessions/{opened}", "AAAA"
        assert (await http.post(base + "/act", json={"request_id": "r", "instruction": "go"})).status_code == 409
        assert (await http.post(base + "/frames", json={"first_index": 1, "frames": [frame]})).status_code == 409
        assert (await http.post(base + "/frames", json={"first_index": 0, "frames": [frame, frame]})).json() == {
            "history_frames": 2}
        assert (await http.post(base + "/frames", json={"first_index": 1, "frames": [frame]})).json() == {
            "history_frames": 2}
        assert (await http.post(base + "/frames", json={"first_index": 1, "frames": ["BBBB"]})).status_code == 409
        body = {"request_id": "r", "instruction": "go"}
        first = (await http.post(base + "/act", json=body)).json()
        assert (await http.post(base + "/act", json=body)).json() == first and len(model.seen) == 1
        assert (await http.post(base + "/act", json={**body, "instruction": "stop"})).status_code == 409
        assert (await http.delete(base)).json()["history_frames"] == 2
        assert (await http.post(base + "/act", json=body)).status_code == 404
        assert (await http.get("/v1/health")).json()["history"] == "full_episode"


async def test_a_leg_walks_until_the_policy_stops_and_the_actor_sees_every_frame(tmp_path):
    sim, model = FakeSim(), FakeModel(forward(), forward(), forward(stop=True))
    planner = Planner(reply(call("observe", "c1", views=["front"])), delegate("c2"), stop("c3"))
    result, service = await run(tmp_path, sim, planner, model)
    assert result["status"] == "completed" and result["actions"] == 2 and not service.sessions
    assert [s["command"]["kind"] for s in sim.steps] == ["velocity", "velocity"]
    assert sim.steps[0]["command"]["linear_mps"] == pytest.approx(2.0)
    assert model.seen == [(1, "walk to the sofa"), (2, "walk to the sofa"), (3, "walk to the sofa")]
    leg = json.loads((tmp_path / "ep/legs/0001.json").read_text())
    assert leg["kind"] == "vla" and leg["status"] == "reached" and leg["leg_id"] == 1
    assert leg["executor"] == "vla" and leg["end_event_index"] == 2
    assert len(json.loads((tmp_path / "ep/legs/0001-frames.json").read_text())) == 2
    journal = (tmp_path / "ep/trace/requests.jsonl").read_text()
    assert journal.count('"kind": "actor_act"') == 3 and '"raw_text": "<act>"' in journal
    assert json.loads((tmp_path / "ep/actor.json").read_text())["session_id"]


async def test_the_leg_budget_ends_the_leg_after_exactly_that_many_steps(tmp_path):
    sim, model = FakeSim(), FakeModel()
    planner = Planner(delegate("c1"), stop("c2"))
    await run(tmp_path, sim, planner, model, leg_max_actions=4)
    assert len(sim.steps) == 4 and len(model.seen) == 4
    leg = json.loads((tmp_path / "ep/legs/0001.json").read_text())
    assert leg["status"] == "leg_budget" and leg["collision_steps"] == 0
    assert leg["distance_m"] > 0 and leg["net_displacement_m"] > 0


async def test_a_decode_error_ends_the_leg_without_arrival_or_motion(tmp_path):
    sim = FakeSim()
    model = FakeModel(forward(rows=STILL, decode_error="no act tokens"))
    await run(tmp_path, sim, Planner(delegate("c1"), stop("c2")), model)
    leg = json.loads((tmp_path / "ep/legs/0001.json").read_text())
    assert leg["status"] == "decode_error" and sim.steps == []


async def test_returns_and_their_frames_stay_in_the_actor_history(tmp_path):
    sim, model = FakeSim(), FakeModel(forward(), forward(), forward(stop=True), forward(), forward(stop=True))
    planner = Planner(delegate("c1"), reply(call("return_to", "c2", leg_id=1)), delegate("c3"), stop("c4"))
    result, _ = await run(tmp_path, sim, planner, model, leg_max_actions=5)
    kinds = [s["command"] for s in sim.steps]
    # leg 1 ends at event 2; the retrace goes to that event (the harness names legs, not the simulator)
    assert kinds[2] == {"kind": "retrace", "event_index": 2, "max_actions": 5}
    # velocity x2, retrace +3 events, then the next act sees all 6 frames of the continuous episode
    assert [length for length, _ in model.seen][-2:] == [6, 7]
    assert result["status"] == "completed"
    retrace = json.loads((tmp_path / "ep/legs/0002.json").read_text())
    assert retrace["kind"] == "retrace" and retrace["leg_id"] == 2 and retrace["retrace_of"] == 1


async def test_a_lost_velocity_reply_is_uncertain_never_retried_and_closes_the_actor(tmp_path):
    sim, model = FakeSim(lose_step=True), FakeModel()
    result, service = await run(tmp_path, sim, Planner(delegate("c1")), model)
    assert result["status"] == "failed_or_uncertain" and len(sim.steps) == 1
    assert len(model.seen) == 1 and model.opened == 1 and not service.sessions


async def test_an_actor_failure_records_the_leg_so_far_fails_the_episode_and_closes_the_session(tmp_path):
    sim, model = FakeSim(), FakeModel(fail=True)
    result, service = await run(tmp_path, sim, Planner(delegate("c1")), model)
    assert result["status"] == "failed" and "actor" in result["error"].lower()
    assert not service.sessions
    assert not (tmp_path / "ep/result.json").exists()
    assert json.loads((tmp_path / "ep/legs/0001.json").read_text())["status"] == "actor_error"


async def test_preflight_requires_matching_control_period_and_camera():
    service, transport = actor(FakeModel())
    contract = await preflight(vla_spec(), httpx.MockTransport(FakeSim()), transport)
    assert contract["actor"]["control"]["dt_s"] == 0.1
    service.control = Control(dt_s=0.25)
    with pytest.raises(PreflightError, match="time step"):
        await preflight(vla_spec(), httpx.MockTransport(FakeSim()), transport)


def test_a_tool_set_needs_its_executor():
    with pytest.raises(ValueError, match="needs executor"):
        KernelSpec.model_validate({**vla_spec().model_dump(), "tool_set": "point_v1"})
