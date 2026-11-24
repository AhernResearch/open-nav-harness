"""Kernel episodes against a fake sim-engine session and a scripted planner."""

import asyncio
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image

from open_nav_harness.kernel.environment import READS_IN_FLIGHT, SimEngineEnvironment
from open_nav_harness.kernel.episode import PreflightError, preflight, run_episode
from open_nav_harness.kernel.spec import EnvironmentSpec, KernelSpec

PNG = io.BytesIO()
Image.new("RGB", (32, 24), (90, 90, 90)).save(PNG, format="PNG")
PNG = PNG.getvalue()
DEPTH = io.BytesIO()
np.savez_compressed(DEPTH, depth_m=np.full((24, 32), 2.0, np.float32))
DEPTH = DEPTH.getvalue()
CAMERA = {"width": 480, "height": 270, "hfov": 120.0, "height_m": 0.88, "pitch_down_rad": 0.0}
PROJECTION = [[0.5, 0, 0, 0], [0, 0.6, 0, 0], [0, 0, -1, -0.02], [0, 0, -1, 0]]


class FakeSim:
    def __init__(self, lose_step=False, reward_mode="none", dataset_max_actions=None):
        self.events, self.observations, self.steps, self.finished = 0, 0, [], None
        self.lose_step, self.reward_mode, self.legs = lose_step, reward_mode, []
        self.dataset_max_actions = dataset_max_actions or {}

    def __call__(self, request: httpx.Request):
        path, body = request.url.path, json.loads(request.content or b"{}")
        if path == "/v1/health":
            return httpx.Response(200, json={
                "workers": 2,
                "protocol": {"reward_mode": self.reward_mode, "max_actions": 500, "cameras": {"cam": CAMERA},
                             "dataset_physics": {"r2r_train": {"velocity": {"time_step_s": 0.1}}},
                             "dataset_max_actions": self.dataset_max_actions},
                "capabilities": {"commands": ["velocity", "point_goal", "retrace"], "observe": True,
                                 "retrace_targets": ["leg_id", "event_index"]}})
        if path == "/v1/sessions":
            return httpx.Response(200, json={"session_id": "s1", "event_index": 0,
                                             "instruction": "Walk to the sofa and stop."})
        if path.endswith("/observe"):
            self.observations += 1
            return httpx.Response(200, json={"observation_id": f"obs-{self.observations}",
                                             "event_index": self.events, "views": {}})
        if "/observations/" in path:
            if request.url.params.get("sensor") == "depth":
                return httpx.Response(200, content=DEPTH, headers={"X-Sensor-Metadata": json.dumps({
                    "calibration": {"projections": {"depth": PROJECTION}},
                    "pose": {"position": [0, 0.88, 0], "rotation_xyzw": [0, 0, 0, 1]}})})
            return httpx.Response(200, content=PNG)
        if path.endswith("/step"):
            self.steps.append(body)
            if self.lose_step:
                raise httpx.ReadTimeout("reply lost")
            assert body["expected_event_index"] == self.events
            start, kind = self.events, body["command"]["kind"]
            self.events += 1 if kind == "velocity" else 3
            if kind == "velocity":
                return httpx.Response(200, json={"event_index": self.events, "done": False,
                                                 "termination": None, "collisions": 0})
            leg = {"leg_id": len(self.legs) + 1, "kind": kind, "status": "reached",
                   "reason": None, "start_event_index": start, "end_event_index": self.events}
            self.legs.append(leg)
            return httpx.Response(200, json={"event_index": self.events, "done": False,
                                             "termination": None, "collisions": 0, "leg": leg})
        if "/frames/" in path:
            index = int(path.rsplit("/", 1)[1])
            return httpx.Response(200, content=bytes(8 * 6 * 3), headers={
                "X-Width": "8", "X-Height": "6", "X-Content-SHA256": "0" * 64,
                "X-Pose": json.dumps({"position": [0, 0, -0.1 * index], "rotation_xyzw": [0, 0, 0, 1]})})
        if path.endswith("/finish"):
            self.finished = body["reason"]
            return httpx.Response(200, json={})
        if path.endswith("/metrics"):
            return httpx.Response(200, json={"metrics": {"success": 1.0},
                                             "legs": [dict(leg, private={"goal": [1, 0, 2]})
                                                      for leg in self.legs]})
        if request.method == "DELETE":
            return httpx.Response(200, json={"closed": True})
        raise AssertionError(path)


class Planner:
    """Returns scripted assistant messages; records every request it receives."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def __call__(self, request: httpx.Request):
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 1000})
        self.requests.append(json.loads(request.content))
        message = self.replies.pop(0)
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


def call(name, ident, **arguments):
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def reply(*calls, content=None):
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


def spec():
    return KernelSpec.model_validate({
        "planner": {"model": "m", "base_url": "http://planner/v1", "context_limit": 32768},
        "environment": {"url": "http://sim", "dataset": "r2r_train", "camera": "cam"}})


async def run(tmp_path, sim, planner):
    return await run_episode(spec(), "4966", tmp_path / "ep", 500,
                             sim_transport=httpx.MockTransport(sim),
                             planner_transport=httpx.MockTransport(planner))


def files(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.json"))


async def test_a_delegated_leg_is_recorded_with_frames_and_the_stop_is_scored(tmp_path):
    sim = FakeSim()
    planner = Planner(
        reply(call("observe", "c1", views=["front", "right", "back", "left"]),
              call("update_plan", "c2", steps=[{"text": "go to the sofa", "done": False}])),
        reply(call("delegate", "c3", view="left", point=[500, 800], instruction="walk to the sofa")),
        reply(call("stop", "c4", status="complete", reason="at the sofa")))
    result = await run(tmp_path, sim, planner)
    root = tmp_path / "ep"
    assert result["status"] == "completed" and result["end_reason"] == "policy_stop"
    assert sim.finished == "policy_stop" and result["legs"] == 1 and result["actions"] == 3
    assert {"legs/0001.json", "legs/0001-frames.json", "evaluation.json", "result.json",
            "spec.json"} <= set(files(root))
    leg = json.loads((root / "legs/0001.json").read_text())
    assert leg["kind"] == "point_goal" and leg["executor"] == "point_expert"
    assert leg["instruction"] == "walk to the sofa" and set(leg["start_views"]) == {"front", "right",
                                                                                     "back", "left"}
    assert len(json.loads((root / "legs/0001-frames.json").read_text())) == 3
    # The model saw four images after observe and four after the leg, never private geometry.
    sent = json.dumps(planner.requests)
    assert "private" not in sent and '"goal"' not in sent
    last = planner.requests[-1]["messages"]
    assert sum(isinstance(m["content"], list) for m in last if m["role"] == "user") == 2
    assert "legs_so_far" in last[-2]["content"]
    # Frames, poses and the observed map are recorded for the delegate call, never sent.
    journal = [json.loads(line) for line in (root / "trace/requests.jsonl").read_text().splitlines()]
    evidence = {r["kind"]: r for r in journal if r["kind"] in ("leg_evidence", "map_rendered")}
    assert evidence["leg_evidence"]["call_id"] == "c3" and len(evidence["leg_evidence"]["poses"]) == 3
    assert [r["call_id"] for r in journal if r["kind"] == "map_rendered"] == ["c1", "c3"]
    assert evidence["map_rendered"]["seen_area_m2"] > 0
    assert "\"position\"" not in sent and "heading_rad" not in sent and "rotation_xyzw" not in sent


async def test_tool_calls_written_as_text_are_repaired_then_fail_after_three(tmp_path):
    sim = FakeSim()
    text = reply(content='<tool_call>\n<function=observe>\n<parameter=views>["front"]')
    text["tool_calls"] = []
    result = await run(tmp_path, sim, Planner(text, text, text))
    assert result["error"] == "planner_protocol_failure" and result["malformed_replies"] == 3
    assert sim.finished == "invalid_output" and sim.steps == []


async def test_plain_text_without_a_tool_call_is_a_recorded_failure(tmp_path):
    sim = FakeSim()
    result = await run(tmp_path, sim, Planner({"role": "assistant", "content": "I think I am done."}))
    assert result["error"] == "planner_ended_without_stop" and sim.finished == "invalid_output"


async def test_invalid_arguments_reject_the_whole_reply_before_anything_moves(tmp_path):
    sim = FakeSim()
    planner = Planner(
        reply(call("observe", "c1", views=["left"]),
              call("delegate", "c2", view="left", point=[500, 1500], instruction="go")),
        reply(call("stop", "c3", status="abort", reason="test")))
    result = await run(tmp_path, sim, planner)
    assert sim.observations == 0 and sim.steps == [] and result["malformed_replies"] == 1
    tools = [m for m in planner.requests[1]["messages"] if m["role"] == "tool"]
    assert all("No tools in this reply were executed" in m["content"] for m in tools)


async def test_delegating_into_an_unseen_view_is_refused_without_motion(tmp_path):
    sim = FakeSim()
    planner = Planner(
        reply(call("observe", "c1", views=["front"])),
        reply(call("delegate", "c2", view="back", point=[500, 800], instruction="go back")),
        reply(call("stop", "c3", status="abort", reason="test")))
    await run(tmp_path, sim, planner)
    assert sim.steps == []
    refused = [m for m in planner.requests[2]["messages"] if m["role"] == "tool"][-1]
    assert "Observe the back view" in refused["content"]


async def test_a_lost_motion_reply_is_uncertain_and_never_retried(tmp_path):
    sim = FakeSim(lose_step=True)
    planner = Planner(
        reply(call("observe", "c1", views=["front"])),
        reply(call("delegate", "c2", view="front", point=[500, 800], instruction="go")))
    result = await run(tmp_path, sim, planner)
    assert result["status"] == "failed_or_uncertain" and len(sim.steps) == 1
    assert (tmp_path / "ep/failed_or_uncertain.json").exists()
    assert not (tmp_path / "ep/result.json").exists()


async def test_preflight_refuses_a_simulator_that_reports_training_rewards():
    with pytest.raises(PreflightError, match="reward_mode"):
        await preflight(spec(), httpx.MockTransport(FakeSim(reward_mode="progress_v1")))
    assert (await preflight(spec(), httpx.MockTransport(FakeSim())))["max_actions"] == 500


async def test_preflight_takes_the_episode_budget_of_the_run_dataset():
    sim = FakeSim(dataset_max_actions={"r2r_train": 1250, "other": 300})
    assert (await preflight(spec(), httpx.MockTransport(sim)))["max_actions"] == 1250


async def test_an_episode_keeps_its_simulator_reads_within_the_shared_queue():
    """A long leg's frames are fetched in parallel, but never more than READS_IN_FLIGHT at once."""
    sim, active, peak = FakeSim(), [0], [0]

    async def handler(request):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        await asyncio.sleep(0.001)
        active[0] -= 1
        return sim(request)

    env = SimEngineEnvironment(EnvironmentSpec(url="http://sim", dataset="r2r_train", camera="cam"),
                               transport=httpx.MockTransport(handler))
    await env.open("1")
    await asyncio.gather(*map(env.frame, range(1, 60)))
    await env.close()
    assert peak[0] == READS_IN_FLIGHT
