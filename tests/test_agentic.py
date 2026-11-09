"""Planner-owned decisions preserve causal history and the strict baseline."""

import base64
import json
from pathlib import Path

import pytest

from navgpt.planner.tools.agentic import AgenticTasks
from navgpt.planner.tools.session import NavigationSession
from navgpt.planner.runner import expected_tools
from navgpt.planner.prompts import build_briefing_for
from navgpt.settings import Settings, SettingsError


def tasks():
    return [
        {
            "id": 1,
            "instruction": "Cross the doorway",
            "status": "pending",
        },
        {
            "id": 2,
            "instruction": "Turn left into the hall",
            "status": "pending",
        },
    ]


def manager():
    m = AgenticTasks("Exit the room then turn left")
    m.observe({"policy_front": "Zg=="}, {}, 0)
    m.update_plan(tasks(), "Initial decomposition")
    return m


def test_model_can_continue_repair_reorder_and_stop_without_review_gates():
    m = manager()
    m.begin(1)
    m.mark_motion()
    m.observe({"policy_front": "Zw=="}, {}, 1)
    m.returned("budget")
    m.begin(1, "Cross the threshold immediately ahead")
    assert m.policy_instruction().startswith("Cross the threshold immediately ahead")
    m.update_plan(tasks(), "Refined local landmarks")
    m.begin(2)
    assert m.current()["id"] == 2
    m.memory.begin({"position": [0, 0, 0]}, "f0", "execute_subtask")
    m.memory.finish({"position": [0, 0, 0]}, "f1", "vla_stop")
    m.remember(0, 0, "Door crossing remains uncertain.", {"map_call": 1})
    assert "uncertain" in m.policy_instruction()
    m.finish("complete")
    assert m.snapshot()["completion_claim"] == "complete"
    assert any(t["status"] != "completed" for t in m.tasks)


@pytest.mark.parametrize("budget,boundary", [(1, 0), (2, 1), (4, 3), (16, 2)])
def test_actor_context_always_contains_current_frame_within_budget(budget, boundary):
    m = manager()
    for i in range(20):
        m.observe({"policy_front": base64.b64encode(str(i).encode()).decode()}, {}, i)
    m.begin(2)
    for i in range(20, 40):
        m.observe({"policy_front": base64.b64encode(str(i).encode()).decode()}, {}, i)
    m.max_frames, m.boundary_frames = budget, boundary
    frames = m.selected_frames()
    assert len(frames) <= budget
    assert frames[-1] == m.frames[-1]
    assert len({f["id"] for f in frames}) == len(frames)
    assert all(m.frames.index(f) >= m.anchor - boundary for f in frames)


def test_agentic_tools_and_prompt_match_without_altering_strict_contract(tmp_path):
    session = NavigationSession(env={"NAVGPT_CONDITION": "planner_agentic"})
    assert set(session.tools.definitions) == set(expected_tools("planner_agentic"))
    schema = session.tools.definitions["execute_subtask"].inputSchema
    assert schema["required"] == ["task_id", "max_distance_m"]
    strict = NavigationSession(env={"NAVGPT_CONDITION": "planner_subtask"})
    assert strict.tools.definitions["execute_subtask"].inputSchema["properties"] == {}
    assert "observe" in session.tools.definitions and "observe_node" not in session.tools.definitions
    prompt = build_briefing_for("planner_agentic", "Exit the room", 500)
    assert "review_subtask" not in prompt and "Exit the room" in prompt
    data = {"planner": {"runtime": "codex"}, "harness": {"mode": "agentic"}}
    config = Settings(Path("test.json"), data, {}).run_config()
    assert config.condition == "planner_agentic" and config.vla_map_enabled()
    with pytest.raises(SettingsError):
        Settings(Path("test.json"), {"harness": {"mode": "agentic"}}, {}).run_config()


@pytest.mark.parametrize("reason,stopped", [("vla_stop", True), ("turn_budget", False)])
def test_actor_handoff_is_not_presented_as_arrival_verification(reason, stopped):
    original = ["endpoint camera", json.dumps({
        "subtask_return": reason, "vla_suggests_arrival": stopped,
        "arrival_note": "Not verified arrival", "distance_travelled_m": 0.5,
        "turn_travelled_deg": 30, "episode_over": False,
    })]
    result = NavigationSession._compact_motion_result(original)
    status = json.loads(result[1])
    assert status == {"subtask_return": reason, "distance_travelled_m": 0.5,
                      "turn_travelled_deg": 30, "episode_over": False}
    assert result[0] == "endpoint camera"
    assert "vla_suggests_arrival" in json.loads(original[1])


def test_repair_is_not_blocked_by_task_lifecycle_and_limits_restore(monkeypatch):
    s = NavigationSession(env={"NAVGPT_CONDITION": "planner_agentic"})
    s._tasks = manager()
    calls = []
    monkeypatch.setattr(
        s, "vla_navigate", lambda instruction: calls.append(instruction) or []
    )
    s.execute_agentic_task(1, 5.0, reset_context=True)
    assert calls[0].startswith("Cross the doorway")
    assert s._rollout_limits is None
    s._tasks.needs_review = True
    s._tasks.needs_replan = True
    s.execute_agentic_task(1, 5.0, instruction="Finish crossing the threshold")
    assert len(calls) == 2
    with pytest.raises(TypeError):
        s.execute_agentic_task(1, 5.0, max_actions=501)
    assert len(calls) == 2


@pytest.mark.parametrize("allow_reset", [True, False])
def test_execution_surface_and_reset_ablation(allow_reset):
    from pydantic import ValidationError
    from navgpt.planner.runner import mcp_env_for

    cfg = Settings(Path("test.json"), {
        "planner": {"runtime": "codex"},
        "harness": {"mode": "agentic", "allow_context_reset": allow_reset},
    }, {}).run_config()
    session = NavigationSession(env=mcp_env_for(cfg, None))
    tool = session.tools.definitions["execute_subtask"]
    fields = {"task_id", "max_distance_m", "instruction"} | ({"reset_context"} if allow_reset else set())
    assert set(tool.inputSchema["properties"]) == fields
    assert tool.inputSchema["required"] == ["task_id", "max_distance_m"]
    for key in ["max_actions", "max_turn_deg"] + ([] if allow_reset else ["reset_context"]):
        with pytest.raises(ValidationError):
            session.tools.invoke("execute_subtask", {"task_id": 1, 'max_distance_m': 5.0, key: True if key == "reset_context" else 1})
    prompt = build_briefing_for("planner_agentic", "Enter", 500, allow_context_reset=allow_reset)
    assert "Enter" in prompt
    assert ("reset_context" in tool.description) is allow_reset


def test_explicit_reset_only_changes_same_task_anchor_when_requested():
    m = manager()
    m.begin(1)
    initial = m.anchor
    m.observe({"policy_front": "Zw=="}, {}, 1)
    m.begin(1, "Continue through door")
    assert m.anchor == initial
    m.begin(1, "Try again", reset_context=True)
    assert m.anchor == 1
    m.observe({"policy_front": "aA=="}, {}, 2)
    m.begin(2)
    assert m.anchor == 2  # Automatic task switch survives both experiment variants.


@pytest.mark.parametrize("ending", ["policy", "blocked", "global"])
def test_agentic_rollout_uses_global_protection_not_old_segment_caps(monkeypatch, ending):
    import math
    from types import SimpleNamespace

    count = 0
    pose = {"position": [0., 0., 0.], "rotation_wxyz": [1., 0., 0., 0.]}
    def act(**kwargs):
        return {"should_stop": ending == "policy" and count == 90,
                "velocity": {"linear_mps": 1., "angular_rps": 1., "dt_s": .1}}

    s = NavigationSession(env={"NAVGPT_CONDITION": "planner_agentic", "NAVGPT_VLA_BACKEND": "lightnav"},
                          actor=SimpleNamespace(act=act, prepare_context=lambda *args: None))
    s._tasks = manager()
    monkeypatch.setattr(s, "_pano", lambda: {"pose": pose, "views": {}, "raw_views": {}})
    monkeypatch.setattr(s, "_clearance_summary", lambda *args: {})
    monkeypatch.setattr(s, "_pano_frames", lambda *args: [])
    monkeypatch.setattr(s, "_frame_here", lambda: None)
    def move(name, inputs):
        nonlocal count, pose
        count += 1
        pose = {"position": [count * .1, 0., 0.],
                "rotation_wxyz": [math.cos(count * .2), 0., math.sin(count * .2), 0.]}
        return {"pose": pose, "step_count": count, "navigable": ending != "blocked",
                "truncated": ending == "global" and count == 90}
    monkeypatch.setattr(s, "_call", move)
    result = s.execute_agentic_task(1, 100.0)  # far beyond this walk: no turn cap applies
    status = next(json.loads(x) for x in result if isinstance(x, str) and '"subtask_return"' in x)
    assert count == (3 if ending == "blocked" else 90)
    if ending == "policy":
        assert status["subtask_return"] == "vla_stop"
        assert status["distance_travelled_m"] > 3
        assert status["turn_travelled_deg"] > 180
    elif ending == "blocked":
        assert status["subtask_return"] == "blocked"
    else:
        assert status["episode_over"]


def test_protocol_feedback_does_not_swallow_failure_after_motion_dispatch(monkeypatch):
    s = NavigationSession(env={"NAVGPT_CONDITION": "planner_agentic"})

    def bad_reply(*args):
        s._motion_epoch += 1
        raise ValueError("invalid movement response")

    s._last_pose = {"position": [0, 0, 0]}
    # Wrappers resolve the captured bound function; force its environment call to fail.
    monkeypatch.setattr(s, "_call", bad_reply)
    with pytest.raises(RuntimeError, match="after dispatching movement"):
        s.tools.invoke("navigate_relative", {"turn_deg": 0.0, "distance_m": 1.0})


@pytest.mark.parametrize("backend,horizon", [("navgpt", 8), ("lightnav", 10)])
def test_rollout_records_actor_output_before_dispatch_and_matches_execution(
    tmp_path, monkeypatch, backend, horizon
):
    from types import SimpleNamespace

    navigation = [[float(i), 0.0, 0.0] for i in range(horizon)]
    advice = {
        "navigation": navigation,
        "should_stop": False,
        "position": [1.0, 0.0, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    if backend == "lightnav":
        advice["velocity"] = {"linear_mps": 1.0, "angular_rps": 0.0, "dt_s": 0.1}
    actor = SimpleNamespace(prepare_context=lambda *args: None, act=lambda **kwargs: advice)
    s = NavigationSession(
        env={
            "NAVGPT_CONDITION": "planner_agentic",
            "NAVGPT_VLA_BACKEND": backend,
            "NAVGPT_LIVE_DIR": str(tmp_path),
            "NAVGPT_VLA_MAP": "0",
        },
        actor=actor,
    )
    s._tasks.observe({"policy_front": "Zg=="}, {}, 0)
    s._tasks.update_plan(tasks(), "Observed scene")
    pose = {"position": [0.0, 0.0, 0.0], "rotation_wxyz": [1.0, 0.0, 0.0, 0.0]}
    monkeypatch.setattr(s, "_pano", lambda: {"views": {}, "raw_views": {}, "pose": pose})
    monkeypatch.setattr(s, "_clearance_summary", lambda *args: {})
    monkeypatch.setattr(s, "_pano_frames", lambda *args: [])
    monkeypatch.setattr(s, "_frame_here", lambda: None)

    def events():
        return [json.loads(line) for line in (tmp_path / "subtasks/events.jsonl").read_text().splitlines()]

    def move(name, inputs):
        assert events()[-1]["kind"] == "policy_execution_dispatch"
        assert events()[-2]["advice"]["navigation"] == navigation
        assert name.endswith("__step_velocity" if backend == "lightnav" else "__teleport")
        return {"pose": pose, "step_count": 1, "navigable": True}

    monkeypatch.setattr(s, "_call", move)
    s.VLA_MAX_STEPS = 1  # Runtime test cap; not a planner-selected tool argument.
    s.execute_agentic_task(1, 5.0)
    step = [e for e in events() if e.get("prediction_id") == 1]
    assert [e["kind"] for e in step] == [
        "policy_context",
        "policy_prediction",
        "policy_execution_dispatch",
        "policy_execution_result",
    ]
    assert step[1]["advice"]["navigation"] == navigation
    assert step[1]["waypoint_count"] == horizon and step[1]["resampled"] is False
    assert step[-1]["feedback"]["step_count"] == 1


def test_instruction_only_plan_contract_and_actor_prompt():
    from pydantic import ValidationError
    session = NavigationSession(env={"NAVGPT_CONDITION": "planner_agentic"})
    session._tasks = manager()
    schema = session.tools.definitions['update_plan'].inputSchema
    assert set(schema['$defs']['PlanItem']['properties']) == {'id', 'instruction', 'status'}
    for field in ('completion_criterion', 'assessment'):
        with pytest.raises(ValidationError):
            session.tools.invoke('update_plan', {'subtasks': [{**tasks()[0], field: 'Invented condition'}],
                                                 'reason': 'Old field should be rejected'})
    session._tasks.begin(1)
    assert session._tasks.policy_instruction() == tasks()[0]['instruction']
    session._tasks.begin(1, 'Cross the doorway and stop inside the room')
    assert session._tasks.policy_instruction() == 'Cross the doorway and stop inside the room'
    assert session._tasks.tasks[0]['instruction'] == tasks()[0]['instruction']
    prompt = build_briefing_for('planner_agentic', 'Exit the room', 500)
    assert 'completion_criterion' not in prompt and 'completion criterion' not in prompt
    assert 'completion_criterion' not in str(session._tasks.snapshot())


def test_no_relative_ablation_freezes_surface_and_matches_briefing():
    from navgpt.settings import AGENTIC_TOOLS
    from navgpt.planner.runner import mcp_env_for

    selected = [name for name in AGENTIC_TOOLS if name != 'navigate_relative']
    config = Settings(Path('test.json'), {
        'planner': {'runtime': 'codex', 'tools': selected},
        'harness': {'mode': 'agentic'},
    }, {}).run_config()
    session = NavigationSession(env=mcp_env_for(config, live_dir=None))
    assert set(session.tools.definitions) == set(selected)
    assert {'execute_subtask', 'navigate_to_node', 'update_memory'} <= set(session.tools.functions)
    with pytest.raises(ValueError, match='Unknown navigation tool'):
        session.tools.invoke('navigate_relative', {'turn_deg': 0.0, 'distance_m': 0.5})
    prompt = build_briefing_for(config.condition, 'Find the stairs', 500, tools=config.tools)
    assert 'navigate_relative' not in prompt
    assert 'execute_subtask for forward progress and local repairs' in prompt
    assert 'navigate_relative' in NavigationSession(env={'NAVGPT_CONDITION': 'planner_agentic'}).tools.functions
    # Other accidental omissions are still configuration errors, not silent ablations.
    with pytest.raises(SettingsError):
        Settings(Path('test.json'), {
            'planner': {'runtime': 'codex', 'tools': [n for n in selected if n != 'update_memory']},
            'harness': {'mode': 'agentic'},
        }, {}).run_config()


class _Reply:
    def __init__(self, status, body):
        self.status_code, self._body, self.reason, self.text = status, body, "reason", json.dumps(body)

    def json(self):
        return self._body


@pytest.mark.parametrize("reply,error,moved", [
    (_Reply(400, {"error": "too far from the travelled path"}), "MotionRejected", False),
    (_Reply(200, {"outputs": {"error": "episode already done"}}), "MotionRejected", False),
    (_Reply(500, {"error": "worker crashed"}), "RuntimeError", True),
    (ConnectionError("reply lost"), "ConnectionError", True),
    (_Reply(200, {"outputs": {"pose": {"position": [0, 0, 0]}}}), None, True),
])
def test_only_unknown_motion_outcomes_invalidate_map_reads(monkeypatch, reply, error, moved):
    from navgpt.planner.tools import session as module

    s = NavigationSession(env={"NAVGPT_CONDITION": "planner_agentic"})
    s._memory_map_read = {"motion_epoch": 0}

    def post(*args, **kwargs):
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(module.requests, "post", post)
    try:
        s._call("env__retrace_path", {})
        raised = None
    except Exception as exc:  # noqa: BLE001
        raised = type(exc).__name__
    assert raised == error
    assert (s._motion_epoch == 1) is moved
    assert (s._memory_map_read is None) is moved


def test_rejected_motion_is_repairable_and_leaves_no_memory_edge(monkeypatch):
    from navgpt.planner.tools.session import MotionRejected

    s = NavigationSession(env={"NAVGPT_CONDITION": "planner_agentic"})

    def rejected(*args):
        raise MotionRejected("too far from the travelled path")

    s._last_pose = {"position": [0, 0, 0]}
    monkeypatch.setattr(s, "_call", rejected)
    result = s.tools.invoke("navigate_relative", {"turn_deg": 0.0, "distance_m": 1.0})
    assert result.isError
    assert json.loads(result.content[0].text)["error"] == "precondition_failed"
    assert s._tasks.memory.active is None and s._tasks.memory.edges == []


@pytest.mark.parametrize("ending", ["stop_called", "max_steps", "already_over"])
def test_completion_is_declared_only_after_confirmed_stop(monkeypatch, ending):
    s = NavigationSession(env={"NAVGPT_CONDITION": "planner_agentic"})
    s._episode_over = ending == "already_over"

    def arrive(where):
        if ending != "already_over":
            s._episode_over, s._end_reason = True, ending
        return [json.dumps({"episode_over": True, "end_reason": ending})]

    monkeypatch.setattr(s, "arrive", arrive)
    s.finish_agentic_episode("complete", "the bed is in front")
    assert (s._tasks.completion_claim == "complete") is (ending == "stop_called")


def test_agentic_delegation_stops_at_the_planner_distance_cap(monkeypatch):
    import math
    from types import SimpleNamespace

    count = 0
    pose = {"position": [0., 0., 0.], "rotation_wxyz": [1., 0., 0., 0.]}
    s = NavigationSession(env={"NAVGPT_CONDITION": "planner_agentic", "NAVGPT_VLA_BACKEND": "lightnav"},
                          actor=SimpleNamespace(act=lambda **k: {"should_stop": False, "velocity": {
                              "linear_mps": 1., "angular_rps": 0., "dt_s": .1}}, prepare_context=lambda *a: None))
    s._tasks = manager()
    monkeypatch.setattr(s, "_pano", lambda: {"pose": pose, "views": {}, "raw_views": {}})
    monkeypatch.setattr(s, "_clearance_summary", lambda *args: {})
    monkeypatch.setattr(s, "_pano_frames", lambda *args: [])
    monkeypatch.setattr(s, "_frame_here", lambda: None)

    def move(name, inputs):
        nonlocal count, pose
        count += 1
        pose = {"position": [count * .25, 0., 0.], "rotation_wxyz": [1., 0., 0., 0.]}
        return {"pose": pose, "step_count": count, "navigable": True}

    monkeypatch.setattr(s, "_call", move)
    for cap in (math.nan, 0.0, -1.0):
        with pytest.raises(ValueError, match="max_distance_m"):
            s.execute_agentic_task(1, cap)
    status = next(json.loads(x) for x in s.execute_agentic_task(1, 1.0)
                  if isinstance(x, str) and '"subtask_return"' in x)
    assert count == 4 and status["subtask_return"] == "distance_budget" and status["max_distance_m"] == 1.0
    assert s._distance_cap is None


def test_stall_return_policy_is_validated_and_frozen_into_the_session():
    from navgpt.planner.runner import mcp_env_for

    def config(value):
        return Settings(Path("test.json"), {"planner": {"runtime": "codex"},
                                            "harness": {"mode": "agentic", "stall_return": value}}, {}).run_config()
    with pytest.raises(SettingsError):
        config("nearest")
    session = NavigationSession(env=mcp_env_for(config("any"), None))
    assert session._stall.policy == "any"
