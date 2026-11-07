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
        {"id": 1, "instruction": "Cross the doorway", "completion_criterion": "Beyond the door"},
        {"id": 2, "instruction": "Turn left into the hall", "completion_criterion": "Facing the hall"},
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
    m.review("continue", [], "Door crossing remains uncertain.", "Need another view")
    assert m.last_review["map_fresh"] is False
    m.finish("complete")
    assert m.snapshot()["completion_claim"] == "complete"
    assert any(t["status"] != "accepted" for t in m.tasks)


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
    assert schema["required"] == ["task_id"]
    strict = NavigationSession(env={"NAVGPT_CONDITION": "planner_subtask"})
    assert strict.tools.definitions["execute_subtask"].inputSchema["properties"] == {}
    assert "observe_node" in session.tools.definitions and "observe_node" not in strict.tools.definitions
    prompt = build_briefing_for("planner_agentic", "Exit the room", 500)
    assert "no review call is required" in prompt and "Exit the room" in prompt
    data = {"planner": {"runtime": "codex"}, "harness": {"mode": "agentic"}}
    config = Settings(Path("test.json"), data, {}).run_config()
    assert config.condition == "planner_agentic" and config.vla_map_enabled()
    with pytest.raises(SettingsError):
        Settings(Path("test.json"), {"harness": {"mode": "agentic"}}, {}).run_config()


def test_repair_is_not_blocked_by_task_lifecycle_and_limits_restore(monkeypatch):
    s = NavigationSession(env={"NAVGPT_CONDITION": "planner_agentic"})
    s._tasks = manager()
    calls = []
    monkeypatch.setattr(
        s, "vla_navigate", lambda instruction: calls.append((instruction, dict(s._rollout_limits))) or []
    )
    s.execute_agentic_task(1, max_distance_m=8, max_actions=120, reset_context=True)
    assert calls[0][1]["max_distance_m"] == 8 and calls[0][1]["max_actions"] == 120
    assert s._rollout_limits is None
    s._tasks.needs_review = True
    s._tasks.needs_replan = True
    s.execute_agentic_task(1, instruction="Finish crossing the threshold")
    assert len(calls) == 2
    with pytest.raises(ValueError):
        s.execute_agentic_task(1, max_actions=501)
    assert len(calls) == 2


def test_protocol_feedback_does_not_swallow_failure_after_motion_dispatch(monkeypatch):
    s = NavigationSession(env={'NAVGPT_CONDITION': 'planner_agentic'})
    def bad_reply(*args):
        s._motion_epoch += 1
        raise ValueError('invalid movement response')
    # Wrappers resolve the captured bound function; force its environment call to fail.
    monkeypatch.setattr(s, '_call', bad_reply)
    with pytest.raises(RuntimeError, match='after dispatching movement'):
        s.tools.invoke('navigate_relative', {'turn_deg': 0.0, 'distance_m': 1.0})
