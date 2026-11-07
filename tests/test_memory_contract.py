"""Plan/memory v1: evidence gating, node identity, transport parity and persistence."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from navgpt.planner.tools.session import NavigationSession
from navgpt.planner.tools.mcp_tools import build_mcp
from navgpt.planner.tools.transport import RemoteTools
from navgpt.vla.worker import ActorWorker
from nav_harness.core.artifacts import Journal
from nav_harness.core.context import ContextWindow


def configure(session, monkeypatch):
    session._last_pose = {"position": [9, 0, 9]}
    session._tasks.observe({"policy_front": "Zg=="}, session._last_pose, 0)
    monkeypatch.setattr(session, "_pano", lambda: {})
    monkeypatch.setattr(session, "_fetch_places", lambda: {"places": []})
    monkeypatch.setattr(session, "_bump_obs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        session,
        "_fetch_map",
        lambda: (
            b"map pixels",
            {
                "seen_area_m2": 2,
                "revisiting_earlier_position": False,
                "places_px": {"3": [20, 30], "4": [40, 30]},
            },
        ),
    )
    return session


@pytest.fixture
def session(tmp_path, monkeypatch):
    s = NavigationSession(
        env={
            "NAVGPT_CONDITION": "planner_agentic",
            "NAVGPT_LIVE_DIR": str(tmp_path / "live"),
            "NAVGPT_PLACE_LABELS": str(tmp_path / "labels.json"),
        }
    )
    return configure(s, monkeypatch)


def text(result):
    return "\n".join(c.text for c in result.content if c.type == "text")


def test_exact_tool_schemas_and_legacy_names_are_not_exposed(session):
    tools = session.tools
    assert "annotate_node" not in tools.definitions and "review_subtask" not in tools.definitions
    schema = tools.definitions["update_memory"].inputSchema
    assert set(schema["properties"]) == {"node_id", "caption"}
    assert set(schema["required"]) == {"node_id", "caption"}
    for value in ({"node_id": "3", "caption": "door"}, {"node_id": 3, "caption": "door", "name": "extra"}):
        with pytest.raises(ValidationError):
            tools.invoke("update_memory", value)
    plan = [{"id": 1, "instruction": "Enter", "completion_criterion": "Beyond door", "status": "in_progress"}]
    tools.invoke("update_plan", {"subtasks": plan, "reason": "Observed doorway"})
    before = copy.deepcopy(session._tasks.snapshot())
    for bad in [{**plan[0], "status": "active"}, {**plan[0], "unexpected": 1}]:
        with pytest.raises(ValidationError):
            tools.invoke("update_plan", {"subtasks": [bad], "reason": "Invalid"})
    result = tools.invoke("update_plan", {"subtasks": plan + [{**plan[0], "id": 2}], "reason": "Invalid"})
    assert "At most one" in text(result)
    assert session._tasks.snapshot() == before


def test_only_explicit_successful_map_grants_node_writes(session, monkeypatch):
    tools = session.tools
    args = {"node_id": 3, "caption": "Doorway; crossing remains uncertain."}
    assert "Call observe_map" in text(tools.invoke("update_memory", args))
    session._tasks.mark_map()  # Automatic execution maps are not explicit reads.
    assert "Call observe_map" in text(tools.invoke("update_memory", args))
    tools.invoke("observe_map", {})
    assert "node_id must" in text(tools.invoke("update_memory", {**args, "node_id": 99}))
    assert not session._tasks.memories
    tools.invoke("update_memory", args)
    tools.invoke("update_memory", {"node_id": 4, "caption": "Another doorway on the same map."})
    assert set(session._tasks.memories) == {3, 4}
    monkeypatch.setattr(session, "_fetch_map", lambda: (None, "renderer unavailable"))
    tools.invoke("observe_map", {})
    assert "Call observe_map" in text(tools.invoke("update_memory", args))
    assert session._tasks.memory_revision == 2


@pytest.mark.parametrize(
    "motion,inputs",
    [
        ("step_hightolow", {"angle_rad": 1.57, "distance_m": 0}),
        ("step_hightolow", {"angle_rad": 0, "distance_m": 0.5}),
        ("retrace_to", {"place": 3}),
        ("step_velocity", {}),
        ("teleport", {}),
    ],
)
def test_every_motion_including_rotation_invalidates_map(session, monkeypatch, motion, inputs):
    session.tools.invoke("observe_map", {})
    reply = SimpleNamespace(status_code=200, json=lambda: {"outputs": {}})
    monkeypatch.setattr("navgpt.planner.tools.session.requests.post", lambda *args, **kwargs: reply)
    session._call("r2rce__" + motion, inputs)
    result = session.tools.invoke("update_memory", {"node_id": 3, "caption": "New claim"})
    assert "Call observe_map" in text(result)
    assert session._tasks.memory_revision == 0


def test_even_vla_stop_without_motion_requires_new_map(session, monkeypatch):
    session._tasks.update_plan(
        [{"id": 1, "instruction": "Enter", "completion_criterion": "Beyond door"}], "Plan"
    )
    session.tools.invoke("observe_map", {})
    monkeypatch.setattr(session, "vla_navigate", lambda instruction: [])
    session.execute_agentic_task(1)
    assert "Call observe_map" in text(
        session.tools.invoke("update_memory", {"node_id": 3, "caption": "Door"})
    )


def test_memory_replaces_caption_preserves_geometry_and_does_not_complete_tasks(session):
    session._tasks.update_plan(
        [{"id": 1, "instruction": "Enter", "completion_criterion": "Beyond door"}], "Plan"
    )
    session._place_labels[3] = {"name": "old node", "anchor": [1, 0, 2], "frame": "old.png", "caption": "Old"}
    tasks = copy.deepcopy(session._tasks.tasks)
    session.tools.invoke("observe_map", {})
    session.tools.invoke("update_memory", {"node_id": 3, "caption": "Door crossing remains uncertain."})
    session.tools.invoke("update_memory", {"node_id": 3, "caption": "Corrected: bathroom doorway."})
    label = session._place_labels[3]
    assert label["anchor"] == [1, 0, 2] and label["frame"] == "old.png"
    assert label["caption"] == "Corrected: bathroom doorway."
    assert session._tasks.tasks == tasks
    assert session._tasks.memory_revision == 2 and len(session._tasks.memories) == 1
    assert "bathroom" in session._tasks.policy_instruction()
    events = [
        json.loads(line) for line in (session._tasks.directory / "events.jsonl").read_text().splitlines()
    ]
    writes = [e for e in events if e["kind"] == "memory_update"]
    assert writes[-1]["previous"]["caption"] == "Door crossing remains uncertain."
    assert writes[-1]["memory"]["map_call"] == 1
    assert json.loads(Path(session.LABELS_PATH).read_text())["3"]["caption"] == label["caption"]
    before = copy.deepcopy(session._tasks.snapshot())
    assert "1-800" in text(session.tools.invoke("update_memory", {"node_id": 3, "caption": " "}))
    assert session._tasks.snapshot() == before


async def test_memory_and_plan_survive_context_pruning(session, tmp_path):
    session.tools.invoke(
        "update_plan",
        {
            "subtasks": [
                {
                    "id": 1,
                    "instruction": "Find closet",
                    "completion_criterion": "At closet doorway",
                    "status": "in_progress",
                }
            ],
            "reason": "Observed room",
        },
    )
    session.tools.invoke("observe_map", {})
    session.tools.invoke("update_memory", {"node_id": 3, "caption": "Right approach failed twice."})
    window = ContextWindow("system", "original route", "bounded", 1500, 100)
    window.append([{"role": "user", "content": "OLD"}])
    window.append([{"role": "user", "content": "CURRENT"}], session._tasks.snapshot())

    async def count(messages, _):
        return 2000 if any(m["content"] == "OLD" for m in messages) else 300

    messages = await window.build(count, [], Journal(tmp_path / "trace"))
    assert "Right approach failed twice" in json.dumps(messages)
    assert "At closet doorway" in json.dumps(messages)
    assert not any(m["content"] == "OLD" for m in messages)


async def test_mcp_rpc_uses_same_memory_gate_and_does_not_fence_recoverable_rejection(tmp_path, monkeypatch):
    actor = SimpleNamespace(info=lambda: {"backend": "lightnav"})
    worker = ActorWorker(actor, "lightnav", tmp_path / "remote", "http://fixed-env")
    remote = RemoteTools(
        {
            "NAVGPT_CONDITION": "planner_agentic",
            "NAVGPT_VLA_BACKEND": "lightnav",
            "NAVGPT_EXECUTION_URL": "http://worker",
            "NAVGPT_LIVE_DIR": str(tmp_path / "local"),
        }
    )
    remote.request = worker.handle
    remote.start()
    configure(worker.session, monkeypatch)
    server = build_mcp(remote)
    result = await server.call_tool("update_memory", {"node_id": 3, "caption": "Door"})
    assert "Call observe_map" in text(result)
    assert result.isError
    assert not worker.failed and not remote.failed
    await server.call_tool("observe_map", {})
    result = await server.call_tool("update_memory", {"node_id": 3, "caption": "Door"})
    assert "updated_memory" in text(result)
    assert (tmp_path / "local/subtasks/state.json").is_file()
    remote.close()
    assert worker.session_id is None
