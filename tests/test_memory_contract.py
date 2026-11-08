"""Topology-memory lifecycle, evidence, retrieval, context and transport invariants."""

import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from navgpt.planner.tools.session import NavigationSession
from navgpt.planner.tools.memory import TopologyMemory
from navgpt.planner.tools.mcp_tools import build_mcp
from navgpt.planner.tools.transport import RemoteTools
from navgpt.vla.worker import ActorWorker
from nav_harness.core.artifacts import Journal
from nav_harness.core.context import ContextWindow


def pose(x=0, y=0):
    return {"position": [x, y, 0]}


def segment(memory, start, end, caption=None):
    memory.begin(start, "frame-000000", "execute_subtask", {"id": 1})
    memory.finish(end, "frame-000001", "vla_stop")
    if caption:
        e = memory.edges[-1]
        memory.remember(e["start_node_id"], e["end_node_id"], caption, {})


def configure(s, monkeypatch):
    s._last_pose = pose()
    s._tasks.observe({"policy_front": "Zg=="}, pose(), 0)
    monkeypatch.setattr(s, "_pano", lambda: {"pose": s._last_pose})
    monkeypatch.setattr(s, "_bump_obs", lambda *args, **kwargs: None)
    monkeypatch.setattr(s, "_pano_content", lambda *args: [])
    monkeypatch.setattr(s, "_fetch_map", lambda: (b"map", {
        "seen_area_m2": 2, "revisiting_earlier_position": False,
        "places_px": {str(n): [20 + n * 20, 30] for n in s._tasks.memory.nodes},
    }))
    return s


@pytest.fixture
def session(tmp_path, monkeypatch):
    return configure(NavigationSession(env={"NAVGPT_CONDITION": "planner_agentic",
                                           "NAVGPT_LIVE_DIR": str(tmp_path / "live")}), monkeypatch)


def text(result):
    return "\n".join(c.text for c in result.content if c.type == "text")


def args(start=0, end=1, caption="Crossed doorway; left turn remains uncertain."):
    return {"start_node_id": start, "end_node_id": end, "caption": caption}


def test_schema_and_plan_validation(session):
    schema = session.tools.definitions["update_memory"].inputSchema
    assert set(schema["properties"]) == set(args()) == set(schema["required"])
    for bad in [dict(args(), start_node_id="0"), dict(args(), edge_id=1)]:
        with pytest.raises(ValidationError):
            session.tools.invoke("update_memory", bad)
    assert "annotate_node" not in session.tools.definitions and "review_subtask" not in session.tools.definitions
    plan = [{"id": 1, "instruction": "Enter", "completion_criterion": "Beyond door", "status": "in_progress"}]
    session.tools.invoke("update_plan", {"subtasks": plan, "reason": "Observed doorway"})
    before = copy.deepcopy(session._tasks.snapshot())
    for bad in [{**plan[0], "status": "active"}, {**plan[0], "unexpected": 1}]:
        with pytest.raises(ValidationError):
            session.tools.invoke("update_plan", {"subtasks": [bad], "reason": "Invalid"})
    result = session.tools.invoke("update_plan", {"subtasks": plan + [{**plan[0], "id": 2}], "reason": "Invalid"})
    assert "At most one" in text(result)
    assert session._tasks.snapshot() == before


def test_only_pending_execution_and_explicit_map_can_be_written(session, monkeypatch):
    tools, memory = session.tools, session._tasks.memory
    assert "Call observe_map" in text(tools.invoke("update_memory", args()))
    tools.invoke("observe_map", {})
    assert "No completed execution" in text(tools.invoke("update_memory", args(end=0)))
    segment(memory, pose(), pose(2))
    assert tools.invoke("update_memory", args()).isError  # map predates this segment
    tools.invoke("observe_map", {})
    assert tools.invoke("update_memory", args(start=1, end=0)).isError
    assert not tools.invoke("update_memory", args()).isError
    geometry = copy.deepcopy(memory.nodes)
    assert not tools.invoke("update_memory", args(caption="Correction: door not crossed.")).isError
    assert memory.nodes == geometry and len(memory.edges) == 1
    assert memory.edges[0]["caption"] == "Correction: door not crossed."
    assert not session._tasks.tasks
    assert tools.invoke("update_memory", args(caption=" ")).isError
    monkeypatch.setattr(session, "_fetch_map", lambda: (None, "unavailable"))
    tools.invoke("observe_map", {})
    assert tools.invoke("update_memory", args()).isError
    persisted = json.loads((memory.directory / "topology.json").read_text())
    assert persisted["edges"][0]["caption"] == memory.edges[0]["caption"]


@pytest.mark.parametrize("motion", ["step_hightolow", "retrace_to", "step_velocity", "teleport"])
def test_motion_including_rotation_revokes_map(session, monkeypatch, motion):
    segment(session._tasks.memory, pose(), pose(2))
    session.tools.invoke("observe_map", {})
    monkeypatch.setattr("navgpt.planner.tools.session.requests.post", lambda *a, **k:
                        SimpleNamespace(status_code=200, json=lambda: {"outputs": {}}))
    session._call("r2rce__" + motion, {"angle_rad": 1.57, "distance_m": 0})
    assert "Call observe_map" in text(session.tools.invoke("update_memory", args()))


def test_revisit_repeat_self_loop_and_failed_return_are_distinct(tmp_path):
    m = TopologyMemory(tmp_path)
    segment(m, pose(), pose(2), "Explored right branch; no stairs observed.")
    segment(m, pose(2), pose(), "Returned to junction after unsuccessful exploration.")
    segment(m, pose(), pose(2), "Rechecked right branch from another view.")
    assert len(m.nodes) == 2 and len(m.edges) == 3
    assert [(e["start_node_id"], e["end_node_id"]) for e in m.edges] == [(0, 1), (1, 0), (0, 1)]
    assert "no stairs" in m.edges[0]["caption"]
    segment(m, pose(2), pose(2), "Rotated in place; still uncertain.")
    assert len(m.nodes) == 2 and m.edges[-1]["start_node_id"] == m.edges[-1]["end_node_id"]
    segment(m, pose(2), pose(2, 3), "Reached another floor.")
    assert len(m.nodes) == 3
    assert [r["sequence"] for r in m.read(0, 1)["records"]] == [3, 1]
    assert [r["sequence"] for r in m.read(1, 0)["records"]] == [2]


def test_uncertain_response_never_fabricates_destination(tmp_path):
    m = TopologyMemory(tmp_path)
    m.begin(pose(), "f0", "execute_subtask")
    m.record(pose(1), "f1")
    m.uncertain("response lost")
    assert m.edges[0]["end_node_id"] is None and m.current_node_id is None
    with pytest.raises(RuntimeError, match="Unresolved"):
        m.begin(pose(1), "f1", "execute_subtask")
    assert json.loads((tmp_path / "topology.json").read_text())["edges"][0]["status"] == "uncertain"


def test_context_budget_pagination_and_omission_index():
    m = TopologyMemory()
    for i in range(25):
        segment(m, pose(i % 2 * 2), pose((i + 1) % 2 * 2), "记" * 800)
    context = m.context(1)
    assert len(json.dumps(context, ensure_ascii=False).encode()) <= m.CONTEXT_BYTES
    assert context["current_execution"]["sequence"] == 25
    assert context["history_index"] or context["unlisted_pair_count"]
    page, seq = m.read(0), []
    while True:
        seq += [r["sequence"] for r in page["records"]]
        if page["next_offset"] is None:
            break
        page = m.read(0, offset=page["next_offset"])
    assert seq == list(range(25, 0, -1))
    assert m.edges[0]["caption"] == "记" * 800


def test_real_tool_boundary_zero_motion_rejection_and_lost_response(session, monkeypatch):
    session._tasks.update_plan([{"id": 1, "instruction": "Enter", "completion_criterion": "Beyond door"}], "Plan")
    monkeypatch.setattr(session, "vla_navigate", lambda instruction: [json.dumps({"subtask_return": "vla_stop"})])
    session.tools.invoke("observe_map", {})
    assert not session.tools.invoke("execute_subtask", {"task_id": 1}).isError
    m = session._tasks.memory
    assert m.edges[-1]["start_node_id"] == m.edges[-1]["end_node_id"] == 0
    assert session.tools.invoke("update_memory", args(end=0)).isError
    session.tools.invoke("observe_map", {})
    assert not session.tools.invoke("update_memory", args(end=0)).isError
    before = copy.deepcopy(m.edges)
    assert session.tools.invoke("execute_subtask", {"task_id": 99}).isError
    assert m.edges == before

    def lose(instruction):
        session._motion_epoch += 1
        raise RuntimeError("lost motion response")

    monkeypatch.setattr(session, "vla_navigate", lose)
    with pytest.raises(RuntimeError, match="lost motion"):
        session.tools.invoke("execute_subtask", {"task_id": 1})
    assert m.edges[-1]["status"] == "uncertain" and m.edges[-1]["end_node_id"] is None


async def test_current_context_replaces_old_snapshots_without_erasing_trace(session, tmp_path):
    segment(session._tasks.memory, pose(), pose(2), "Old assessment.")
    window = ContextWindow("system", "original route", "bounded", 1500, 100)
    original = json.dumps({"subtask_state": session._tasks.snapshot()})
    window.append([{"role": "tool", "content": original}], session._tasks.snapshot())
    session._tasks.memory.remember(0, 1, "Corrected assessment.", {})
    window.append([{"role": "tool", "content": json.dumps({"subtask_state": session._tasks.snapshot()})}],
                  session._tasks.snapshot())

    async def count(messages, _):
        return 300

    messages = await window.build(count, [], Journal(tmp_path / "trace"))
    assert "Old assessment" not in json.dumps(messages)
    assert "Corrected assessment" in json.dumps(messages)
    assert window.blocks[0][0]["content"] == original
    assert sum("Current persisted" in m["content"] for m in messages) == 1


async def test_mcp_rpc_gate_and_history_do_not_fence_recoverable_rejections(tmp_path, monkeypatch):
    worker = ActorWorker(SimpleNamespace(info=lambda: {"backend": "lightnav"}), "lightnav",
                         tmp_path / "remote", "http://fixed-env")
    remote = RemoteTools({"NAVGPT_CONDITION": "planner_agentic", "NAVGPT_VLA_BACKEND": "lightnav",
                          "NAVGPT_EXECUTION_URL": "http://worker", "NAVGPT_LIVE_DIR": str(tmp_path / "local")})
    remote.request = worker.handle
    remote.start()
    configure(worker.session, monkeypatch)
    segment(worker.session._tasks.memory, pose(), pose(2))
    server = build_mcp(remote)
    result = await server.call_tool("update_memory", args())
    assert result.isError and "Call observe_map" in text(result)
    assert not worker.failed and not remote.failed
    await server.call_tool("observe_map", {})
    assert not (await server.call_tool("update_memory", args())).isError
    result = await server.call_tool("read_memory", {"node_id": 0, "other_node_id": 1, "include_images": True})
    assert "Crossed doorway" in text(result)
    assert sum(c.type == "image" for c in result.content) <= 4
    assert (tmp_path / "local/subtasks/topology.json").is_file()
    remote.close()
    assert worker.session_id is None
