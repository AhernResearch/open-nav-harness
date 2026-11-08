import gzip
import hashlib
import json
import struct
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from nav_harness.core.config import ExperimentSpec, PlannerProfile, ServiceSpec

from nav_harness.platform.api import create_app
from nav_harness.platform.harness import Feed, JsonlTail, fit_map

EP = 7


def test_pending_job_explains_retained_leases_without_releasing_or_retrying(tmp_path):
    app = create_app(tmp_path)
    db = app.state.db
    env = db.register_service(ServiceSpec(name="queue-env", kind="environment", url="http://queue-env:19000"))
    vla = db.register_service(ServiceSpec(name="queue-vla", kind="vla", url="http://queue-vla:18000"))
    planner = db.register_service(ServiceSpec(name="queue-planner", kind="planner", url="http://queue-planner:18001/v1",
        planner=PlannerProfile(model="test", base_url="http://queue-planner:18001/v1")))
    experiment = db.create_experiment(ExperimentSpec(name="queue test", environment_service=env, vla_service=vla,
        planner_service=planner, episodes=[0], config={"dataset": {"name": "r2r", "split": "train"}, "harness": {"mode": "agentic"}}))
    old = db.enqueue(experiment)
    attempt = db.claim()
    db.block(attempt["id"], "Uncertain movement; preserve leases")
    pending = db.enqueue(experiment)
    before = db.rows("SELECT * FROM leases")
    with TestClient(app) as client:
        response = client.get(f"/api/harness/jobs/{pending}")
        assert response.status_code == 200
        assert response.json()["waiting_reason"] == "needs_reconciliation"
        assert response.json()["blocked_attempt_id"] == attempt["id"]
        assert response.json()["attempt_id"] is None
        assert client.get(f"/api/harness/jobs/{old}").json()["attempt_id"] == attempt["id"]
        assert client.get("/api/harness/jobs/missing").status_code == 404
        assert db.rows("SELECT * FROM leases") == before
        assert len(db.rows("SELECT * FROM attempts")) == 1


def test_harness_choices_probe_models_without_changing_ownership_or_status(tmp_path):
    app = create_app(tmp_path)
    db = app.state.db
    ids = {}
    for name in ("fp4", "other", "offline", "stopped"):
        url = f"http://{name}:18000/v1"
        ids[name] = db.register_service(ServiceSpec(
            name=name, kind="planner", url=url, planner=PlannerProfile(model=name, base_url=url),
            managed=name == "stopped", template="trusted" if name == "stopped" else None,
        ))
    ids["sdk"] = db.register_service(ServiceSpec(
        name="sdk", kind="planner", url="http://sdk:18000",
        planner=PlannerProfile(model="sdk", base_url="http://sdk:18000", runtime="claude", context_policy="full"),
    ))
    health = {
        ids["fp4"]: {"ok": True, "details": {"data": [{"id": "fp4"}]}},
        ids["other"]: {"ok": True, "details": {"data": [{"id": "wrong-model"}]}},
        ids["offline"]: {"ok": False, "error": "connection refused"},
        ids["stopped"]: {"ok": True, "details": {"data": [{"id": "stopped"}]}},
        ids["sdk"]: {"ok": None},
    }
    app.state.services.health = AsyncMock(side_effect=lambda sid: health[sid])
    app.state.services.ssh = AsyncMock()
    before = db.rows("SELECT id,status FROM services ORDER BY id")
    with TestClient(app) as client:
        response = client.get("/api/harness/services")
        assert response.status_code == 200
        rows = {r["spec"]["name"]: r for r in response.json()}
        assert rows["fp4"]["available"] is True  # external status is not a readiness gate
        assert rows["sdk"]["available"] is True  # SDK access is checked at preflight
        assert rows["other"]["available"] is False
        assert rows["other"]["unavailable_reason"] == "服务未提供此模型"
        assert rows["offline"]["available"] is False
        assert rows["stopped"]["available"] is False  # do not adopt an answering process
        assert before == db.rows("SELECT id,status FROM services ORDER BY id")
        assert db.rows("SELECT * FROM jobs") == []
    app.state.services.ssh.assert_not_called()


def write_jsonl(path: Path, records, mode="a"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class Attempt:
    """A minimal attempt output folder with the three files the harness reads."""

    def __init__(self, root: Path):
        self.root = root
        self.live = root / "episode" / "live" / f"ep{EP:04d}"
        self.trace = root / "episode" / "work" / f"ep{EP:04d}" / "trace"
        self.seq = 0

    def journal(self, kind, **fields):
        self.seq += 1
        write_jsonl(self.trace / "requests.jsonl",
                    [{"schema_version": 1, "seq": self.seq, "time": 100.0 + self.seq, "kind": kind, **fields}])

    def call(self, call_id, name, arguments=None):
        self.journal("tool_dispatch", call_id=call_id, name=name, arguments=arguments or {})

    def result(self, call_id, name, lines=(), state=None, error=False, images=()):
        content = "\n".join([*lines, json.dumps({"subtask_state": state or {"tasks": []}})])
        self.journal("tool_result", call_id=call_id, name=name, is_error=error,
                     feedback={"role": "tool", "content": content},
                     images=[{"type": "image_url", "image_url": {"url": {"artifact": d}}} for d in images])

    def event(self, kind, **fields):
        write_jsonl(self.live / "subtasks" / "events.jsonl", [{"kind": kind, **fields}])

    def pose(self, x):
        self.event("policy_execution_result",
                   feedback={"pose": {"position": [x, 0, 0], "heading_rad": 0.0}, "step_count": 1})

    def topology(self, nodes):
        path = self.live / "subtasks" / "topology.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"nodes": {k: {"position": v} for k, v in nodes.items()}}))

    def media(self, width=400, height=300):
        raw = b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + struct.pack(">II", width, height) + b"rest"
        digest = hashlib.sha256(raw).hexdigest()
        (self.trace / "media").mkdir(parents=True, exist_ok=True)
        (self.trace / "media" / digest).write_bytes(raw)
        return digest


def updates(events, call_id):
    return [e["item"] for e in events if e["type"] == "item.updated" and e["item"]["id"] == call_id]


def test_tail_waits_for_complete_lines_and_detects_truncation(tmp_path):
    path = tmp_path / "x.jsonl"
    tail = JsonlTail(path)
    assert tail.read() == []
    path.write_bytes('{"a": "中'.encode()[:-1])  # unterminated, split inside a UTF-8 character
    assert tail.read() == []
    path.write_bytes('{"a": "中"}\n{"b": 1'.encode())
    assert [r for _, r in tail.read()] == [{"a": "中"}]
    with path.open("ab") as f:
        f.write(b"}\n")
    assert [r for _, r in tail.read()] == [{"b": 1}]
    path.write_text("")
    assert tail.read() is None


def test_motion_segments_belong_to_the_calls_that_moved(tmp_path):
    a = Attempt(tmp_path)
    a.call("rejected", "execute_subtask")
    a.result("rejected", "execute_subtask", error=True)
    a.call("walk", "execute_subtask")
    a.pose(1.0)
    feed = Feed(tmp_path, EP)
    live = feed.poll()
    assert [u["pose"]["position"][0] for u in updates(live, "walk")] == [1.0]  # open call, live
    a.pose(2.0)
    a.event("observation", context_key=None, images={})  # observe-tool frame: not motion progress
    a.event("execution_memory", current_execution={"sequence": 1}, current_node_id=1)
    a.result("walk", "execute_subtask")
    a.call("turn", "navigate_relative")
    a.event("execution_memory", current_execution={"sequence": 2}, current_node_id=2)
    a.result("turn", "navigate_relative")
    events = feed.poll()
    assert updates(live + events, "rejected") == []
    assert len(updates(live + events, "walk")) == 3
    assert updates(events, "turn") == [{"id": "turn", "execution": {"sequence": 2}, "node": 2}]
    assert [e["item"]["status"] for e in events if e["type"] == "item.completed"] == ["completed"] * 2


def test_events_wait_for_their_call_and_state_comes_from_results(tmp_path):
    a = Attempt(tmp_path)
    a.pose(1.0)  # synced before the journal shows the call
    feed = Feed(tmp_path, EP)
    assert feed.poll() == []
    plan = {"tasks": [{"id": 1, "instruction": "go", "status": "in_progress"}]}
    a.call("p", "update_plan", {"subtasks": []})
    a.result("p", "update_plan", ['{"plan_updated": true}'], state=plan)
    a.call("m", "execute_subtask")
    events = feed.poll()
    done = next(e["item"] for e in events if e["type"] == "item.completed")
    assert done["state"] == plan and done["result"] == ['{"plan_updated": true}']
    assert [e["item"]["type"] for e in events if e["type"] == "item.started"] == ["plan", "motion"]
    assert len(updates(events, "m")) == 1


def test_cursor_resumes_without_duplicates(tmp_path):
    a = Attempt(tmp_path)
    a.call("x", "observe")
    first = Feed(tmp_path, EP)
    assert len(first.poll()) == 1
    a.result("x", "observe")
    a.journal("model_response", response={"choices": [{"message": {"content": "hi"}}]})
    resumed = Feed(tmp_path, EP, first.cursor).poll()
    assert [e["type"] for e in resumed] == ["item.completed", "turn.completed"]
    assert Feed(tmp_path, EP, "garbage").poll()[0]["type"] == "item.started"


def test_map_is_placed_once_node_positions_are_known(tmp_path):
    a = Attempt(tmp_path)
    digest = a.media()
    a.call("map", "observe_map")
    a.result("map", "observe_map", [json.dumps({"places_px": {"0": [100, 100], "1": [150, 100]}})],
             images=[digest])
    feed = Feed(tmp_path, EP)
    completed = feed.poll()[-1]["item"]
    assert completed["map"]["size"] == [400, 300] and completed["map"]["transform"] is None
    a.topology({"0": [0, 0, 0], "1": [1, 0, 0]})
    placed = feed.poll()[0]["item"]["map"]
    assert placed["transform"]["ppm"] == 50 and placed["transform"]["matrix"][2] == 100
    assert feed.poll() == []


def test_only_whitelisted_files_are_opened(tmp_path, monkeypatch):
    a = Attempt(tmp_path)
    a.call("x", "observe")
    a.live.mkdir(parents=True)
    remote = a.live / "remote_calls.jsonl"
    remote.write_text("{}\n")
    remote.chmod(0)
    opened = []
    original = Path.open
    monkeypatch.setattr(Path, "open", lambda self, *args, **kw: opened.append(self.name) or original(self, *args, **kw))
    Feed(tmp_path, EP).poll()
    assert set(opened) <= {"requests.jsonl", "events.jsonl", "topology.json", f"episode_{EP}.jsonl"}


def project(points, ppm, theta, origin):
    import cmath

    a = cmath.rect(ppm, theta)
    return {str(i): [round(v.real, 3), round(v.imag, 3)]
            for i, v in enumerate(a * complex(x, z) + complex(*origin) for x, _, z in points)}


def test_fit_map_snaps_scale_and_rejects_inconsistent_labels():
    nodes = {"0": [0, 0, 0], "1": [2, 0, 1], "2": [-1, 0, 3]}
    places = project(nodes.values(), 100, 0, (30, 40))
    fit = fit_map(places, {str(i): v for i, v in enumerate(nodes.values())})
    assert fit["ppm"] == 100 and fit["residual_px"] < 1e-6 and fit["nodes"] == 3
    noisy = {k: [x + 0.4, y - 0.3] for k, (x, y) in places.items()}
    assert fit_map(noisy, nodes)["ppm"] == 100
    turned = fit_map(project(nodes.values(), 50, 1.5708, (0, 0)), nodes)
    assert abs(turned["matrix"][3] - 50) < 0.01  # a quarter turn maps +x to +py
    assert fit_map({**places, "2": [0, 0]}, nodes) is None  # an outlier node disagrees
    mirrored = {k: [-x, y] for k, (x, y) in places.items()}
    assert fit_map(mirrored, nodes) is None  # reflections are not a renderer transform
    assert fit_map({"0": places["0"]}, nodes) is None
    assert fit_map({"0": [0, 0], "1": [1, 0]}, {"0": [0, 0, 0], "1": [0.01, 0, 0]}) is None


def finished_attempt(tmp_path):
    app = create_app(tmp_path / "state")
    db = app.state.db
    now = time.time()
    with db.transaction() as c:
        from sqlalchemy import text

        c.execute(text("INSERT INTO experiments VALUES('e', :s, :t)"), {"s": json.dumps({"name": "demo"}), "t": now})
        c.execute(text("INSERT INTO jobs(id,experiment_id,status,created,updated) VALUES('j','e','completed',:t,:t)"),
                  {"t": now})
        c.execute(text("INSERT INTO attempts(id,job_id,episode,ordinal,status,output,created,updated) "
                       "VALUES('a','j',:e,1,'completed',:o,:t,:t)"), {"e": EP, "o": str(tmp_path / "out"), "t": now})
    a = Attempt(tmp_path / "out")
    a.call("x", "observe")
    a.result("x", "observe")
    return app, a.media()


def test_stream_replays_and_ends_with_cache_rules(tmp_path):
    app, digest = finished_attempt(tmp_path)
    with TestClient(app) as c:
        body = c.get("/api/attempts/a/stream").text
        frames = [f for f in body.split("\n\n") if f.startswith("id:")]
        events = [e for f in frames for e in json.loads(f.split("data: ", 1)[1])]
        assert [e["type"] for e in events] == ["item.started", "item.completed", "attempt.status"]
        assert body.rstrip().endswith('"job_status": "completed"}') and "event: end" in body
        cursor = frames[0].split("\n")[0][4:]
        resumed = c.get("/api/attempts/a/stream", headers={"Last-Event-ID": cursor}).text
        assert '"item.started"' not in resumed
        assert c.get("/api/attempts/missing/stream").status_code == 404
        art = c.get(f"/api/attempts/a/artifact/{digest}")
        assert "immutable" in art.headers["cache-control"]
        listing = c.get("/api/harness/attempts")
        assert listing.headers["cache-control"] == "no-store"
        assert listing.json()[0] | {"name": "demo", "status": "completed"} == listing.json()[0]


def test_view_routes_serve_the_console(tmp_path):
    web = Path(__file__).resolve().parents[1] / "nav_harness" / "web" / "index.html"
    if not web.is_file():
        pytest.skip("frontend is not built")
    with TestClient(create_app(tmp_path)) as c:
        for route in ("/", "/harness", "/workbench"):
            assert c.get(route).status_code == 200


def test_executed_instruction_comes_from_the_first_episode_line_only(tmp_path):
    a = Attempt(tmp_path)
    log = tmp_path / "episode" / f"episode_{EP}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    meta = {"kind": "episode_meta", "index": EP, "episode_id": "9", "scene_id": "mp3d/s/s.glb", "instruction": "Go."}
    log.write_text(json.dumps(meta))  # unterminated: not yet trustworthy
    feed = Feed(tmp_path, EP)
    assert feed.poll() == []
    with log.open("a") as f:
        f.write("\n" + json.dumps({"kind": "episode_metrics", "metrics": {"success": 1}}) + "\n")
    a.call("x", "observe")
    events = feed.poll()
    assert events[0] == {"type": "session.started", "instruction": "Go.", "scene_id": "mp3d/s/s.glb",
                         "episode_id": "9", "index": EP}
    assert "metrics" not in json.dumps(events) and feed.poll() == []


def test_catalog_lists_scenes_and_text_only(tmp_path):
    app = create_app(tmp_path)
    (tmp_path / "catalog").mkdir()
    episodes = [{"episode_id": i, "scene_id": f"mp3d/{s}/{s}.glb", "goals": [{"position": [1, 2, 3]}],
                 "reference_path": [[0, 0, 0]], "instruction": {"instruction_text": t}}
                for i, (s, t) in enumerate([("A", "Walk to the sofa. "), ("B", "Go up"), ("A", "Exit the room")])]
    with gzip.open(tmp_path / "catalog" / "r2r_train.json.gz", "wt") as f:
        json.dump({"episodes": episodes}, f)
    with TestClient(app) as c:
        assert c.get("/api/catalog").json() == [{"name": "r2r_train", "dataset": "r2r", "split": "train",
                                                 "scenes": [{"scene": "A", "count": 2}, {"scene": "B", "count": 1}]}]
        rows = c.get("/api/catalog/r2r_train/episodes", params={"scene": "A", "q": "sofa"}).json()
        assert rows == [{"index": 0, "episode_id": 0, "scene": "A", "instruction": "Walk to the sofa."}]
        assert c.get("/api/catalog/nope/episodes", params={"scene": "A"}).status_code == 404
