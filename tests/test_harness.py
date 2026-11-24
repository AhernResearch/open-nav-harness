import gzip
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from open_nav_harness.platform.api import create_app
from open_nav_harness.platform.kernel_feed import JsonlTail


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


def test_view_routes_serve_the_console(tmp_path):
    web = Path(__file__).resolve().parents[1] / "open_nav_harness" / "web" / "index.html"
    if not web.is_file():
        pytest.skip("frontend is not built")
    with TestClient(create_app(tmp_path)) as c:
        for route in ("/", "/harness", "/workbench"):
            assert c.get(route).status_code == 200


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
