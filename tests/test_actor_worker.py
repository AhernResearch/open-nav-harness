"""Ownership, transport parity and recovery without GPU dependencies."""

import base64
from pathlib import Path
from unittest.mock import Mock

import pytest

from navgpt.planner.tools.registry import Image
from navgpt.planner.tools.transport import RemoteTools
from navgpt.vla.worker import ActorWorker, BusyError
from navgpt.vla.lightnav import LightNavBridge


@pytest.fixture
def remote(tmp_path):
    actor = Mock()
    actor.info.return_value = {"backend": "lightnav"}
    worker = ActorWorker(actor, "lightnav", tmp_path / "remote", "http://fixed-environment")
    env = {
        "NAVGPT_CONDITION": "planner_subtask",
        "NAVGPT_VLA_BACKEND": "lightnav",
        "NAVGPT_EXECUTION_URL": "http://worker",
        "NAVGPT_LIVE_DIR": str(tmp_path / "local"),
        "NAVGPT_PLACE_LABELS": str(tmp_path / "labels.json"),
    }
    tools = RemoteTools(env)
    tools.request = worker.handle
    return worker, tools


async def test_remote_probe_is_lazy_and_evidence_sync_is_acknowledged(remote):
    worker, tools = remote
    assert len(await tools.list_tools()) == 10
    assert worker.session_id is None
    tools.start()
    folder = Path(worker.session.LIVE_DIR)
    folder.mkdir()
    (folder / "view.png").write_bytes(b"fake pixels")
    (folder / "index.json").write_text('{"path":"' + str(folder / "view.png") + '"}')
    worker.session.tools.functions["observe_forward"] = lambda: ["observed", Image(b"fake pixels")]
    result = await tools.call_tool("observe_forward", {})
    assert result.content[0].text == "observed"
    assert base64.b64decode(result.content[1].data) == b"fake pixels"
    local = Path(tools.env["NAVGPT_LIVE_DIR"])
    assert (local / "view.png").read_bytes() == b"fake pixels"
    assert str(local / "view.png") in (local / "index.json").read_text()
    assert worker.artifacts(Path(tools.root)) == []
    tools.close()
    assert worker.session_id is None


def test_duplicate_dispatch_and_cross_episode_reset_are_fenced(remote):
    worker, tools = remote
    tools.start()
    call = Mock(return_value=["moved"])
    worker.session.tools.functions["navigate_relative"] = call
    path = "/sessions/" + tools.session_id + "/call"
    data = {
        "request_id": "once",
        "name": "navigate_relative",
        "arguments": {"turn_deg": 0.0, "distance_m": 1.0},
    }
    worker.handle("POST", path, data)
    with pytest.raises(ValueError, match="repeated"):
        worker.handle("POST", path, data)
    assert call.call_count == 1
    with pytest.raises(BusyError):
        worker.handle("POST", "/reset", {})
    with pytest.raises(BusyError):
        worker.handle("POST", "/sessions", {"env": tools.env})
    worker.lock.acquire()
    try:
        with pytest.raises(BusyError):
            worker.handle("POST", path, {**data, "request_id": "second"})
    finally:
        worker.lock.release()
    assert call.call_count == 1


async def test_failed_motion_retains_worker_and_local_ownership(remote):
    worker, tools = remote
    tools.start()

    def failure(**_):
        raise RuntimeError("motion reply lost")

    worker.session.tools.functions["navigate_relative"] = failure
    with pytest.raises(RuntimeError, match="reply lost"):
        await tools.call_tool("navigate_relative", {"turn_deg": 0.0, "distance_m": 1.0})
    ident = worker.session_id
    tools.close()
    assert worker.session_id == ident and worker.failed and tools.failed
    with pytest.raises(BusyError):
        worker.handle("POST", "/sessions/" + ident + "/close", {})
    assert "motion reply lost" in (Path(tools.root) / "calls.jsonl").read_text()


def test_owned_lightnav_reuses_controller_without_connecting_websocket():
    inference = Mock(
        return_value={
            "actions": {"actions": [[0.1, 0.0, 0.0]] * 10},
            "stop": False,
            "raw_text": "decoded",
            "latency_ms": 5,
        }
    )
    bridge = LightNavBridge(inference=inference)
    assert bridge.info()["execution"] == "owned_vllm"
    inference.assert_not_called()
    result = bridge.act({"rgb_policy": "fake-image", "text": "Walk to the door"})
    assert result[0]["action_args"]["linear_mps"] == 1.0
    assert result[0]["raw_text"] == "decoded"
    assert bridge.ws is None
