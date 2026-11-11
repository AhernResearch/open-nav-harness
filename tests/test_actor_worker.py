"""Ownership, transport parity and recovery without GPU dependencies."""

import base64
from pathlib import Path
from unittest.mock import Mock, AsyncMock

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
    tools.request = AsyncMock(side_effect=worker.handle)
    return worker, tools


async def test_remote_probe_is_lazy_and_evidence_sync_is_acknowledged(remote):
    worker, tools = remote
    assert len(await tools.list_tools()) == 10
    assert worker.session_id is None
    await tools.start()
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
    await tools.close()
    assert worker.session_id is None


async def test_duplicate_dispatch_and_cross_episode_reset_are_fenced(remote):
    worker, tools = remote
    await tools.start()
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
    await tools.start()

    def failure(**_):
        raise RuntimeError("motion reply lost")

    worker.session.tools.functions["navigate_relative"] = failure
    with pytest.raises(RuntimeError, match="reply lost"):
        await tools.call_tool("navigate_relative", {"turn_deg": 0.0, "distance_m": 1.0})
    ident = worker.session_id
    await tools.close()
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


def test_nonmotion_rpc_failure_still_retains_the_actor_lease():
    from open_nav_harness.core.recovery import resource_state

    # No movement was dispatched, but the remote process still owns a session.
    for info in (
        {"protocol_version": 1, "session_id": "orphan"},
        {"protocol_version": 1, "failed": True},
        {"error": "connection lost"},
    ):
        assert resource_state([], wire=True, worker_info=info) == "unknown"
    assert (
        resource_state(
            [], wire=True, worker_info={"protocol_version": 1, "session_id": None, "failed": False}
        )
        == "idle"
    )


async def test_real_mcp_stdio_to_rpc_closes_remote_ownership(tmp_path, unused_tcp_port, monkeypatch):
    import asyncio
    import multiprocessing
    import sys
    import httpx
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from navgpt.vla import worker as module

    real_session = module.NavigationSession

    def session(*args, **kwargs):
        value = real_session(*args, **kwargs)
        value.tools.functions["observe"] = lambda views: ["real MCP/RPC transport"]
        return value

    monkeypatch.setattr(module, "NavigationSession", session)
    actor = Mock()
    actor.info.return_value = {"backend": "lightnav"}
    worker = ActorWorker(actor, "lightnav", tmp_path / "worker", "http://unused-env")
    process = multiprocessing.get_context("fork").Process(
        target=module.serve, args=(worker, "127.0.0.1", unused_tcp_port), daemon=True
    )
    process.start()
    url = f"http://127.0.0.1:{unused_tcp_port}"
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=2) as http:
            for _ in range(40):
                try:
                    (await http.get(url + "/info")).raise_for_status()
                    break
                except httpx.HTTPError:
                    await asyncio.sleep(0.05)
            env = {
                "NAVGPT_CONDITION": "planner_agentic",
                "NAVGPT_VLA_BACKEND": "lightnav",
                "NAVGPT_EXECUTION_URL": url,
                "NAVGPT_LIVE_DIR": str(tmp_path / "live"),
                "NAVGPT_PLACE_LABELS": str(tmp_path / "labels.json"),
            }
            params = StdioServerParameters(
                command=sys.executable, args=["-m", "navgpt.planner.tools.mcp_tools"], env=env
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    assert len((await client.list_tools()).tools) == 7  # VLA + node return; no direct movement.
                    assert (await http.get(url + "/info")).json()["session_id"] is None
                    result = await client.call_tool("observe", {"views": ["front"]})
                    assert result.content[0].text == "real MCP/RPC transport"
                    assert (await http.get(url + "/info")).json()["session_id"] is not None
            assert (await http.get(url + "/info")).json()["session_id"] is None
    finally:
        process.terminate()
        process.join(timeout=5)


def test_cancel_does_not_wait_for_inference_and_discards_result_before_motion(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    entered, release = threading.Event(), threading.Event()
    moved = []
    actor = Mock()
    actor.info.return_value = {'backend': 'navgpt'}
    def act(_):
        entered.set()
        release.wait(2)
        return 'prediction'
    actor.act.side_effect = act
    worker = ActorWorker(actor, 'navgpt', tmp_path, 'http://environment')
    worker.handle('POST', '/sessions', {'env': {'NAVGPT_VLA_BACKEND': 'navgpt'}})
    def move():
        worker.actor_call('POST', '/act', {})
        moved.append('motion')
    with ThreadPoolExecutor(1) as pool:
        task = pool.submit(move)
        assert entered.wait(1)
        worker.handle('POST', f'/sessions/{worker.session_id}/cancel', {})
        release.set()
        with pytest.raises(BusyError, match='discarded'):
            task.result()
    assert moved == [] and worker.failed


async def test_no_relative_tool_is_absent_on_both_rpc_sides(tmp_path):
    import json
    from navgpt.settings import AGENTIC_TOOLS

    selected = [name for name in AGENTIC_TOOLS if name != 'navigate_relative']
    actor = Mock()
    actor.info.return_value = {'backend': 'lightnav'}
    worker = ActorWorker(actor, 'lightnav', tmp_path / 'remote', 'http://fixed-environment')
    tools = RemoteTools({
        'NAVGPT_CONDITION': 'planner_agentic', 'NAVGPT_TOOLS': json.dumps(selected),
        'NAVGPT_VLA_BACKEND': 'lightnav', 'NAVGPT_EXECUTION_URL': 'http://worker',
        'NAVGPT_LIVE_DIR': str(tmp_path / 'local'),
    })
    tools.request = AsyncMock(side_effect=worker.handle)
    await tools.start()
    assert set(tools.definitions) == set(worker.session.tools.definitions) == set(selected)
    assert 'navigate_relative' not in worker.session.tools.functions
    await tools.close()


async def test_rpc_omitted_tools_cannot_restore_direct_movement(tmp_path):
    actor = Mock()
    actor.info.return_value = {'backend': 'lightnav'}
    worker = ActorWorker(actor, 'lightnav', tmp_path / 'remote', 'http://fixed-environment')
    tools = RemoteTools({'NAVGPT_CONDITION': 'planner_agentic',
                         'NAVGPT_VLA_BACKEND': 'lightnav', 'NAVGPT_EXECUTION_URL': 'http://worker',
                         'NAVGPT_LIVE_DIR': str(tmp_path / 'local')})
    tools.request = AsyncMock(side_effect=worker.handle)
    await tools.start()
    assert len(tools.definitions) == 7
    assert {'execute_subtask', 'navigate_to_node'} <= set(worker.session.tools.definitions)
    assert 'navigate_relative' not in tools.definitions
    before = tools.request.await_count
    with pytest.raises(ValueError, match='Unknown navigation tool'):
        await tools.call_tool('navigate_relative', {'turn_deg': 0., 'distance_m': 1.})
    assert tools.request.await_count == before  # Refused before any RPC or cancellation.
    with pytest.raises(ValueError, match='Unknown navigation tool'):
        worker.handle('POST', f'/sessions/{tools.session_id}/call', {
            'request_id': 'forbidden', 'name': 'navigate_relative',
            'arguments': {'turn_deg': 0., 'distance_m': 1.},
        })
    assert not (worker.directory / worker.session_id / 'calls.jsonl').exists()
    assert not worker.failed
    assert not tools.failed
    await tools.close()
