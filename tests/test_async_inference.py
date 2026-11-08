"""Concurrency, cancellation, backpressure and state isolation without a GPU."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import httpx
import pytest
from fastapi import HTTPException

from navgpt.vla.inference import InferenceService, OpenSession, Operation, create_app


class Factory:
    backend = "navgpt"

    def __init__(self, workers=2):
        self.executor = ThreadPoolExecutor(workers)
        self.started = threading.Event()
        self.release = threading.Event()
        self.actors = []

    async def start(self):
        pass

    def make(self):
        history = []

        def act(data):
            self.started.set()
            self.release.wait(3)
            history.append(data["frame"])
            if data.get("fail"):
                raise ValueError("bad output")
            return list(history)

        actor = SimpleNamespace(
            act=act, reset=lambda _: history.clear(), state=lambda: list(history), close=lambda: None
        )
        self.actors.append(actor)
        return actor

    async def run(self, fn):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn)

    async def close(self):
        self.release.set()
        self.executor.shutdown(wait=True)


async def open_sessions(service, count):
    ids = [f"{i:032x}" for i in range(count)]
    for ident in ids:
        await service.open(OpenSession(session_id=ident, backend="navgpt"))
    return ids


def request(ident, frame, **extra):
    return Operation(request_id=ident, operation="act", data={"frame": frame, **extra})


async def test_parallel_histories_reset_and_duplicate_protection():
    factory = Factory()
    service = InferenceService(factory)
    a, b = await open_sessions(service, 2)
    calls = [
        asyncio.create_task(service.call(a, request("a", "A"))),
        asyncio.create_task(service.call(b, request("b", "B"))),
    ]
    await asyncio.to_thread(factory.started.wait, 2)
    assert service.pending == 2
    with pytest.raises(HTTPException, match="409"):
        await service.call(a, request("another", "C"))
    factory.release.set()
    assert await asyncio.gather(*calls) == [["A"], ["B"]]
    with pytest.raises(HTTPException):
        await service.call(a, request("a", "duplicate"))
    await service.call(a, Operation(request_id="reset", operation="reset"))
    assert await service.call(b, request("b2", "B2")) == ["B", "B2"]
    assert await service.call(a, request("a2", "A2")) == ["A2"]
    await service.shutdown()


async def test_bounded_queue_rejects_before_mutation_and_cancel_waits_for_drain():
    factory = Factory(workers=1)
    service = InferenceService(factory, max_pending=2)
    a, b, c = await open_sessions(service, 3)
    first = asyncio.create_task(service.call(a, request("a", "A")))
    await asyncio.to_thread(factory.started.wait, 2)
    second = asyncio.create_task(service.call(b, request("b", "B")))
    await asyncio.sleep(0)
    with pytest.raises(HTTPException) as exc:
        await service.call(c, request("c", "C"))
    assert exc.value.status_code == 429
    assert service.session(c).seen == set()
    service.cancel(b)
    with pytest.raises(HTTPException):
        await service.close(b)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert service.pending == 2
    factory.release.set()
    with pytest.raises(RuntimeError, match="Cancelled"):
        await second
    for _ in range(50):
        if service.pending == 0:
            break
        await asyncio.sleep(0.01)
    assert service.pending == 0
    assert factory.actors[1].state() == []  # Cancelled while queued: no mutation.
    assert service.session(a).failed
    assert await service.call(c, request("c", "C")) == ["C"]
    await service.shutdown()


async def test_failed_session_does_not_block_other_sessions_and_close_retires_id():
    factory = Factory()
    factory.release.set()
    service = InferenceService(factory)
    a, b = await open_sessions(service, 2)
    with pytest.raises(ValueError):
        await service.call(a, request("a", "bad", fail=True))
    assert await service.call(b, request("b", "B")) == ["B"]
    await service.close(a)
    with pytest.raises(HTTPException):
        await service.open(OpenSession(session_id=a, backend="navgpt"))
    await service.shutdown()


async def test_http_protocol_and_capacity():
    factory = Factory()
    factory.release.set()
    service = InferenceService(factory, max_sessions=1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test"
    ) as client:
        ident = "a" * 32
        assert (
            await client.post("/sessions", json={"session_id": ident, "backend": "navgpt"})
        ).status_code == 200
        assert (
            await client.post("/sessions", json={"session_id": "b" * 32, "backend": "navgpt"})
        ).status_code == 429
        reply = await client.post(f"/sessions/{ident}/call", json=request("once", "A").model_dump())
        assert reply.json()["result"] == ["A"]
        assert (await client.get("/info")).json()["completed_requests"] == 1
    await service.shutdown()


def test_runner_checks_separate_bindings_and_never_globally_resets_shared_actor(monkeypatch):
    from navgpt.planner.runner import _check_execution, _reset_vla, PreflightError
    from nav_harness.core.version import implementation_digest
    cfg = SimpleNamespace(execution_url='http://worker:8000', vla_url='http://inference:8000', env_url='http://env:8000')
    inference = {'shared_inference': True, 'protocol_version': 2, 'implementation_digest': implementation_digest()}
    worker = {'implementation_digest': implementation_digest(), 'environment_url': cfg.env_url,
              'inference_url': cfg.vla_url, 'session_id': None, 'failed': False}
    monkeypatch.setattr('navgpt.planner.runner.requests.get', lambda *a, **kw: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: worker))
    _check_execution(cfg, inference)
    worker['inference_url'] = 'http://wrong:8000'
    with pytest.raises(PreflightError, match='different inference'):
        _check_execution(cfg, inference)
    monkeypatch.setattr('navgpt.planner.runner.requests.post', lambda *a, **kw: pytest.fail('Global reset'))
    _reset_vla(cfg)


async def test_native_factory_loads_weights_once_and_serializes_forward(monkeypatch):
    import time
    from navgpt.vla import owned
    from navgpt.vla.shared import BackendFactory
    loads = []
    active = 0
    maximum = 0
    class Engine:
        def __init__(self, model_path, shared_model=None):
            if shared_model is None:
                loads.append(model_path)
            model, processor = shared_model or (object(), object())
            self.agent = SimpleNamespace(model=model, processor=processor)
            self.history = []
        def info(self):
            return {'backend': 'navgpt'}
        def act(self):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            time.sleep(.01)
            self.history.append(len(self.history))
            active -= 1
            return list(self.history)
    monkeypatch.setattr(owned, 'NativeEngine', Engine)
    factory = BackendFactory({'backend': 'navgpt', 'model_path': 'weights'})
    await factory.start()
    actors = await asyncio.gather(*(factory.run(factory.make) for _ in range(4)))
    results = await asyncio.gather(*(factory.run(actor.act) for actor in actors))
    assert loads == ['weights']
    assert len({id(a.agent.model) for a in actors}) == 1
    assert len({id(a.history) for a in actors}) == 4
    assert results == [[0]] * 4 and maximum == 1
    await factory.close()


@pytest.mark.parametrize('health', [{'policy_camera': {'width': 480}}, {'history_cameras': ['policy_front']}])
def test_native_rejects_lightnav_passive_history_before_motion(health):
    from navgpt.planner.runner import _check_history_cameras, PreflightError
    with pytest.raises(PreflightError, match='history cameras'):
        _check_history_cameras('navgpt', health)
    _check_history_cameras('lightnav', health)


def test_history_contract_accepts_native_and_legacy_native_environment():
    from navgpt.planner.runner import _check_history_cameras
    _check_history_cameras('navgpt', {'history_cameras': ['front','right','back','left']})
    _check_history_cameras('navgpt', {'policy_camera': None})
