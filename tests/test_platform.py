import base64
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from open_nav_harness.core.artifacts import ArtifactStore, Journal
from open_nav_harness.core.config import (
    PlannerProfile,
    PlatformSettings,
    ServiceSpec,
)
from open_nav_harness.core.context import ContextOverflow, ContextWindow, ImageBudget
from open_nav_harness.platform.api import create_app
from open_nav_harness.platform.database import Database
from open_nav_harness.platform.kernel_feed import JsonlTail
from open_nav_harness.platform.services import ServiceManager
from open_nav_harness.providers.compatible import CompatibleClient


def read_jsonl(path):
    """Complete journal lines; a trailing uncommitted write is not yet a record."""
    return [record for _, record in JsonlTail(path).read()]


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path)
    d.migrate()
    yield d
    d.engine.dispose()


async def test_external_planner_health_uses_credential_env(db, monkeypatch):
    sid = db.register_service(ServiceSpec(
        name='gpt', kind='planner', url='https://gateway.test/v1',
        planner=PlannerProfile(model='gpt', api_format='responses',
                               base_url='https://gateway.test/v1',
                               credential_env='ONH_TEST_KEY'),
    ))
    services = ServiceManager(db, PlatformSettings())
    monkeypatch.delenv('ONH_TEST_KEY', raising=False)
    assert (await services.health(sid))['ok'] is False
    monkeypatch.setenv('ONH_TEST_KEY', 'test-secret')
    get = AsyncMock(return_value=httpx.Response(200, json={'data': [{'id': 'gpt'}]},
        request=httpx.Request('GET', 'https://gateway.test/v1/models')))
    monkeypatch.setattr(httpx.AsyncClient, 'get', get)
    assert (await services.health(sid))['ok'] is True
    get.assert_awaited_once_with('https://gateway.test/v1/models',
                               headers={'Authorization': 'Bearer test-secret'})
    assert 'test-secret' not in db.service(sid).model_dump_json()


async def test_external_services_cannot_be_stopped(db):
    env = db.register_service(ServiceSpec(name="env", kind="environment", url="http://env:8000"))
    manager = ServiceManager(db, PlatformSettings())
    manager.ssh = AsyncMock()
    with pytest.raises(ValueError, match="External"):
        await manager.operate(env, "stop")
    manager.ssh.assert_not_called()


def test_lossless_media_roundtrip_and_corruption(tmp_path):
    store = ArtifactStore(tmp_path)
    value = {"image_url": {"url": "data:image/png;base64," + base64.b64encode(b"image").decode()}}
    frozen = store.freeze(value)
    assert store.thaw(frozen) == value
    (tmp_path / "media" / frozen["image_url"]["url"]["artifact"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash"):
        store.thaw(frozen)


def test_journal_never_overwrites_and_rejects_corrupt_line(tmp_path):
    journal = Journal(tmp_path)
    journal.emit("request", password="secret")
    assert read_jsonl(journal.path)[0]["password"] == "[REDACTED]"
    with pytest.raises(FileExistsError):
        Journal(tmp_path)
    with journal.path.open("a") as f:
        f.write("{")
    assert len(read_jsonl(journal.path)) == 1
    with journal.path.open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError):
        read_jsonl(journal.path)


async def test_context_prunes_atomic_blocks_preserves_current_state(tmp_path):
    w = ContextWindow("system", "task", "bounded", 1500, 128)
    w.append(
        [{"role": "assistant", "content": "old"}, {"role": "tool", "content": "old result"}], {"current": "a"}
    )
    w.append(
        [{"role": "assistant", "content": "new"}, {"role": "tool", "content": "new result"}], {"current": "b"}
    )

    async def counter(messages, tools):
        return 1300 if any(m.get("content") == "old" for m in messages) else 600

    messages = await w.build(counter, [], Journal(tmp_path))
    assert not any(m.get("content") == "old" for m in messages)
    assert [m["content"] for m in messages[-2:]] == ["new", "new result"]
    assert '"current": "b"' in messages[2]["content"]


def _image_turn(name, images):
    return [{"role": "assistant", "content": name},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": name}}] * images}]


def _urls(messages):
    return [item["image_url"]["url"] for m in messages if isinstance(m.get("content"), list)
            for item in m["content"] if item.get("type") == "image_url"]


async def test_image_budget_omits_oldest_images_to_low_water_and_keeps_turns(tmp_path):
    w = ContextWindow("system", "task", "bounded", 100000, 128, max_images=4)
    w.append(_image_turn("old", 3))
    w.append(_image_turn("new", 2))

    async def counter(*_):
        return 1000

    journal = Journal(tmp_path)
    messages = await w.build(counter, [], journal)
    # Over budget: omit down to half the budget, never into the newest turn.
    assert _urls(messages) == ["new", "new"]
    assert any(m.get("content") == "old" for m in messages)
    w.append(_image_turn("newest", 1))
    messages = await w.build(counter, [], journal)
    # Within budget again: nothing new is omitted and omitted images never reappear,
    # so the previous request remains a prefix of this one.
    assert _urls(messages) == ["new", "new", "newest"]


async def test_prune_leaves_headroom_so_next_turn_keeps_the_prefix(tmp_path):
    w = ContextWindow("system", "task", "bounded", 1384, 128)  # input budget 1000
    for i in range(7):
        w.append([{"role": "assistant", "content": f"turn {i}"}, {"role": "tool", "content": "x"}], {"s": i})

    async def counter(messages, tools):
        return 150 * sum(m["role"] == "assistant" for m in messages)

    journal = Journal(tmp_path)
    first = await w.build(counter, [], journal)
    assert 150 * sum(m["role"] == "assistant" for m in first) <= 750
    w.append([{"role": "assistant", "content": "turn 7"}, {"role": "tool", "content": "x"}], {"s": 7})
    second = await w.build(counter, [], journal)
    history = [m for m in first if m["role"] in ("assistant", "tool")]
    assert [m for m in second if m["role"] in ("assistant", "tool")][: len(history)] == history


async def test_newest_turn_over_image_budget_is_a_distinct_failure(tmp_path):
    w = ContextWindow("system", "task", "bounded", 100000, 128, max_images=2)
    w.append(_image_turn("huge", 3))

    async def counter(*_):
        return 1000

    with pytest.raises(ImageBudget):
        await w.build(counter, [], Journal(tmp_path))


async def test_context_overflow_without_safe_summary(tmp_path):
    w = ContextWindow("system", "task", "bounded", 2048, 128)
    w.append([{"role": "user", "content": "history"}])

    async def counter(*_):
        return 2000

    with pytest.raises(ContextOverflow):
        await w.build(counter, [], Journal(tmp_path))


async def test_wire_requests_and_model_retry_are_recorded(tmp_path):
    p = PlannerProfile(model="test")
    client = CompatibleClient(p)
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]},
        )

    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    journal = Journal(tmp_path)
    await client.generate([{"role": "user", "content": "hello"}], [], journal)
    await client.close()
    events = read_jsonl(journal.path)
    assert events[0]["request"] == seen[0]
    assert events[1]["kind"] == "model_response"


def test_api_open_access_and_unknown_templates(tmp_path):
    app = create_app(tmp_path)
    assert not (tmp_path / "admin-password").exists()
    with TestClient(app) as c:
        assert c.get("/api/services").status_code == 200
        spec = {"name": "test-env", "kind": "environment", "url": "http://localhost:19200"}
        response = c.post("/api/services", json=spec)
        assert response.status_code == 201
        assert c.get("/api/services").json()[0]["id"] == response.json()["id"]
        assert c.post(
            "/api/services", json={**spec, "managed": True, "template": "unknown"}
        ).status_code == 422
        assert "/api/login" not in app.openapi()["paths"]
        assert "/api/logout" not in app.openapi()["paths"]


