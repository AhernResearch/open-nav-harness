"""Transient inference faults must not kill episodes or replay navigation."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from open_nav_harness.core.artifacts import Journal
from open_nav_harness.core.config import PlannerProfile
from open_nav_harness.core.context import CounterLimit
from open_nav_harness.providers.compatible import CompatibleClient, ModelProtocolError


@pytest.fixture
def retry_sleep(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("open_nav_harness.providers.compatible.asyncio.sleep", sleep)
    return sleep


async def client_for(handler, tmp_path):
    journal = Journal(tmp_path / "trace")
    client = CompatibleClient(PlannerProfile(model="test"), journal)
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client, journal


async def test_counter_recovers_disconnect_and_unavailable_without_changing_payload(tmp_path, retry_sleep):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            raise httpx.RemoteProtocolError("Server disconnected", request=request)
        if len(requests) == 2:
            return httpx.Response(503, text="unavailable", headers={"content-type": "application/json"})
        return httpx.Response(200, json={"count": 123, "max_model_len": 16384})

    client, journal = await client_for(handler, tmp_path)
    try:
        assert await client.count([{"role": "user", "content": "route"}], []) == 123
    finally:
        await client.close()
    assert len(requests) == 3
    assert len({request.content for request in requests}) == 1
    assert all(request.url.path == "/tokenize" for request in requests)
    assert all(request.extensions["timeout"]["read"] == 15 for request in requests)
    events = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert [e["attempt"] for e in events] == [1, 2, 3]
    assert events[0]["kind"] == "token_count_transport_error"
    assert all(e["elapsed_s"] >= 0 for e in events)
    assert retry_sleep.await_count == 2


@pytest.mark.parametrize("status,body,error", [
    (400, {"error": "maximum context length exceeded"}, CounterLimit),
    (401, {"error": "unauthorized"}, ModelProtocolError),
    (200, {"count": True}, ModelProtocolError),
    (200, [], ModelProtocolError),
])
async def test_counter_does_not_retry_permanent_errors(tmp_path, retry_sleep, status, body, error):
    client, _ = await client_for(lambda _: httpx.Response(status, json=body), tmp_path)
    try:
        with pytest.raises(error):
            await client.count([], [])
    finally:
        await client.close()
    retry_sleep.assert_not_awaited()


@pytest.mark.parametrize("failure", ["transport", "status", "cancel"])
async def test_counter_bounds_retries_and_propagates_cancellation(tmp_path, retry_sleep, failure):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if failure == "transport":
            raise httpx.ReadTimeout("slow counter", request=request)
        if failure == "cancel":
            raise asyncio.CancelledError()
        return httpx.Response(429, json={"error": "busy"})

    client, _ = await client_for(handler, tmp_path)
    error = {"transport": httpx.ReadTimeout, "status": ModelProtocolError,
             "cancel": asyncio.CancelledError}[failure]
    try:
        with pytest.raises(error):
            await client.count([], [])
    finally:
        await client.close()
    assert attempts == (1 if failure == "cancel" else 3)
    assert retry_sleep.await_count == (0 if failure == "cancel" else 2)


async def test_generation_retries_before_returning_one_tool_call(tmp_path, retry_sleep):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(502)
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "tool_calls": [{"id": "move-once", "type": "function",
            "function": {"name": "execute_subtask", "arguments": '{"task_id":1}'}}],
        }}]})

    client, journal = await client_for(handler, tmp_path)
    try:
        result = await client.generate([], [], journal)
    finally:
        await client.close()
    assert attempts == 2
    assert len(result["choices"][0]["message"]["tool_calls"]) == 1
    events = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert [e["kind"] for e in events] == ["model_request", "model_response"] * 2
