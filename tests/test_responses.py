import base64
import io
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from open_nav_harness.core.artifacts import Journal
from open_nav_harness.core.config import PlannerProfile
from open_nav_harness.providers.compatible import CompatibleClient, ModelProtocolError
from open_nav_harness.providers.responses import request_input, normalize_response
from open_nav_harness.providers.vision import vision_tokens


TOOLS = [{'type': 'function', 'function': {'name': 'observe', 'description': 'Look',
          'parameters': {'type': 'object', 'properties': {}}}}]
OUTPUT = [{'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'encrypted'},
          {'type': 'function_call', 'id': 'fc_1', 'call_id': 'call_1',
           'name': 'observe', 'arguments': '{}', 'status': 'completed'}]


def test_responses_roundtrip_preserves_reasoning_tools_and_images():
    body = {'status': 'completed', 'output': OUTPUT,
            'usage': {'input_tokens': 123, 'output_tokens': 45,
                      'input_tokens_details': {'cached_tokens': 12}}}
    result = normalize_response(body)
    message = result['choices'][0]['message']
    assert result['usage']['prompt_tokens_details']['cached_tokens'] == 12
    messages = [{'role': 'system', 'content': 'Navigate'}, message,
                {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'observed'},
                {'role': 'user', 'content': [{'type': 'text', 'text': 'front'},
                 {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc', 'detail': 'high'}}]}]
    payload = request_input(messages, TOOLS)
    assert payload['input'][1:3] == OUTPUT
    assert payload['input'][3] == {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'observed'}
    assert payload['input'][4]['content'][1]['detail'] == 'high'
    assert payload['tools'][0]['strict'] is False
    assert TOOLS[0]['function'].get('strict') is None


@pytest.mark.parametrize('replay_reasoning', [True, False])
async def test_responses_wire_uses_native_fields_and_normalizes_usage(tmp_path, replay_reasoning):
    captured = []

    def handler(request):
        captured.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith('/input_tokens'):
            return httpx.Response(200, json={'input_tokens': 120})
        return httpx.Response(200, json={'object': 'response', 'status': 'completed', 'output': OUTPUT,
                                        'usage': {'input_tokens': 120, 'output_tokens': 30}})

    journal = Journal(tmp_path / 'trace')
    client = CompatibleClient(PlannerProfile(model='test', api_format='responses', effort='low',
                                             temperature=None, replay_reasoning=replay_reasoning), journal)
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        assert await client.count([{'role': 'user', 'content': 'Go'}], TOOLS) == 120
        result = await client.generate([{'role': 'user', 'content': 'Go'}], TOOLS, journal)
        assert result['choices'][0]['message']['tool_calls'][0]['id'] == 'call_1'
        assert result['usage']['completion_tokens'] == 30
    finally:
        await client.close()
    assert captured[0][0] == '/v1/responses/input_tokens'
    wire = captured[1][1]
    assert wire['reasoning'] == {'effort': 'low'}
    assert wire['store'] is False
    assert wire.get('include') == (['reasoning.encrypted_content'] if replay_reasoning else None)
    assert not {'messages', 'chat_template_kwargs', 'max_tokens', 'temperature'} & wire.keys()
    events = [json.loads(l) for l in journal.path.read_text().splitlines()]
    assert next(e for e in events if e['kind'] == 'model_response')['response']['output'] == OUTPUT


def test_gateway_can_omit_reasoning_without_dropping_calls_or_mutating_trace():
    message = normalize_response({'status': 'completed', 'output': OUTPUT})['choices'][0]['message']
    payload = request_input([message, {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'observed'}],
                            TOOLS, replay_reasoning=False)
    assert [item['type'] for item in payload['input']] == ['function_call', 'function_call_output']
    assert payload['input'][0]['call_id'] == payload['input'][1]['call_id'] == 'call_1'
    assert message['provider_items'] == OUTPUT


def png(width, height):
    stream = io.BytesIO()
    Image.new('RGB', (width, height)).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()


@pytest.mark.parametrize('size,tokens', [((480, 270), 122), ((640, 480), 302), ((1280, 720), 882),
                                          ((128, 96), 72), ((1920, 1080), 2042), ((4000, 3000), 11752)])
def test_vision_tokens_match_the_qwen_processor_measured_through_vllm(size, tokens):
    # Measured with the 27B vLLM /tokenize: the per-image increase over a text-only message.
    assert vision_tokens(*size) == tokens


async def test_explicit_estimate_budgets_images_by_resolution_and_never_calls_server(tmp_path):
    client = CompatibleClient(PlannerProfile(model='test', token_count_mode='estimate'))
    client._post_inference = AsyncMock()
    try:
        def messages(*urls):
            return [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': u}}
                                                 for u in urls]}]
        async def text(n):  # the same request with each image replaced by its placeholder
            return await client.count([{'role': 'user', 'content': [{'type': 'image'}] * n}], TOOLS)
        small = await client.count(messages(png(480, 270)), TOOLS)
        large = await client.count(messages(png(1920, 1080)), TOOLS)
        # Base64 length never counts as text; only the resolution does.
        assert small - await text(1) == 122 and large - await text(1) == 2042
        assert await client.count(messages(png(480, 270), png(480, 270)), TOOLS) - await text(2) == 2 * 122
        with pytest.raises(ValueError, match='inline'):
            await client.count(messages('https://example.com/a.png'), TOOLS)
        client._post_inference.assert_not_called()
    finally:
        await client.close()


@pytest.mark.parametrize('status', ['incomplete', 'failed', 'in_progress'])
def test_partial_response_never_executes_tools(status):
    with pytest.raises(ValueError, match='did not complete'):
        normalize_response({'status': status, 'output': OUTPUT})


def test_invalid_response_body_is_protocol_error():
    with pytest.raises(ValueError, match='did not return an object'):
        normalize_response([])


async def test_remote_count_failure_does_not_silently_estimate():
    client = CompatibleClient(PlannerProfile(model='test', api_format='responses'))
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(404)))
    try:
        with pytest.raises(ModelProtocolError, match='404'):
            await client.count([], [])
    finally:
        await client.close()


async def test_estimate_counts_the_responses_body_actually_sent():
    """Unreplayed reasoning is never sent, so it is never counted; images count by resolution."""
    reasoning = {'type': 'reasoning', 'id': 'rs', 'summary': [], 'encrypted_content': 'e' * 20000}
    call = {'type': 'function_call', 'id': 'fc', 'call_id': 'c1', 'name': 'observe', 'arguments': '{}'}

    def history(items):
        return [{'role': 'user', 'content': 'go'},
                {'role': 'assistant', 'content': None, 'provider_items': items,
                 'tool_calls': [{'id': 'c1', 'type': 'function',
                                 'function': {'name': 'observe', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'},
                {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': png(480, 270)}}]}]

    counts = {}
    for replay in (False, True):
        client = CompatibleClient(PlannerProfile(model='test', token_count_mode='estimate',
                                                 api_format='responses', replay_reasoning=replay))
        try:
            counts[replay] = (await client.count(history([reasoning, call]), TOOLS),
                              await client.count(history([call]), TOOLS))
        finally:
            await client.close()
    assert counts[False][0] == counts[False][1]          # dropped reasoning costs nothing
    assert counts[True][0] > counts[True][1] + 20000     # replayed reasoning is counted
    image_free = counts[False][1] - 122
    assert 0 < image_free < 2000                         # the image is 122, not its base64
