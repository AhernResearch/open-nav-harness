"""One HTTP adapter for image-capable OpenAI-compatible endpoints."""

import asyncio
import json
import os
import time
import httpx
from open_nav_harness.core.config import PlannerProfile
from open_nav_harness.core.context import CounterLimit
from .responses import request_input, normalize_response
from .vision import image_size, vision_tokens


class ModelProtocolError(RuntimeError):
    pass


class CompatibleClient:
    def __init__(self, profile: PlannerProfile, journal=None, transport=None):
        self.profile = profile
        self.journal = journal
        headers = {}
        if profile.credential_env:
            key = os.environ.get(profile.credential_env)
            if not key:
                raise ValueError("Missing credential environment variable: " + profile.credential_env)
            headers["Authorization"] = "Bearer " + key
        self.http = httpx.AsyncClient(timeout=profile.request_timeout_s, headers=headers, trust_env=False,
                                      transport=transport)

    async def close(self):
        await self.http.aclose()

    async def _post_inference(self, url, payload, *, operation, journal, timeout=None):
        """Retry only read-only inference requests, never navigation/tool execution."""
        for attempt in range(1, 4):
            if journal and operation == "model":
                journal.emit("model_request", request=payload, attempt=attempt)
            started = time.monotonic()
            try:
                options = {"timeout": timeout} if timeout is not None else {}
                response = await self.http.post(url, json=payload, **options)
            except httpx.RequestError as exc:
                if journal:
                    journal.emit(
                        operation + "_transport_error", error=f"{type(exc).__name__}: {exc}",
                        elapsed_s=time.monotonic() - started, attempt=attempt,
                    )
                if attempt == 3:
                    raise
            else:
                if journal:
                    try:
                        body = response.json()
                    except ValueError:
                        body = {"error": response.text}
                    if operation == "model":
                        fields = {"response": body}
                    elif isinstance(body, dict):
                        fields = {key: body.get(key) for key in ("count", "input_tokens", "max_model_len", "error")}
                    else:
                        fields = {"error": "Non-object tokenizer response"}
                    journal.emit(
                        "model_response" if operation == "model" else "token_count",
                        status=response.status_code, elapsed_s=time.monotonic() - started,
                        attempt=attempt, **fields,
                    )
                if response.status_code not in (429, 502, 503, 504) or attempt == 3:
                    return response
            await asyncio.sleep(2 ** (attempt - 1))
        raise ModelProtocolError("Request attempts exhausted")

    async def count(self, messages: list, tools: list) -> int:
        p = self.profile
        if p.token_count_mode == 'estimate':
            # Explicit opt-in for gateways without counting. Measure the body actually sent
            # (Responses drops unreplayed reasoning), never tokenize base64 as text, and
            # budget each image from its own resolution.
            wire = (request_input(messages, tools, replay_reasoning=p.replay_reasoning)
                    if p.api_format == 'responses' else [messages, tools])
            sizes: list[tuple[int, int]] = []

            def without_images(value):
                if isinstance(value, dict):
                    if value.get('type') in ('image_url', 'input_image'):
                        url = value['image_url']
                        sizes.append(image_size(url['url'] if isinstance(url, dict) else url))
                        return {'type': 'image'}
                    return {k: without_images(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [without_images(v) for v in value]
                return value

            size = len(json.dumps(without_images(wire), ensure_ascii=False).encode())
            images = sum(vision_tokens(width, height) for width, height in sizes)
            estimated = size + images + 256
            if self.journal:
                self.journal.emit('token_count', count=estimated, method='estimate',
                                  text_bytes=size, images=len(sizes), image_tokens=images)
            return estimated
        if p.api_format == 'responses':
            payload = {'model': p.model, **request_input(messages, tools, replay_reasoning=p.replay_reasoning)}
            url = p.token_count_url or p.base_url.rstrip('/') + '/responses/input_tokens'
        else:
            payload = {'model': p.model, 'messages': messages, 'tools': tools,
                       'add_generation_prompt': True,
                       'chat_template_kwargs': {'enable_thinking': p.thinking}}
            url = p.token_count_url or p.base_url.rstrip('/').removesuffix('/v1') + '/tokenize'
        # vLLM's tokenize endpoint applies the actual processor/chat template,
        # including image placeholders. An unavailable counter fails closed.
        r = await self._post_inference(
            url,
            payload,
            operation="token_count", journal=self.journal,
            timeout=min(15, p.request_timeout_s),
        )
        if r.status_code == 400 and any(
            phrase in r.text.lower()
            for phrase in (
                "maximum context length",
                "max_model_len",
                "maximum model length",
                "at most 32 image",
            )
        ):
            raise CounterLimit(r.text)
        if r.is_error:
            raise ModelProtocolError(f"Tokenizer HTTP {r.status_code}: {r.text[:4000]}")
        try:
            body = r.json()
        except ValueError as exc:
            raise ModelProtocolError("Tokenizer endpoint returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise ModelProtocolError("Tokenizer endpoint did not return an object")
        count = body.get('input_tokens' if p.api_format == 'responses' else 'count')
        if type(count) is not int or count < 0:
            raise ModelProtocolError("Tokenizer endpoint did not return a valid count")
        return count

    async def generate(self, messages: list, tools: list, journal):
        p = self.profile
        if p.api_format == 'responses':
            payload = {'model': p.model, **request_input(messages, tools, replay_reasoning=p.replay_reasoning),
                       'tool_choice': 'auto', 'parallel_tool_calls': False,
                       'max_output_tokens': p.output_tokens, 'store': False}
            if p.replay_reasoning:
                payload['include'] = ['reasoning.encrypted_content']
            if p.effort:
                payload['reasoning'] = {'effort': p.effort}
            if p.temperature is not None:
                payload['temperature'] = p.temperature
            r = await self._post_inference(p.base_url.rstrip('/') + '/responses', payload,
                                           operation='model', journal=journal)
            r.raise_for_status()
            try:
                return normalize_response(r.json())
            except (ValueError, KeyError, TypeError) as exc:
                raise ModelProtocolError(str(exc)) from exc
        payload = {
            "model": p.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": p.temperature,
            "max_tokens": p.output_tokens,
            "chat_template_kwargs": {"enable_thinking": p.thinking},
        }
        if not tools:
            for key in ('tools', 'tool_choice', 'parallel_tool_calls'):
                payload.pop(key, None)
        elif p.omit_unique_items:
            def portable(value):
                if isinstance(value, dict):
                    return {k: portable(v) for k, v in value.items() if k != 'uniqueItems'}
                if isinstance(value, list):
                    return [portable(v) for v in value]
                return value
            payload['tools'] = portable(payload['tools'])
        r = await self._post_inference(
            p.base_url.rstrip("/") + "/chat/completions", payload,
            operation="model", journal=journal,
        )
        r.raise_for_status()
        body = r.json()
        choices = body.get("choices") or []
        if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
            raise ModelProtocolError("Expected exactly one assistant response")
        if choices[0].get("finish_reason") == "length":
            raise ModelProtocolError("Model response truncated before completion")
        return body
