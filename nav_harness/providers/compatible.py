"""One HTTP adapter for image-capable OpenAI-compatible endpoints."""

import asyncio
import os
import time
import httpx
from nav_harness.core.config import PlannerProfile
from nav_harness.core.context import CounterLimit


class ModelProtocolError(RuntimeError):
    pass


class CompatibleClient:
    def __init__(self, profile: PlannerProfile):
        self.profile = profile
        headers = {}
        if profile.credential_env:
            key = os.environ.get(profile.credential_env)
            if not key:
                raise ValueError("Missing credential environment variable: " + profile.credential_env)
            headers["Authorization"] = "Bearer " + key
        self.http = httpx.AsyncClient(timeout=profile.request_timeout_s, headers=headers, trust_env=False)

    async def close(self):
        await self.http.aclose()

    async def count(self, messages: list, tools: list) -> int:
        p = self.profile
        url = p.token_count_url or p.base_url.removesuffix("/v1") + "/tokenize"
        # vLLM's tokenize endpoint applies the actual processor/chat template,
        # including image placeholders. An unavailable counter fails closed.
        r = await self.http.post(
            url,
            json={
                "model": p.model,
                "messages": messages,
                "tools": tools,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": p.thinking},
            },
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
        r.raise_for_status()
        body = r.json()
        count = body.get("count")
        if not isinstance(count, int) or count < 0:
            raise ModelProtocolError("Tokenizer endpoint did not return a valid count")
        return count

    async def generate(self, messages: list, tools: list, journal):
        p = self.profile
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
        for attempt in range(3):
            journal.emit("model_request", request=payload, attempt=attempt + 1)
            started = time.monotonic()
            try:
                r = await self.http.post(p.base_url.rstrip("/") + "/chat/completions", json=payload)
            except httpx.RequestError as e:
                journal.emit("model_transport_error", error=str(e), attempt=attempt + 1)
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)
                continue
            journal.emit(
                "model_response",
                status=r.status_code,
                elapsed_s=time.monotonic() - started,
                response=r.json() if "json" in r.headers.get("content-type", "") else r.text,
                attempt=attempt + 1,
            )
            if r.status_code == 429 or r.status_code in (502, 503, 504):
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
            r.raise_for_status()
            body = r.json()
            choices = body.get("choices") or []
            if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
                raise ModelProtocolError("Expected exactly one assistant response")
            if choices[0].get("finish_reason") == "length":
                raise ModelProtocolError("Model response truncated before completion")
            return body
        raise ModelProtocolError("Request attempts exhausted")
