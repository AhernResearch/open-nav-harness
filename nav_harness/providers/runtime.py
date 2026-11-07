"""Managed model/tool loop, sharing the original MCP navigation tools."""

import json
import os
import traceback
from contextlib import AsyncExitStack
import jsonschema
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from navgpt.planner.contracts import SessionOutcome
from navgpt.planner.runner import expected_tools, mcp_launch
from nav_harness.core.artifacts import Journal
from nav_harness.core.config import PlannerProfile
from nav_harness.core.context import ContextWindow
from .compatible import CompatibleClient, ModelProtocolError


def tool_feedback(result, call_id):
    """Chat tool messages are text-only; image observations follow the batch.

    Every image retains its originating call id. This explicit mapping is
    recorded in the wire request, never hidden from trace consumers.
    """
    texts, images = [], []
    for item in result.content:
        if item.type == "text":
            texts.append(item.text)
        elif item.type == "image":
            images.extend(
                [
                    {"type": "text", "text": "Observation from tool call " + call_id},
                    {"type": "image_url", "image_url": {"url": f"data:{item.mimeType};base64,{item.data}"}},
                ]
            )
        else:
            raise ModelProtocolError("Unsupported tool result content: " + item.type)
    return {"role": "tool", "tool_call_id": call_id, "content": "\n".join(texts)}, images, texts


class ManagedRuntime:
    name = "openai-compatible"

    def __init__(self, profile: PlannerProfile):
        self.profile = profile

    def prepare(self, cfg):
        if cfg.max_budget_usd is not None:
            raise ValueError("Unknown billing: use request/time budgets, not a USD ceiling")

    def describe(self, ctx):
        return {
            "planner_runtime": self.name,
            "input_visibility": "exact_wire_requests",
            "profile": self.profile.model_dump(exclude={"credential_env"}),
        }

    async def run(self, ctx, sink):
        journal = Journal(ctx.workdir / "trace")
        window = ContextWindow(
            ctx.briefing,
            ctx.first_prompt,
            self.profile.context_policy,
            self.profile.context_limit,
            self.profile.output_tokens,
            self.profile.max_images,
        )
        client = CompatibleClient(self.profile)
        usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        seen_ids = set()
        error = None
        stopped = False
        try:
            async with AsyncExitStack() as stack:
                spec = mcp_launch(ctx.mcp_env)
                transport = await stack.enter_async_context(
                    stdio_client(
                        StdioServerParameters(
                            command=spec["command"], args=spec["args"], env={**os.environ, **spec["env"]}
                        )
                    )
                )
                session = await stack.enter_async_context(ClientSession(*transport))
                await session.initialize()
                listed = (await session.list_tools()).tools
                definitions = {t.name: t.inputSchema for t in listed}
                want = expected_tools(
                    ctx.mcp_env["NAVGPT_CONDITION"], ctx.mcp_env.get("NAVGPT_CAPABILITY_BUNDLE", "full")
                )
                if set(definitions) != set(want):
                    raise ModelProtocolError("MCP tool set differs from the configured contract")
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.inputSchema,
                        },
                    }
                    for t in listed
                ]
                journal.emit(
                    "session_start",
                    model=self.profile.model,
                    tools=tools,
                    input_visibility="exact_wire_requests",
                )
                for _ in range(ctx.max_turns):
                    messages = await window.build(client.count, tools, journal)
                    response = await client.generate(messages, tools, journal)
                    u = response.get("usage") or {}
                    usage["input_tokens"] += u.get("prompt_tokens", 0)
                    usage["cached_input_tokens"] += (u.get("prompt_tokens_details") or {}).get(
                        "cached_tokens", 0
                    )
                    usage["output_tokens"] += u.get("completion_tokens", 0)
                    message = response["choices"][0]["message"]
                    calls = message.get("tool_calls") or []
                    if message.get("content"):
                        sink.emit("assistant_text", {"text": message["content"]})
                    if not calls:
                        raise ModelProtocolError("Planner ended without an explicit successful STOP tool")
                    prepared = []
                    for call in calls:
                        ident, fn = call["id"], call["function"]
                        name = fn["name"]
                        if ident in seen_ids:
                            raise ModelProtocolError("Repeated tool call id; refusing duplicate execution")
                        seen_ids.add(ident)
                        arguments = json.loads(fn["arguments"])
                        if name not in definitions:
                            raise ModelProtocolError("Unknown navigation tool: " + name)
                        jsonschema.validate(arguments, definitions[name])
                        prepared.append((ident, name, arguments))
                    if any(name == "terminate_episode" for _, name, _ in prepared[:-1]):
                        raise ModelProtocolError("STOP must be the last tool in a response")
                    block = [{"role": "assistant", "content": message.get("content"), "tool_calls": calls}]
                    observations = []
                    state = None
                    for ident, name, arguments in prepared:
                        sink.emit("tool_use", {"id": ident, "name": name, "input": arguments})
                    for ident, name, arguments in prepared:
                        journal.emit("tool_dispatch", call_id=ident, name=name, arguments=arguments)
                        # A motion call is never retried: a lost response has unknown effects.
                        result = await session.call_tool(name, arguments)
                        feedback, images, texts = tool_feedback(result, ident)
                        journal.emit(
                            "tool_result",
                            call_id=ident,
                            name=name,
                            is_error=bool(result.isError),
                            feedback=feedback,
                            images=images,
                        )
                        sink.emit(
                            "tool_result",
                            {"tool_use_id": ident, "texts": texts, "is_error": bool(result.isError)},
                        )
                        block.append(feedback)
                        observations.extend(images)
                        for text in texts:
                            try:
                                value = json.loads(text)
                                if isinstance(value, dict):
                                    state = value.get("subtask_state", state)
                            except ValueError:
                                pass
                        if (
                            name == "terminate_episode"
                            and sink.last_step_result
                            and sink.last_step_result.get("end_reason") == "stop_called"
                        ):
                            stopped = True
                            break
                    if observations:
                        block.append({"role": "user", "content": observations})
                    window.append(block, state)
                    if stopped:
                        break
                if not stopped:
                    error = "planner_turn_budget_exhausted"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            journal.emit("session_error", error=error, traceback="".join(traceback.format_exception(exc)))
        finally:
            await client.close()
        journal.emit("session_end", error=error, explicit_stop=stopped, usage=usage)
        return SessionOutcome(
            turns=sink.planner_turns.count,
            usage=usage,
            error=error,
            extra={"input_visibility": "exact_wire_requests", "trace": str(journal.path)},
        )
