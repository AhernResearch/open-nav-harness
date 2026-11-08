"""Managed model/tool loop, using transport-neutral navigation tools."""

import json
import traceback
import jsonschema
from navgpt.planner.contracts import SessionOutcome
from navgpt.planner.runner import expected_tools_from_env
from navgpt.planner.tools.transport import open_tools
from nav_harness.core.artifacts import Journal
from nav_harness.core.config import PlannerProfile
from nav_harness.core.context import ContextWindow
from .compatible import CompatibleClient, ModelProtocolError


def tool_feedback(result, call_id):
    """Chat tool messages are text-only; image observations follow the batch.

    Every image retains its originating call id. This explicit mapping is
    recorded in the wire request, never hidden from trace consumers.
    """
    texts, images, feedback_texts = [], [], []
    image_label = ""
    for item in result.content:
        if item.type == "text":
            texts.append(item.text)
            feedback_texts.append(item.text)
            # Only short prose immediately preceding an image is its label.
            # Structured status stays in the tool response for lifecycle handling.
            image_label = item.text if len(item.text) <= 300 and not item.text.lstrip().startswith(("{", "[")) else ""
        elif item.type == "image":
            if image_label:
                feedback_texts.pop()
            images.extend(
                [
                    {"type": "text", "text": "Tool " + call_id + (": " + image_label if image_label else "")},
                    {"type": "image_url", "image_url": {"url": f"data:{item.mimeType};base64,{item.data}"}},
                ]
            )
            image_label = ""
        else:
            raise ModelProtocolError("Unsupported tool result content: " + item.type)
    return {"role": "tool", "tool_call_id": call_id, "content": "\n".join(feedback_texts)}, images, texts


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
        client = CompatibleClient(self.profile, journal=journal)
        usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        seen_ids = set()
        error = None
        stopped = False
        terminal_reason = None
        try:
            async with open_tools(ctx.mcp_env) as session:
                listed = await session.list_tools()
                definitions = {t.name: t.inputSchema for t in listed}
                want = expected_tools_from_env(ctx.mcp_env)
                if set(definitions) != set(want):
                    raise ModelProtocolError("Navigation tool set differs from the configured contract")
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
                    validation_errors = []
                    for call in calls:
                        ident, fn = call["id"], call["function"]
                        name = fn["name"]
                        if ident in seen_ids:
                            raise ModelProtocolError("Repeated tool call id; refusing duplicate execution")
                        seen_ids.add(ident)
                        try:
                            arguments = json.loads(fn["arguments"])
                            if name not in definitions:
                                raise ValueError("Unknown navigation tool: " + name)
                            jsonschema.validate(arguments, definitions[name])
                            prepared.append((ident, name, arguments))
                        except (ValueError, jsonschema.ValidationError) as exc:
                            validation_errors.append(name + ": " + getattr(exc, "message", str(exc)))
                    if any(name == "terminate_episode" for _, name, _ in prepared[:-1]):
                        validation_errors.append("STOP must be the last tool in a response")
                    if {"observe_map", "update_memory"} <= {name for _, name, _ in prepared}:
                        validation_errors.append(
                            "observe_map and update_memory require separate model turns: read the map first"
                        )
                    block = [{"role": "assistant", "content": message.get("content"), "tool_calls": calls}]
                    if validation_errors:
                        # Reject the ENTIRE batch before any movement, then let the model
                        # repair its arguments in a new call. Never replay an executed call.
                        text = "No tools in this batch were executed. Correct the request: " + "; ".join(
                            validation_errors
                        )
                        journal.emit("tool_validation_error", errors=validation_errors)
                        for call in calls:
                            sink.emit(
                                "tool_use",
                                {
                                    "id": call["id"],
                                    "name": call["function"]["name"],
                                    "input": call["function"]["arguments"],
                                },
                            )
                        for call in calls:
                            ident, fn = call["id"], call["function"]
                            feedback = {"role": "tool", "tool_call_id": ident, "content": text}
                            block.append(feedback)
                            journal.emit(
                                "tool_result",
                                call_id=ident,
                                name=fn["name"],
                                is_error=True,
                                executed=False,
                                feedback=feedback,
                                images=[],
                            )
                            sink.emit(
                                "tool_result", {"tool_use_id": ident, "texts": [text], "is_error": True}
                            )
                        window.append(block)
                        continue
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
                        last = sink.last_step_result or {}
                        if last.get("episode_over") is True:
                            terminal_reason = last.get("end_reason") or "environment_ended"
                            stopped = terminal_reason == "stop_called"
                            journal.emit(
                                "environment_end", end_reason=terminal_reason,
                                explicit_stop=stopped, source_tool=name, call_id=ident,
                                steps_taken_total=last.get("steps_taken_total"),
                            )
                            break
                    if observations:
                        block.append({"role": "user", "content": observations})
                    window.append(block, state)
                    if terminal_reason is not None:
                        break
                if terminal_reason is None:
                    error = "planner_turn_budget_exhausted"
        except Exception as exc:

            def describe(error):
                nested = getattr(error, "exceptions", None)
                return (
                    "; ".join(describe(e) for e in nested) if nested else f"{type(error).__name__}: {error}"
                )

            error = describe(exc)
            journal.emit("session_error", error=error, traceback="".join(traceback.format_exception(exc)))
        finally:
            await client.close()
        journal.emit("session_end", error=error, explicit_stop=stopped,
                     environment_end_reason=terminal_reason, usage=usage)
        return SessionOutcome(
            turns=sink.planner_turns.count,
            usage=usage,
            error=error,
            extra={"input_visibility": "exact_wire_requests", "trace": str(journal.path),
                   "environment_end_reason": terminal_reason, "explicit_stop": stopped},
        )
