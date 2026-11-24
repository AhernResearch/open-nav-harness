"""The dialogue model/tool loop: one operator turn runs until the model replies without tools."""

import json
import traceback
from dataclasses import dataclass, field

import jsonschema
from open_nav_harness.core.artifacts import Journal
from open_nav_harness.core.config import PlannerProfile
from open_nav_harness.core.context import ContextWindow
from .compatible import CompatibleClient, ModelProtocolError
from .validation import StrictValidator


@dataclass
class Outcome:
    turns: int = 0
    usage: dict = field(default_factory=dict)
    error: str | None = None


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
    """Runs until the model answers without tool calls; a newer operator message skips the rest
    of an unexecuted batch. A motion call is never retried: a lost response has unknown effects."""

    def __init__(self, profile: PlannerProfile):
        self.profile = profile

    async def run(self, ctx, sink, session, window: ContextWindow | None = None) -> Outcome:
        journal = Journal(ctx.workdir / "trace")
        window = window or ContextWindow(ctx.briefing, ctx.first_prompt, self.profile.context_policy,
                                         self.profile.context_limit, self.profile.output_tokens,
                                         self.profile.max_images)
        client = CompatibleClient(self.profile, journal=journal)
        usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        seen_ids = {c["id"] for block in window.blocks for m in block for c in (m.get("tool_calls") or [])}
        outcome = Outcome(usage=usage, error="planner_turn_budget_exhausted")
        try:
            listed = await session.list_tools()
            definitions = {t.name: t.inputSchema for t in listed}
            tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                       "parameters": t.inputSchema}} for t in listed]
            journal.emit("session_start", model=self.profile.model, tools=tools,
                         input_visibility="exact_wire_requests")
            for _ in range(ctx.max_turns):
                sink.receive(window)
                messages = await window.build(client.count, tools, journal)
                response = await client.generate(messages, tools, journal)
                outcome.turns += 1
                u = response.get("usage") or {}
                usage["input_tokens"] += u.get("prompt_tokens", 0)
                usage["cached_input_tokens"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                usage["output_tokens"] += u.get("completion_tokens", 0)
                message = response["choices"][0]["message"]
                calls = message.get("tool_calls") or []
                if message.get("content"):
                    sink.emit("assistant_text", {"text": message["content"]})
                if not calls:
                    window.append([message])
                    outcome.error = None
                    break
                block = [{"role": "assistant", "content": message.get("content"), "tool_calls": calls}]
                if "provider_items" in message:
                    block[0]["provider_items"] = message["provider_items"]
                prepared, errors = self._validate(calls, definitions, seen_ids)
                if errors:
                    self._reject(calls, errors, block, journal, sink)
                else:
                    await self._execute(prepared, block, session, journal, sink)
                window.append(block)
        except Exception as exc:
            outcome.error = describe(exc)
            journal.emit("session_error", error=outcome.error, traceback="".join(traceback.format_exception(exc)))
        finally:
            await client.close()
        journal.emit("session_end", error=outcome.error, usage=usage)
        return outcome

    @staticmethod
    def _validate(calls, definitions, seen_ids):
        prepared, errors = [], []
        for call in calls:
            ident, name = call["id"], call["function"]["name"]
            if ident in seen_ids:
                raise ModelProtocolError("Repeated tool call id; refusing duplicate execution")
            seen_ids.add(ident)
            try:
                arguments = json.loads(call["function"]["arguments"])
                if name not in definitions:
                    raise ValueError("Unknown tool: " + name)
                StrictValidator(definitions[name]).validate(arguments)
                prepared.append((ident, name, arguments))
            except (ValueError, jsonschema.ValidationError) as exc:
                errors.append(name + ": " + getattr(exc, "message", str(exc)))
        return prepared, errors

    @staticmethod
    def _reject(calls, errors, block, journal, sink):
        """Reject the entire batch before any motion; the model repairs it in a new call."""
        text = "No tools in this batch were executed. Correct the request: " + "; ".join(errors)
        journal.emit("tool_validation_error", errors=errors)
        for call in calls:
            sink.emit("tool_use", {"id": call["id"], "name": call["function"]["name"],
                                   "input": call["function"]["arguments"]})
        for call in calls:
            feedback = {"role": "tool", "tool_call_id": call["id"], "content": text}
            block.append(feedback)
            journal.emit("tool_result", call_id=call["id"], name=call["function"]["name"], is_error=True,
                         executed=False, feedback=feedback, images=[])
            sink.emit("tool_result", {"tool_use_id": call["id"], "texts": [text], "is_error": True})

    @staticmethod
    async def _execute(prepared, block, session, journal, sink):
        observations = []
        for ident, name, arguments in prepared:
            sink.emit("tool_use", {"id": ident, "name": name, "input": arguments})
        for ident, name, arguments in prepared:
            journal.emit("tool_dispatch", call_id=ident, name=name, arguments=arguments)
            result = sink.skipped() if sink.pending() else await session.call_tool(name, arguments)
            feedback, images, texts = tool_feedback(result, ident)
            journal.emit("tool_result", call_id=ident, name=name, is_error=bool(result.isError),
                         feedback=feedback, images=images)
            sink.emit("tool_result", {"tool_use_id": ident, "texts": texts, "images": images,
                                      "is_error": bool(result.isError)})
            block.append(feedback)
            observations.extend(images)
        if observations:
            block.append({"role": "user", "content": observations})


def describe(error) -> str:
    nested = getattr(error, "exceptions", None)
    return "; ".join(describe(e) for e in nested) if nested else f"{type(error).__name__}: {error}"
