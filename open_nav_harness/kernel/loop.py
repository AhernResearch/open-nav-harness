"""The explicit model/tool loop of a kernel episode.

One response is validated as a whole before anything executes. A malformed response
(bad arguments, or a tool call written as text) gets repair feedback and executes
nothing. A reply with neither is a recorded failure: only ``stop`` ends an episode.
"""

from __future__ import annotations

import base64
import json
import re

import jsonschema

from open_nav_harness.core.context import ContextWindow
from open_nav_harness.providers.compatible import CompatibleClient
from open_nav_harness.providers.validation import StrictValidator

from .environment import SimulatorError
from .tools import TOOL_SETS, EpisodeState, ToolOutput, ToolRefused, legs_summary, tool_message

# A tool call the serving parser did not extract, e.g. Qwen XML written into the content.
TOOL_TEXT = re.compile(r"<tool_call>|<function=|\b(observe|delegate|return_to|update_plan|stop)\s*\(")
REPAIR = ("Your reply contained no executable tool call. Nothing was executed. "
          "Call one of the tools with the tool-calling format.")


class AgentLoop:
    def __init__(self, spec, journal, transport=None):
        self.spec, self.journal = spec, journal
        self.tools = {tool.name: tool for tool in TOOL_SETS[spec.tool_set]}
        self.definitions = [tool.definition() for tool in self.tools.values()]
        self.client = CompatibleClient(spec.planner, journal=journal, transport=transport)
        self.turns = self.malformed = 0
        self.tokens = {"input_tokens": 0, "output_tokens": 0}
        # Unknown, not zero, once any response omits its cache detail.
        self.cached: int | None = 0

    async def run(self, state: EpisodeState, system: str, first: str) -> str | None:
        """Drive the episode until a tool ends it; returns a failure reason or None."""
        profile = self.spec.planner
        window = ContextWindow(system, first, profile.context_policy, profile.context_limit,
                               profile.output_tokens, profile.max_images)
        self.journal.emit("session_start", model=profile.model, tools=self.definitions,
                          input_visibility="exact_wire_requests")
        seen: set[str] = set()
        consecutive = 0
        try:
            for _ in range(self.spec.budgets.max_turns):
                self.turns += 1
                messages = await window.build(self.client.count, self.definitions, self.journal)
                response = await self.client.generate(messages, self.definitions, self.journal)
                self._usage(response.get("usage") or {})
                message = response["choices"][0]["message"]
                calls = message.get("tool_calls") or []
                block: list[dict] = [{"role": "assistant", "content": message.get("content")}]
                if calls:
                    block[0]["tool_calls"] = calls
                if "provider_items" in message:
                    block[0]["provider_items"] = message["provider_items"]
                if not calls:
                    if not TOOL_TEXT.search(message.get("content") or ""):
                        return "planner_ended_without_stop"
                    self.malformed += 1
                    consecutive += 1
                    self.journal.emit("malformed_call", consecutive=consecutive)
                    if consecutive >= self.spec.budgets.max_malformed:
                        return "planner_protocol_failure"
                    block.append({"role": "user", "content": REPAIR})
                    window.append(block, self._state(state))
                    continue
                consecutive = 0
                prepared, errors = self._validate(calls, seen)
                if errors:
                    self.malformed += 1
                    text = "No tools in this reply were executed. Correct the request: " + "; ".join(errors)
                    self.journal.emit("tool_validation_error", errors=errors)
                    for call in calls:
                        block.append({"role": "tool", "tool_call_id": call["id"], "content": text})
                        self.journal.emit("tool_result", call_id=call["id"], name=call["function"]["name"],
                                          is_error=True, executed=False, feedback=block[-1], images=[])
                    window.append(block, self._state(state))
                    continue
                images: list[dict] = []
                ended = False
                for ident, name, arguments in prepared:
                    self.journal.emit("tool_dispatch", call_id=ident, name=name, arguments=arguments)
                    state.evidence.call = ident
                    output, is_error = await self._execute(name, state, arguments)
                    feedback = {"role": "tool", "tool_call_id": ident, "content": tool_message(output)}
                    block.append(feedback)
                    shown = [item for label, png in output.images for item in (
                        {"type": "text", "text": f"Tool {ident}: {label}"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                                                            + base64.b64encode(png).decode()}})]
                    images.extend(shown)
                    self.journal.emit("tool_result", call_id=ident, name=name, is_error=is_error,
                                      feedback=feedback, images=shown)
                    if output.ends_episode:
                        ended = True
                        break
                if images:
                    block.append({"role": "user", "content": images})
                window.append(block, self._state(state))
                if ended:
                    return None
            return "planner_turn_budget_exhausted"
        finally:
            await self.client.close()

    async def _execute(self, name, state, arguments) -> tuple[ToolOutput, bool]:
        try:
            return await self.tools[name].run(state, arguments), False
        except (ToolRefused, SimulatorError) as error:
            # Refusals and simulator 4xx replies are known not to have moved the robot.
            return ToolOutput({"error": str(error)}), True

    def _validate(self, calls, seen):
        prepared, errors = [], []
        for call in calls:
            ident, function = call["id"], call["function"]
            if ident in seen:
                errors.append("repeated tool call id " + ident)
                continue
            seen.add(ident)
            try:
                arguments = json.loads(function["arguments"] or "{}")
                tool = self.tools.get(function["name"])
                if tool is None:
                    raise ValueError("unknown tool " + function["name"])
                StrictValidator(tool.parameters).validate(arguments)
                prepared.append((ident, function["name"], arguments))
            except (ValueError, jsonschema.ValidationError) as error:
                errors.append(function["name"] + ": " + getattr(error, "message", str(error)))
        if any(name == "stop" for _, name, _ in prepared[:-1]):
            errors.append("stop must be the last tool in a reply")
        return prepared, errors

    @property
    def usage(self) -> dict:
        return {**self.tokens, "cached_input_tokens": self.cached}

    def _usage(self, usage):
        self.tokens["input_tokens"] += usage.get("prompt_tokens") or 0
        self.tokens["output_tokens"] += usage.get("completion_tokens") or 0
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        self.cached = None if cached is None or self.cached is None else self.cached + cached

    @staticmethod
    def _state(state: EpisodeState) -> dict:
        """Pinned when older turns are pruned; the model never loses its plan or legs."""
        return {"plan": state.plan, "legs": legs_summary(state.legs)}
