"""Model-visible tools. A tool set is the permission boundary: only listed tools exist."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .environment import VIEWS, Observation, SimEngineEnvironment
from .evidence import Evidence
from .executors import Executor, LegAborted, LegRequest
from .recorder import Recorder
from .spec import Budgets

UNSTARTED = ("invalid_target", "unreachable")
VIEW_SCHEMA = {"type": "string", "enum": list(VIEWS)}


class ToolRefused(ValueError):
    """A precondition failed; nothing moved and the model may correct the call."""


@dataclass
class ToolOutput:
    payload: dict
    images: list[tuple[str, bytes]] = field(default_factory=list)  # (label, PNG)
    ends_episode: bool = False


@dataclass
class EpisodeState:
    instruction: str
    env: SimEngineEnvironment
    executor: Executor
    budgets: Budgets
    recorder: Recorder
    evidence: Evidence
    observation: Observation | None = None
    shown: set[str] = field(default_factory=set)
    plan: list[dict] = field(default_factory=list)
    legs: list[dict] = field(default_factory=list)
    stop: dict | None = None
    end_reason: str | None = None

    def current(self) -> Observation | None:
        valid = self.observation and self.observation.event_index == self.env.event_index
        return self.observation if valid else None

    async def look(self) -> Observation:
        if self.current() is None:
            self.observation, self.shown = await self.env.observe(), set()
            await self.evidence.observation(self.observation)
        assert self.observation is not None
        return self.observation


class Tool(ABC):
    name: str
    description: str
    parameters: dict

    def definition(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}

    @abstractmethod
    async def run(self, state: EpisodeState, args: dict) -> ToolOutput: ...


def views_output(observation: Observation, views) -> list[tuple[str, bytes]]:
    return [(f"{view} view", observation.images[view]) for view in views]


class Observe(Tool):
    name = "observe"
    description = "Photos from the chosen cameras at the robot's current position."
    parameters = {"type": "object", "additionalProperties": False, "required": ["views"],
                  "properties": {"views": {"type": "array", "items": VIEW_SCHEMA,
                                           "minItems": 1, "maxItems": 4, "uniqueItems": True}}}

    async def run(self, state, args):
        observation = await state.look()
        state.shown |= set(args["views"])
        return ToolOutput({"views": args["views"]}, views_output(observation, args["views"]))


class Motion(Tool):
    """Shared handling after any leg: record it, look around and summarize all legs."""

    async def finish_leg(self, state, reply, record):
        # The harness numbers legs: a simulator only numbers the legs it ran itself.
        leg = {**reply["leg"], "leg_id": len(state.legs) + 1}
        state.legs.append({**leg, **record})
        state.recorder.leg({**leg, **record, "start_views": state.recorder.views(state.observation)
                            if state.observation and leg["kind"] != "retrace" else None})
        await state.evidence.leg(leg)
        if reply["done"]:
            state.end_reason = reply["termination"] or "episode_budget"
            return ToolOutput({"leg": leg, "episode_over": True}, ends_episode=True)
        if leg["status"] in UNSTARTED:
            return ToolOutput({"leg": leg, "moved": False})
        observation = await state.look()
        state.shown = set(VIEWS)
        return ToolOutput({"leg": leg, "legs_so_far": legs_summary(state.legs),
                           "control_steps_used": state.env.event_index},
                          views_output(observation, VIEWS))

    async def run_executor(self, state, request, record):
        try:
            reply = await state.executor.run_leg(state.env, request)
        except LegAborted as aborted:
            await self.finish_leg(state, aborted.reply, record)  # the motion happened: record it
            raise aborted.cause from None
        return await self.finish_leg(state, reply, record)


def legs_summary(legs):
    lines = []
    for leg in legs:
        what = (f"back to the end of leg {leg['retrace_of']}" if leg["kind"] == "retrace"
                else leg["instruction"])
        lines.append(f"{leg['leg_id']} {leg['status']}: {what}")
    return lines


class Delegate(Motion):
    name = "delegate"
    description = ("Send the walking controller to a floor spot in one of the latest photos. "
                   "point is [x, y] from 0 to 1000 with the origin at the top left. "
                   "instruction says what this leg does, e.g. \"walk to the doorway on the left\".")
    parameters = {"type": "object", "additionalProperties": False,
                  "required": ["view", "point", "instruction"],
                  "properties": {
                      "view": VIEW_SCHEMA,
                      "point": {"type": "array", "minItems": 2, "maxItems": 2,
                                "items": {"type": "integer", "minimum": 0, "maximum": 1000}},
                      "instruction": {"type": "string", "minLength": 1, "maxLength": 200}}}

    async def run(self, state, args):
        observation = state.current()
        if observation is None or args["view"] not in state.shown:
            raise ToolRefused(f"Observe the {args['view']} view at the current position first.")
        request = LegRequest(args["instruction"], state.budgets.leg_max_actions,
                             observation.observation_id, args["view"], args["point"])
        record = {"instruction": args["instruction"], "view": args["view"], "point": args["point"],
                  "observation_id": observation.observation_id, "executor": state.executor.name}
        return await self.run_executor(state, request, record)


class DelegateInstruction(Motion):
    name = "delegate"
    description = ("Let the walking policy follow a route instruction from where the robot is, until it "
                   "decides it has arrived or runs out of steps. Write plain English: an action, a "
                   "direction if needed, a landmark it can see and where to stop.")
    parameters = {"type": "object", "additionalProperties": False, "required": ["instruction"],
                  "properties": {"instruction": {"type": "string", "minLength": 1, "maxLength": 400}}}

    async def run(self, state, args):
        request = LegRequest(args["instruction"], state.budgets.leg_max_actions)
        return await self.run_executor(state, request, {
            "instruction": args["instruction"], "executor": state.executor.name})


class ReturnTo(Motion):
    name = "return_to"
    description = "Walk back along your own path to where an earlier leg ended."
    parameters = {"type": "object", "additionalProperties": False, "required": ["leg_id"],
                  "properties": {"leg_id": {"type": "integer", "minimum": 1}}}

    async def run(self, state, args):
        if not 1 <= args["leg_id"] <= len(state.legs):
            raise ToolRefused("There is no leg with that id.")
        end = state.legs[args["leg_id"] - 1]["end_event_index"]
        reply = await state.env.command({"kind": "retrace", "event_index": end,
                                         "max_actions": state.budgets.leg_max_actions})
        return await self.finish_leg(state, reply, {"retrace_of": args["leg_id"]})


class UpdatePlan(Tool):
    name = "update_plan"
    description = "Keep a short checklist of the route; send the whole list each time."
    parameters = {"type": "object", "additionalProperties": False, "required": ["steps"],
                  "properties": {"steps": {"type": "array", "minItems": 1, "maxItems": 20, "items": {
                      "type": "object", "additionalProperties": False, "required": ["text", "done"],
                      "properties": {"text": {"type": "string", "minLength": 1, "maxLength": 200},
                                     "done": {"type": "boolean"}}}}}}

    async def run(self, state, args):
        state.plan = args["steps"]
        return ToolOutput({"plan": state.plan})


class Stop(Tool):
    name = "stop"
    description = ("End the episode. status=complete when you are at the destination, "
                   "abort when you cannot go on.")
    parameters = {"type": "object", "additionalProperties": False, "required": ["status", "reason"],
                  "properties": {"status": {"type": "string", "enum": ["complete", "abort"]},
                                 "reason": {"type": "string", "minLength": 1, "maxLength": 500}}}

    async def run(self, state, args):
        await state.env.finish("policy_stop")
        state.stop, state.end_reason = dict(args), "policy_stop"
        return ToolOutput({"stopped": True}, ends_episode=True)


TOOL_SETS: dict[str, tuple[Tool, ...]] = {
    "point_v1": (Observe(), Delegate(), ReturnTo(), UpdatePlan(), Stop()),
    "vla_v1": (Observe(), DelegateInstruction(), ReturnTo(), UpdatePlan(), Stop()),
}


def tool_message(output: ToolOutput) -> str:
    return json.dumps(output.payload, ensure_ascii=False)
