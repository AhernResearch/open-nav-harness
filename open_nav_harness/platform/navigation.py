"""A conversation's sim-engine session, exposed as conversation tools.

The kernel's point_v1 tool set runs unchanged over one bound episode. Evidence (per-leg
frames, poses, the observed map) is recorded into the conversation folder exactly as an
episode bundle is, and never returned to the model. A lost motion reply ends the binding:
the session is released and the conversation must bind again, because motion is never
replayed.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from pydantic import Field, field_validator

from open_nav_harness.core.config import PlannerProfile, StrictModel
from open_nav_harness.kernel.environment import MotionUncertain, SimEngineEnvironment, SimulatorError
from open_nav_harness.kernel.evidence import Evidence, EvidenceError
from open_nav_harness.kernel.executors import EXECUTORS
from open_nav_harness.kernel.recorder import Recorder
from open_nav_harness.kernel.spec import Budgets, EnvironmentSpec
from open_nav_harness.kernel.tools import TOOL_SETS, EpisodeState, ToolRefused, tool_message
from open_nav_harness.mapping.observed import MapSpec

from navgpt.planner.tools.registry import ImageContent, TextContent, ToolDefinition, ToolResult

from .database import ident


class NavigationTarget(StrictModel):
    """One sim-engine endpoint a conversation may bind to; the episode is chosen per binding."""

    url: str
    dataset: str = Field(min_length=1)
    camera: str = Field(min_length=1)
    seed: int = 0
    budgets: Budgets = Field(default_factory=Budgets)
    map: MapSpec = Field(default_factory=MapSpec)

    _endpoint = field_validator("url")(PlannerProfile.endpoint.__func__)


class NavigationEnded(RuntimeError):
    """The bound session ended; further navigation needs a new binding."""


def refusal(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(text=json.dumps({"error": text}))], isError=True)


class KernelNavigation:
    def __init__(self, target: NavigationTarget, folder: Path, transport=None):
        self.target, self.folder, self.transport = target, folder, transport
        self.tools = {tool.name: tool for tool in TOOL_SETS["point_v1"]}
        self.env: SimEngineEnvironment | None = None
        self.state: EpisodeState | None = None
        self.ended: str | None = None

    async def start(self, episode: str) -> dict:
        """Open the episode and record its receipt before the first tool call."""
        env = SimEngineEnvironment(EnvironmentSpec(url=self.target.url, dataset=self.target.dataset,
                                                   camera=self.target.camera, seed=self.target.seed),
                                   transport=self.transport)
        try:
            opened = await env.open(episode)
        except BaseException:
            await env.close()
            raise
        recorder = Recorder(self.folder)
        recorder.write("session.json", {"url": self.target.url, "session_id": env.session_id,
                                        "episode_id": episode, "instruction": opened["instruction"]})
        recorder.write("spec.json", {"navigation": self.target.model_dump(mode="json"), "episode": episode})
        evidence = Evidence(env, recorder, self.target.map)
        await evidence.start()
        self.env = env
        self.state = EpisodeState(opened["instruction"], env, EXECUTORS["point_expert"],
                                  self.target.budgets, recorder, evidence)
        return opened

    def definitions(self) -> list[ToolDefinition]:
        return [ToolDefinition(name=tool.name, description=tool.description, inputSchema=tool.parameters)
                for tool in self.tools.values()]

    async def call(self, name: str, arguments: dict) -> ToolResult:
        if self.state is None:
            return refusal("No environment is bound; bind one before navigating.")
        if self.ended:
            return refusal(f"The navigation session ended ({self.ended}); bind a new environment.")
        tool = self.tools.get(name)
        if tool is None:
            raise ValueError("Unknown navigation tool " + name)
        # Evidence records are keyed by the call that produced them, as the kernel loop does.
        self.state.evidence.call = ident()
        try:
            output = await tool.run(self.state, arguments)
        except (ToolRefused, SimulatorError) as error:
            return refusal(str(error))
        except (MotionUncertain, EvidenceError) as error:
            await self.fail(type(error).__name__)
            raise
        if output.ends_episode:
            self.ended = self.state.end_reason or "episode_over"
        content: list = [TextContent(text=tool_message(output))]
        for label, png in output.images:
            # The short text before an image is what the runtime carries to the model as its label.
            content += [TextContent(text=label),
                        ImageContent(data=base64.b64encode(png).decode(), mimeType="image/png")]
        return ToolResult(content=content, isError=False)

    async def fail(self, reason: str):
        """Record an uncertain or incomplete episode and release its session; never replay it."""
        self.ended = reason
        if self.state is not None:
            self.state.recorder.write("failed_or_uncertain.json", {
                "reason": reason, "session_id": self.env.session_id if self.env else None,
                "legs": len(self.state.legs), "event_index": self.env.event_index if self.env else None})
        await self.close()

    async def close(self):
        if self.env is not None:
            env, self.env = self.env, None
            await env.close()
