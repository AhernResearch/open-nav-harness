"""Frozen run configuration for kernel episodes."""

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from open_nav_harness.core.config import PlannerProfile, StrictModel
from open_nav_harness.mapping.observed import MapSpec


class EnvironmentSpec(StrictModel):
    url: str
    dataset: str = Field(min_length=1)
    camera: str = Field(min_length=1)
    seed: int = 0

    _endpoint = field_validator("url")(PlannerProfile.endpoint.__func__)


class PointExpertSpec(StrictModel):
    kind: Literal["point_expert"] = "point_expert"


class VlaSpec(StrictModel):
    kind: Literal["vla"]
    url: str  # the actor service

    _endpoint = field_validator("url")(PlannerProfile.endpoint.__func__)


ExecutorSpec = Annotated[PointExpertSpec | VlaSpec, Field(discriminator="kind")]
# Each tool set is written for one kind of executor: it decides what a delegate call carries.
EXECUTOR_OF_TOOL_SET = {"point_v1": "point_expert", "vla_v1": "vla"}


class Budgets(StrictModel):
    max_turns: int = Field(default=40, ge=1, le=500)
    timeout_s: int = Field(default=1200, ge=10, le=7200)
    leg_max_actions: int = Field(default=200, ge=1, le=400)
    max_malformed: int = Field(default=3, ge=1, le=10)


class KernelSpec(StrictModel):
    """Everything an episode needs besides its episode id; frozen per run."""

    tool_set: Literal["point_v1", "vla_v1"] = "point_v1"
    planner: PlannerProfile
    environment: EnvironmentSpec
    executor: ExecutorSpec = Field(default_factory=PointExpertSpec)
    budgets: Budgets = Field(default_factory=Budgets)
    map: MapSpec = Field(default_factory=MapSpec)

    @model_validator(mode="after")
    def executor_fits_tool_set(self):
        if self.executor.kind != EXECUTOR_OF_TOOL_SET[self.tool_set]:
            raise ValueError(f"tool set {self.tool_set} needs executor {EXECUTOR_OF_TOOL_SET[self.tool_set]}")
        return self
