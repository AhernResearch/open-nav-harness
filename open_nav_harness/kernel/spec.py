"""Frozen run configuration for kernel episodes."""

from typing import Literal

from pydantic import Field, field_validator

from open_nav_harness.core.config import PlannerProfile, StrictModel


class EnvironmentSpec(StrictModel):
    url: str
    dataset: str = Field(min_length=1)
    camera: str = Field(min_length=1)
    seed: int = 0

    _endpoint = field_validator("url")(PlannerProfile.endpoint.__func__)


class ExecutorSpec(StrictModel):
    kind: Literal["point_expert"] = "point_expert"


class Budgets(StrictModel):
    max_turns: int = Field(default=40, ge=1, le=500)
    timeout_s: int = Field(default=1200, ge=10, le=7200)
    leg_max_actions: int = Field(default=200, ge=1, le=400)
    max_malformed: int = Field(default=3, ge=1, le=10)


class KernelSpec(StrictModel):
    """Everything an episode needs besides its episode id; frozen per run."""

    tool_set: Literal["point_v1"] = "point_v1"
    planner: PlannerProfile
    environment: EnvironmentSpec
    executor: ExecutorSpec = Field(default_factory=ExecutorSpec)
    budgets: Budgets = Field(default_factory=Budgets)
