"""Versioned, validated configuration shared by CLI, API and workers."""

from typing import Literal
from urllib.parse import urlsplit
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlannerProfile(StrictModel):
    runtime: Literal["openai_compatible", "codex", "claude"] = "openai_compatible"
    api_format: Literal["chat_completions", "responses"] = "chat_completions"
    replay_reasoning: bool = True
    token_count_mode: Literal["remote", "estimate"] = "remote"
    image_token_estimate: int = Field(default=4096, ge=256, le=32768)
    model: str = Field(min_length=1)
    base_url: str = "http://127.0.0.1:8001/v1"
    credential_env: str | None = None
    context_limit: int = Field(default=16384, ge=2048)
    output_tokens: int = Field(default=2048, ge=128)
    max_images: int = Field(default=32, ge=1, le=256)
    context_policy: Literal["full", "bounded"] = "bounded"
    omit_unique_items: bool = False
    thinking: bool = False
    effort: str | None = None
    temperature: float | None = Field(default=0.0, ge=0, le=2)
    request_timeout_s: int = Field(default=120, ge=5, le=1200)
    tokenizer_path: None = None  # Legacy null field; counting uses the serving processor.
    token_count_url: str | None = None
    max_concurrency: int = Field(default=1, ge=1, le=32)

    @field_validator("base_url", "token_count_url")
    @classmethod
    def endpoint(cls, v):
        if v is not None:
            p = urlsplit(v)
            if p.scheme not in ("http", "https") or not p.hostname or p.username or p.password:
                raise ValueError("endpoint must be HTTP(S), without inline credentials")
        return v

    @model_validator(mode="after")
    def budget(self):
        if self.output_tokens + 512 >= self.context_limit:
            raise ValueError("context must reserve input and output space")
        if self.runtime != "openai_compatible" and self.context_policy != "full":
            raise ValueError("SDK runtime requires full context; internal context is not observable")
        return self


class ServiceSpec(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,47}$")
    kind: Literal["planner", "environment", "vla", "execution"]
    url: str
    managed: bool = False
    template: str | None = None
    gpu_ids: list[int] = Field(default_factory=list)
    reserve_mib: int = Field(default=0, ge=0)
    planner: PlannerProfile | None = None
    metadata: dict = Field(default_factory=dict)
    max_concurrency: int = Field(default=1, ge=1, le=32)

    _endpoint = field_validator("url")(PlannerProfile.endpoint.__func__)

    @model_validator(mode="after")
    def complete(self):
        if self.kind == "planner" and self.planner is None:
            raise ValueError("Planner service needs a profile")
        if self.managed and not self.template:
            raise ValueError("Managed service needs an administrator-defined template")
        if bool(self.gpu_ids) != bool(self.reserve_mib):
            raise ValueError("GPU assignments and memory reservations must be provided together")
        if len(set(self.gpu_ids)) != len(self.gpu_ids) or any(g < 0 for g in self.gpu_ids):
            raise ValueError("GPU ids must be unique and nonnegative")
        if (
            self.kind == "planner"
            and self.planner
            and self.planner.base_url.rstrip("/") != self.url.rstrip("/")
        ):
            raise ValueError("Planner profile URL must match the registered endpoint")
        return self


class ExecutionBinding(StrictModel):
    environment_service: str
    execution_service: str


class ExperimentSpec(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    planner_service: str | None = None
    environment_service: str | None = None
    execution_pool: list[ExecutionBinding] = Field(default_factory=list, max_length=16)
    vla_service: str | None = None
    config: dict
    episodes: list[int] = Field(min_length=1, max_length=10000)
    purpose: Literal["development", "evaluation", "collection"] = "development"

    @model_validator(mode="after")
    def execution(self):
        if (self.config.get("dataset", {}).get("name") == "scan"
                and self.purpose != "development"):
            raise ValueError("Scan integration currently supports development attempts only")
        if bool(self.environment_service) == bool(self.execution_pool):
            raise ValueError("Choose one environment_service or an execution_pool")
        if self.execution_pool:
            if not self.vla_service:
                raise ValueError("Execution pool requires shared VLA inference")
            for key in ("environment_service", "execution_service"):
                values = [getattr(binding, key) for binding in self.execution_pool]
                if len(set(values)) != len(values):
                    raise ValueError("Execution pool must contain independent environments and workers")
        return self

    @field_validator("episodes")
    @classmethod
    def unique_episodes(cls, v):
        if len(set(v)) != len(v) or any(x < 0 for x in v):
            raise ValueError("Episode indices must be unique and nonnegative")
        return v


class ServiceTemplate(StrictModel):
    """Trusted administrator configuration, never accepted through the browser."""

    host: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    argv: list[str] = Field(min_length=1)
    working_directory: str
    environment: dict[str, str] = Field(default_factory=dict)
    startup_timeout_s: int = Field(default=600, ge=10, le=2400)


class PlatformSettings(StrictModel):
    conversations_config: str | None = None
    host: str = "127.0.0.1"
    port: int = 8878
    max_workers: int = Field(default=4, ge=1, le=16)
    ssh_config: str | None = None
    service_templates: dict[str, ServiceTemplate] = Field(default_factory=dict)
    import_roots: list[str] = Field(default_factory=list)
