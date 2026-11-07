"""Provider-neutral session contracts; no runner or SDK dependencies."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EpisodeContext:
    briefing: str
    first_prompt: str
    model: str
    max_turns: int
    max_budget_usd: float | None
    thinking: dict
    effort: str | None
    betas: list
    mcp_env: dict[str, str]
    workdir: Path
    timeout: int


@dataclass
class SessionOutcome:
    turns: int = 0
    cost_usd: float | None = None
    usage: dict = field(default_factory=dict)
    error: str | None = None
    subtype: str | None = None
    extra: dict = field(default_factory=dict)
