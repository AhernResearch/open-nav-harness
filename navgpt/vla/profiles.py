"""Actor input contracts, independent of planner context management."""
from dataclasses import dataclass


@dataclass(frozen=True)
class InputProfile:
    history: str
    include_memory: bool
    force_stop_at_budget: bool = False

    @property
    def manual_reset(self):
        return self.history == "bounded"


INPUT_PROFILES = {
    "navgpt": InputProfile(history="bounded", include_memory=True),
    "lightnav": InputProfile(history="episode", include_memory=False, force_stop_at_budget=True),
}


def input_profile(backend):
    return INPUT_PROFILES[backend]
