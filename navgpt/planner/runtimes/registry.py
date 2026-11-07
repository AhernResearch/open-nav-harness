"""One runtime factory shared by CLI and platform workers."""

def create_runtime(cfg):
    if cfg.planner_runtime == 'codex':
        from .codex import CodexPlannerRuntime
        return CodexPlannerRuntime()
    if cfg.planner_runtime == 'claude':
        from .claude import ClaudePlannerRuntime
        return ClaudePlannerRuntime()
    if cfg.planner_runtime == 'openai_compatible':
        from nav_harness.core.config import PlannerProfile
        from nav_harness.providers.runtime import ManagedRuntime
        return ManagedRuntime(PlannerProfile.model_validate(cfg.planner_profile))
    raise ValueError('Unknown runtime: ' + cfg.planner_runtime)
