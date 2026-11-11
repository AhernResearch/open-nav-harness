"""Experiment settings: one YAML file drives the Environment, the VLA and the Planner.

An experiment file is self-contained: up to six sections (dataset, environment,
vla, planner, harness, output) and an optional ``name``. Omitting ``planner`` runs the VLA alone.

Machine-specific paths live in ``configs/paths.yaml`` (copied from
``configs/paths.example.yaml``): dataset roots, scene assets, curated episode
files and checkpoint directories. Experiments refer to
those entries by key.

Every key is validated, so a misspelled setting fails instead of being ignored.
This module needs only the standard library and PyYAML, so all three component
environments can load it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from .planner.tool_profiles import (
    AGENTIC_TOOLS, AGENTIC_TOOLS_WITH_DIRECT_MOVEMENT, AGENTIC_TOOLS_WITH_RECALL,
    ALL_TOOLS, CAPABILITIES, EXPECTED_TOOLS, SUBTASK_TOOLS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "configs" / "experiments"

KEYS = {
    "dataset": {"name", "split", "episodes", "episodes_file", "ground_truth_split",
                "ground_truth_file", "languages", "roles"},
    "environment": {"url", "max_steps", "forward_view_px", "panorama_view_px",
                    "turn_angle_deg", "step_size_m", "allow_sliding", "backend", "scene_manifest"},
    "vla": {"url", "checkpoint", "visual_token_budget", "frame_cache",
            "gpu_memory_fraction", "seed", "backend", "policy_url", "control_dt_s",
            "max_linear_mps", "max_angular_dps"},
    "planner": {"runtime", "model", "effort", "service_tier", "tools", "max_turns",
                "episode_timeout_s", "max_budget_usd", "profile"},
    "harness": {"interface", "review_every", "route_finding", "vla_steps_per_call",
                "mode", "execution_url", "allow_context_reset", "actor_history_policy", "stall_return", "subtask_max_distance_m", "subtask_max_turn_deg", "subtask_max_actions", "vla_instruction_style"},
    "output": {"dir", "save_frames"},
}
DOCS = "docs/USAGE.md#experiment-configs"
TOP_LEVEL = set(KEYS) | {"name", "gpus"}
EFFORTS = {"claude": ("low", "medium", "high", "xhigh", "max"),
           "codex": ("none", "minimal", "low", "medium", "high", "xhigh")}
PATH_GROUPS = {"data", "episodes", "checkpoints", "codex_bin", "python"}


class SettingsError(ValueError):
    pass


def _read(path: Path) -> dict:
    try:
        with open(path) as fh:
            if path.suffix == '.json':
                data = json.load(fh)
            else:
                import yaml
                data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        raise SettingsError("config file not found: {}".format(path)) from None
    if not isinstance(data, dict):
        raise SettingsError("{} must contain a mapping".format(path))
    return data


def _validate(data: dict, source: Path) -> None:
    unknown = set(data) - TOP_LEVEL
    if unknown:
        raise SettingsError("{}: unknown section(s) {}".format(source, sorted(unknown)))
    for section, allowed in KEYS.items():
        value = data.get(section)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise SettingsError("{}: '{}' must be a mapping".format(source, section))
        extra = set(value) - allowed
        if extra:
            raise SettingsError("{}: unknown {} key(s) {}; allowed: {} (see {})".format(
                source, section, sorted(extra), sorted(allowed), DOCS))
    if "gpus" in data:
        check_gpus(data["gpus"], "{}: gpus".format(source))


def check_gpus(gpus, what: str = "gpus") -> list[int]:
    if (not isinstance(gpus, list) or not gpus
            or any(isinstance(g, bool) or not isinstance(g, int) or g < 0 for g in gpus)):
        raise SettingsError("{} must be a non-empty list of GPU indices, e.g. [0, 1, 2, 3]"
                            .format(what))
    return gpus


def find_paths_file(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SettingsError("paths file not found: {}".format(path))
        return path
    for candidate in (Path.cwd() / "configs" / "paths.yaml", REPO_ROOT / "configs" / "paths.yaml"):
        if candidate.is_file():
            return candidate
    return None


def load_paths(explicit: str | Path | None = None) -> dict:
    path = find_paths_file(explicit)
    if path is None:
        return {}
    data = _read(path)
    unknown = set(data) - PATH_GROUPS
    if unknown:
        raise SettingsError("{}: unknown key(s) {}; allowed: {}".format(
            path, sorted(unknown), sorted(PATH_GROUPS)))
    return data


def load(config: str | Path, paths: str | Path | None = None) -> "Settings":
    source = Path(config).resolve()
    data = _read(source)
    _validate(data, source)
    return Settings(source=source, data=data, paths=load_paths(paths))


def tool_set(tools) -> tuple[str, str]:
    """Map a configured tool list to the internal (condition, capability_bundle)
    that registers exactly those tools and briefs the Planner on them."""

    known: dict = {frozenset(v): (k, "full") for k, v in EXPECTED_TOOLS.items()}
    # Agentic movement ablation: preserve task/memory semantics and remove only
    # direct metric movement. The explicit tool list is frozen in the experiment.
    for agentic in (AGENTIC_TOOLS_WITH_DIRECT_MOVEMENT, AGENTIC_TOOLS_WITH_DIRECT_MOVEMENT + ['read_memory']):
        known[frozenset(agentic)] = known[frozenset(agentic) - {"navigate_relative"}] = ("planner_agentic", "full")
    for bundle, names in CAPABILITIES.items():
        known.setdefault(frozenset(names), ("planner_vla_memory", bundle))
    every = sorted(set().union(*known))
    unknown = sorted(set(tools) - set(every))
    if unknown:
        raise SettingsError("unknown tool(s) {}; tools are: {}".format(unknown, ", ".join(every)))
    match = known.get(frozenset(tools))
    if match is None:
        options = "\n".join("  - " + ", ".join(sorted(s)) for s in
                            sorted(known, key=lambda s: (len(s), sorted(s))))
        raise SettingsError("planner.tools is not a supported combination; use one of:\n" + options)
    return match


def _url(value: str) -> tuple[str, int]:
    parts = urlsplit(value)
    if not parts.hostname or not parts.port:
        raise SettingsError("service url needs a host and port: {!r}".format(value))
    return parts.hostname, parts.port


@dataclass
class Settings:
    source: Path
    data: dict
    paths: dict

    def section(self, name: str) -> dict:
        return dict(self.data.get(name) or {})

    @property
    def name(self) -> str:
        if self.data.get("name"):
            return str(self.data["name"])
        try:
            rel = self.source.with_suffix("").relative_to(EXPERIMENTS_DIR)
        except ValueError:
            return self.source.stem
        return "_".join(rel.parts)

    @property
    def has_planner(self) -> bool:
        return bool(self.data.get("planner"))

    def lookup(self, group: str, key: str, what: str) -> str:
        """Resolve a paths.yaml entry by key; values containing a slash are paths."""
        if "/" in str(key):
            return str(key)
        table = self.paths.get(group) or {}
        if key not in table or not table[key]:
            raise SettingsError(
                "{} '{}' is not set: add it under '{}' in configs/paths.yaml "
                "(see configs/paths.example.yaml)".format(what, key, group))
        return str(table[key])

    # ── Planner / VLA-only run ──

    def run_config(self):
        from .planner.config import (DEFAULT_CODEX_MODEL, DEFAULT_MODEL, INTERFACES,
                                     RunConfig, VLA_ONLY)

        ds, env, vla = self.section("dataset"), self.section("environment"), self.section("vla")
        planner, harness, output = (self.section("planner"), self.section("harness"),
                                    self.section("output"))
        cfg = RunConfig(
            dataset=str(ds.get("name", "r2r")),
            environment_backend=str(env.get("backend", "habitat")),
            split=str(ds.get("split", "val_unseen")),
            episodes=str(ds.get("episodes", "all")),
            env_url=str(env.get("url", RunConfig.env_url)),
            vla_url=str(vla.get("url", RunConfig.vla_url)),
            vla_backend=str(vla.get("backend", "navgpt")),
            step_budget=int(env.get("max_steps", RunConfig.step_budget)),
            ablation_seed=int(vla.get("seed", RunConfig.ablation_seed)),
            run_name=self.name,
            output_root=str(output.get("dir", RunConfig.output_root)),
            live=bool(output.get("save_frames", False)),
        )
        if cfg.vla_backend not in ('navgpt', 'lightnav'):
            raise SettingsError('vla.backend must be navgpt or lightnav')
        cfg.harness_mode = str(harness.get('mode', 'global'))
        cfg.execution_url = str(harness.get('execution_url', ''))
        cfg.allow_context_reset = harness.get('allow_context_reset', True)
        if type(cfg.allow_context_reset) is not bool:
            raise SettingsError('allow_context_reset must be boolean')
        cfg.actor_history_policy = harness.get('actor_history_policy', 'episode')
        if cfg.actor_history_policy not in ('episode', 'completed_subtask'):
            raise SettingsError('actor_history_policy must be episode or completed_subtask')
        cfg.stall_return = harness.get('stall_return', 'previous')
        if cfg.stall_return not in ('previous', 'any'):
            raise SettingsError('stall_return must be previous or any')
        if cfg.actor_history_policy == 'completed_subtask' and (
            cfg.harness_mode != 'agentic' or cfg.vla_backend != 'lightnav'
        ):
            raise SettingsError('completed_subtask history requires agentic LightNav')
        if cfg.execution_url:
            _url(cfg.execution_url)
        if cfg.harness_mode not in ('global', 'subtask', 'agentic'):
            raise SettingsError('harness.mode must be global, subtask or agentic')
        cfg.vla_instruction_style = harness.get('vla_instruction_style', 'natural')
        if cfg.vla_instruction_style not in ('natural', 'r2r'):
            raise SettingsError('harness.vla_instruction_style must be natural or r2r')
        if cfg.vla_instruction_style == 'r2r' and cfg.harness_mode != 'agentic':
            raise SettingsError('r2r instruction style requires agentic mode')
        if not self.has_planner:
            if cfg.harness_mode in ('subtask', 'agentic'):
                raise SettingsError('Subtask mode requires a Planner to manage local goals')
            cfg.condition = VLA_ONLY
            if cfg.vla_backend == 'lightnav':
                cfg.model = 'LightNav-0'
            cfg.max_budget_usd = None
            return cfg

        runtime = str(planner.get("runtime", "claude"))
        if runtime not in ("claude", "codex", "openai_compatible"):
            raise SettingsError("planner.runtime must be claude or codex, not {!r}".format(runtime))
        cfg.planner_runtime = runtime
        cfg.planner_profile = planner.get("profile", {})
        if runtime == "openai_compatible":
            from open_nav_harness.core.config import PlannerProfile
            cfg.planner_profile = PlannerProfile.model_validate(cfg.planner_profile).model_dump()
        cfg.model = str(planner.get("model") or (DEFAULT_CODEX_MODEL if runtime == "codex"
                                                 else DEFAULT_MODEL))
        cfg.effort = planner.get("effort")
        if cfg.effort is not None and cfg.effort not in EFFORTS.get(runtime, ()):
            raise SettingsError("planner.effort for the {} runtime must be one of {}".format(
                runtime, ", ".join(EFFORTS[runtime])))
        cfg.max_turns = int(planner.get("max_turns", cfg.max_turns))
        cfg.episode_timeout = int(planner.get("episode_timeout_s", cfg.episode_timeout))
        budget = planner.get("max_budget_usd", 5.0 if runtime == "claude" else None)
        cfg.max_budget_usd = float(budget) if budget else None
        cfg.history_frames = int(harness.get("history_frames", 16))
        cfg.boundary_frames = int(harness.get("boundary_frames", 2))
        if not 1 <= cfg.history_frames <= 16 or not 0 <= cfg.boundary_frames <= min(4, cfg.history_frames - 1):
            raise SettingsError("Invalid actor history frame budgets")
        default_tools = {'subtask': SUBTASK_TOOLS, 'agentic': AGENTIC_TOOLS}.get(cfg.harness_mode, ALL_TOOLS)
        cfg.tools = list(planner.get("tools") or default_tools)
        cfg.condition, cfg.capability_bundle = tool_set(cfg.tools)
        for mode in ('subtask', 'agentic'):
            if (cfg.condition == 'planner_' + mode) != (cfg.harness_mode == mode):
                raise SettingsError(mode + ' mode requires exactly its task-management tool set')
        for key in ('subtask_max_distance_m', 'subtask_max_turn_deg', 'subtask_max_actions'):
            value = harness.get(key, getattr(cfg, key))
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise SettingsError(key + ' must be positive')
            if key.endswith('actions') and not isinstance(value, int):
                raise SettingsError(key + ' must be an integer')
            setattr(cfg, key, value)

        cfg.interface_variant = str(harness.get("interface", "standard"))
        if cfg.vla_backend == 'lightnav' and cfg.interface_variant != 'standard':
            raise SettingsError('LightNav supports standard harness mode; upstream seeded-reset studies are unsupported')
        if cfg.interface_variant not in INTERFACES:
            raise SettingsError("harness.interface must be one of {}".format(", ".join(INTERFACES)))
        review_every = harness.get("review_every", 0)
        if isinstance(review_every, bool) or not isinstance(review_every, int) or review_every < 0:
            raise SettingsError("harness.review_every must be a whole number of VLA steps (0: never)")
        cfg.review_every = review_every
        study = cfg.capability_bundle != "full" or cfg.review_every
        if study and cfg.interface_variant == "standard":
            raise SettingsError(
                "this tool set or harness.review_every uses the controlled-study briefing; "
                "set harness.interface to reference (or one of its variants)")
        if cfg.interface_variant != "standard" and cfg.condition != "planner_vla_memory":
            raise SettingsError("study interfaces need the eight-tool VLA tool set "
                                "or one of its reduced sets")
        if "route_finding" in harness:
            cfg.move_routes = bool(harness["route_finding"])
        if "vla_steps_per_call" in harness:
            cfg.vla_max_steps_per_call = int(harness["vla_steps_per_call"])
        return cfg

    def provider_env(self) -> dict:
        """Process environment for the Planner runtime (Codex CLI location, tier)."""
        env = {}
        if self.paths.get("codex_bin"):
            env["NAVGPT_CODEX_BIN"] = str(self.paths["codex_bin"])
        tier = self.section("planner").get("service_tier")
        if tier:
            env["NAVGPT_CODEX_SERVICE_TIER"] = str(tier)
        return env

    # ── services ──

    def _environment_config(self, env: dict, *, default_max_steps: int) -> dict:
        """Simulator config shared by every environment backend."""
        config = {"max_steps": int(env.get("max_steps", default_max_steps))}
        for key, target in (("forward_view_px", "rgb_size"), ("panorama_view_px", "pano_size"),
                            ("turn_angle_deg", "turn_angle_deg"),
                            ("step_size_m", "step_size_m"), ("allow_sliding", "allow_sliding")):
            if key in env:
                config[target] = env[key]
        vla = self.section("vla")
        if vla.get("backend") == "lightnav":
            config["policy_camera"] = {"width": 480, "height": 270, "height_m": 0.88, "hfov": 120.0}
            config["velocity_control"] = {key: vla.get(key, default) for key, default in
                (("control_dt_s", 0.1), ("max_linear_mps", 2.5), ("max_angular_dps", 300.0))}
        return config

    def environment_options(self) -> dict:
        ds, env = self.section("dataset"), self.section("environment")
        name = str(ds.get("name", "r2r"))
        backend = env.get("backend", "habitat")
        if backend not in ("habitat", "habitat_gs"):
            raise SettingsError("environment.backend must be habitat or habitat_gs")
        if backend == "habitat_gs":
            if name != "scan" or ds.get("split") != "development" or not env.get("scene_manifest"):
                raise SettingsError("Habitat-GS requires dataset.name=scan, split=development and scene_manifest")
            if set(ds) - {"name", "split", "episodes"}:
                raise SettingsError("Scan assets and episodes must come from the frozen scene manifest")
            host, port = _url(str(env.get("url", "http://127.0.0.1:19270")))
            config = self._environment_config(env, default_max_steps=100)
            return {"backend": backend, "scene_manifest": str(env["scene_manifest"]),
                    "dataset": name, "split": "development", "data_root": None, "scene_root": None,
                    "episodes_file": None, "gt_split": None, "gt_file": None,
                    "languages": None, "roles": None, "host": host, "port": port, "config": config,
                    "interface": str(self.section("harness").get("interface", "standard"))}
        if env.get("scene_manifest"):
            raise SettingsError("scene_manifest requires environment.backend=habitat_gs")
        if name not in ("r2r", "rxr"):
            raise SettingsError("dataset.name must be r2r or rxr")
        host, port = _url(str(env.get("url", "http://127.0.0.1:9200")))
        episodes_file = ds.get("episodes_file")
        if episodes_file:
            episodes_file = self.lookup("episodes", episodes_file, "episode file")
        config = self._environment_config(env, default_max_steps=500)
        listify = lambda v: ",".join(v) if isinstance(v, (list, tuple)) else v
        return {
            "dataset": name,
            "split": str(ds.get("split", "val_unseen")),
            "data_root": None if episodes_file and not ds.get("ground_truth_split") else
                         self.lookup("data", name, "dataset root"),
            "scene_root": self.lookup("data", "scenes", "scene root"),
            "episodes_file": episodes_file,
            "gt_split": ds.get("ground_truth_split"),
            "gt_file": ds.get("ground_truth_file"),
            "languages": listify(ds.get("languages", ["en-US", "en-IN"])),
            "roles": listify(ds.get("roles", ["guide"])),
            "interface": str(self.section("harness").get("interface", "standard")),
            "host": host,
            "port": port,
            "config": config,
        }

    def vla_options(self) -> dict:
        vla, output = self.section("vla"), self.section("output")
        host, port = _url(str(vla.get("url", "http://127.0.0.1:8000")))
        backend = vla.get('backend', 'navgpt')
        if backend == 'lightnav':
            if not str(vla.get('policy_url', '')).startswith(('ws://', 'wss://')):
                raise SettingsError('LightNav requires vla.policy_url (ws:// or wss://)')
            return {'backend': 'lightnav', 'host': host, 'port': port,
                    'result_path': str(Path(output.get('dir', 'outputs'))/'.vla_service'),
                    'policy_url': vla['policy_url'],
                    **{key: vla.get(key, default) for key, default in
                       (('control_dt_s', 0.1), ('max_linear_mps', 2.5), ('max_angular_dps', 300.0))}}
        if backend != 'navgpt':
            raise SettingsError('vla.backend must be navgpt or lightnav')
        options = {
            "model_path": self.lookup("checkpoints", vla.get("checkpoint", "navgpt3-8b"),
                                      "checkpoint"),
            "host": host,
            "port": port,
            "result_path": str(Path(output.get("dir", "outputs")) / ".vla_service"),
        }
        for key, target in (("visual_token_budget", "max_nav_vis_tokens"),
                            ("frame_cache", "frame_cache"),
                            ("gpu_memory_fraction", "gpu_memory_fraction")):
            if key in vla:
                options[target] = vla[key]
        return options
