"""Run or inspect one navigation configuration against separately managed services.

Service ownership, concurrency, recovery and replay belong to open-nav-harness.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .settings import SettingsError, load


def _shard(value: str) -> tuple[int, int]:
    try:
        index, count = (int(x) for x in value.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError("expected I/N, e.g. 0/8") from None
    if count < 1 or not 0 <= index < count:
        raise argparse.ArgumentTypeError("shard index must be in [0, N)")
    return index, count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="navgpt", description="Navigation runtime and preflight checks")
    sub = parser.add_subparsers(dest="command", required=True)

    def experiment(p):
        p.add_argument("--config", required=True, help="experiment YAML file")
        p.add_argument("--paths", help="machine paths file (default: configs/paths.yaml)")

    run = sub.add_parser("run", help="run (or resume) an experiment")
    experiment(run)
    run.add_argument("--episodes", help="index spec overriding dataset.episodes, e.g. 0-9")
    run.add_argument("--shard", type=_shard, help="run shard I of N (interleaved episodes)")
    run.add_argument("--env-url", help="override environment.url (one per shard)")
    run.add_argument("--vla-url", help="override vla.url (one per shard)")

    experiment(sub.add_parser("preflight", help="check services, tools and model; spend nothing"))
    experiment(sub.add_parser("check-data", help="check episodes, dense GT and scene files"))
    experiment(sub.add_parser("check-checkpoint", help="check the configured NavGPT3 checkpoint"))

    return parser


def _report(check, *args, **kwargs) -> int:
    """Run a read-only check and print its JSON summary or its first problem."""
    try:
        result = check(*args, **kwargs)
    except (ValueError, OSError) as exc:
        print("check failed: {}".format(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = load(args.config, args.paths)

        if args.command == "check-data":
            from .environment.datasets import check_dataset
            options = settings.environment_options()
            if options.get("backend") == "habitat_gs":
                from .environment.scan_assets import check_scan_package
                return _report(check_scan_package, options["scene_manifest"])
            return _report(check_dataset,
                options["dataset"], options["data_root"], options["scene_root"],
                split=options["split"], episodes_file=options["episodes_file"],
                gt_split=options["gt_split"], gt_file=options["gt_file"],
                languages=options["languages"], roles=options["roles"])

        if args.command == "check-checkpoint":
            if settings.section('vla').get('backend') == 'lightnav':
                raise SettingsError('LightNav weights are owned by the external lightnav-serve process; '
                                    'use preflight with the Planner config to check the bridge')
            from .vla.checkpoint import validate_checkpoint
            return _report(validate_checkpoint, Path(settings.vla_options()["model_path"]))

        cfg = settings.run_config()
        if args.command == "run":
            if args.episodes:
                cfg.episodes = args.episodes
            if args.shard:
                cfg.shard = args.shard
            if args.env_url:
                cfg.env_url = args.env_url
            if args.vla_url:
                cfg.vla_url = args.vla_url

        if not settings.has_planner and args.command == "preflight":
            raise SettingsError("preflight checks the Planner; this experiment has none")
        os.environ.update(settings.provider_env())
        from .planner.runner import PreflightError, preflight, run_eval
        try:
            if not settings.has_planner:
                from .planner.vla_only import run as run_vla_only
                run_vla_only(cfg)
            elif args.command == "preflight":
                print(json.dumps(preflight(cfg), indent=2))
            else:
                from .planner.runtimes.registry import create_runtime
                asyncio.run(run_eval(create_runtime(cfg), cfg))
        except PreflightError as exc:
            print("\n[NavGPT Planner] PREFLIGHT FAILED\n  {}\n".format(exc), file=sys.stderr)
            return 2
        return 0
    except SettingsError as exc:
        print("navgpt: {}".format(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nnavgpt: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
