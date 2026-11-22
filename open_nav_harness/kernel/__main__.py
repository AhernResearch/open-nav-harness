"""Development entry: python -m open_nav_harness.kernel SPEC --episodes ID... --output DIR."""

import argparse
import asyncio
import json
from pathlib import Path

from .episode import run_batch
from .spec import KernelSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=1)
    args = parser.parse_args()
    spec = KernelSpec.model_validate(json.loads(args.spec.read_text()))
    asyncio.run(run_batch(spec, args.episodes, args.output, args.lanes))


if __name__ == "__main__":
    main()
