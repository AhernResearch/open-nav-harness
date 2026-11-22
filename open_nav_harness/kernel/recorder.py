"""Episode bundle: exact wire journal, content-addressed media, leg records, results.

Every record is created once; an existing file is never replaced.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from open_nav_harness.core.artifacts import Journal


def create_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


class Recorder:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=False)
        self.root = root
        self.journal = Journal(root / "trace")
        self.store = self.journal.store
        self.legs = 0

    def image(self, png: bytes) -> str:
        return self.store.put(png, "image/png")["artifact"]

    def views(self, observation) -> dict:
        return {view: self.image(png) for view, png in observation.images.items()}

    def leg(self, record: dict) -> None:
        self.legs += 1
        create_json(self.root / "legs" / f"{self.legs:04d}.json", record)

    def write(self, name: str, value) -> None:
        create_json(self.root / name, value)
