"""Read-only instruction catalog for choosing an episode to run.

Files are copies of the environment's own episode files, placed in ``<state>/catalog`` and
named ``<dataset>_<split>.json.gz``; list position is the environment's episode index.
Only text fields are exposed: goals and reference paths are evaluation data.
"""

import gzip
import json
from pathlib import Path


class Catalog:
    def __init__(self, directory: Path):
        self.directory = directory
        self._cache: dict[str, tuple[int, list[dict]]] = {}

    def datasets(self) -> list[dict]:
        out = []
        for path in sorted(self.directory.glob("*_*.json.gz")):
            name = path.name.removesuffix(".json.gz")
            dataset, split = name.split("_", 1)
            scenes: dict[str, int] = {}
            for e in self._episodes(name):
                scenes[e["scene"]] = scenes.get(e["scene"], 0) + 1
            out.append({"name": name, "dataset": dataset, "split": split,
                        "scenes": [{"scene": k, "count": v} for k, v in sorted(scenes.items())]})
        return out

    def episodes(self, name: str, scene: str, query: str = "", limit: int = 50) -> list[dict]:
        words = query.lower().split()
        rows = [e for e in self._episodes(name)
                if e["scene"] == scene and all(w in e["instruction"].lower() for w in words)]
        return rows[:limit]

    def _episodes(self, name: str) -> list[dict]:
        path = self.directory / (name + ".json.gz")
        if "/" in name or not path.is_file():
            raise KeyError("Unknown catalog " + name)
        stamp = path.stat().st_mtime_ns
        cached = self._cache.get(name)
        if not cached or cached[0] != stamp:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                raw = json.load(f)["episodes"]
            rows = [{"index": i, "episode_id": e["episode_id"], "scene": Path(e["scene_id"]).stem,
                     "instruction": e["instruction"]["instruction_text"].strip()}
                    for i, e in enumerate(raw)]
            self._cache[name] = cached = (stamp, rows)
        return cached[1]
