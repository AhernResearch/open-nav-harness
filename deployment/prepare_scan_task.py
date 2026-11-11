"""Create a new, immutable development task from a verified scan package."""

import argparse
import json
from pathlib import Path
import shutil
import uuid

from navgpt.environment.scan_assets import load_scan_package, sha256, write_json


def prepare(source, destination, instruction, episode_id):
    path, _, _, episodes = load_scan_package(source)
    destination = Path(destination).resolve()
    if not instruction.strip():
        raise ValueError("Instruction must not be empty")
    # copytree refuses an existing destination, including completed task packages.
    shutil.copytree(path.parent, destination)
    episode = {**episodes[0], "episode_id": episode_id,
               "instruction": {"instruction_text": instruction}}
    episode_path = destination / "navigation/episodes.json"
    episode_path.unlink()
    write_json(episode_path, {"purpose": "development", "episodes": [episode]})
    receipt_path = destination / "navigation/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["parent_navigation_receipt_sha256"] = sha256(path.parent / "navigation/receipt.json")
    receipt["owner"] = uuid.uuid4().hex
    receipt["assets"]["episodes"]["sha256"] = sha256(episode_path)
    receipt_path.unlink()
    write_json(receipt_path, receipt)
    load_scan_package(destination / "scene.json")
    return destination / "scene.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--episode-id", required=True)
    args = parser.parse_args()
    print(prepare(args.source, args.destination, args.instruction, args.episode_id))
