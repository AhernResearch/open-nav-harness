"""Single result reader used by the UI, comparison and export."""

import csv
import io
import json
from pathlib import Path


def read_jsonl(path):
    events: list[dict] = []
    if not path.exists():
        return events
    lines = path.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if index == len(lines) - 1 and not line.endswith("\n"):
            break  # Only a trailing uncommitted write may be ignored.
        events.append(json.loads(line))
    return events


def attempt_detail(a):
    root = Path(a["output"])
    ep = a["episode"]
    journal = root / "episode" / "work" / f"ep{ep:04d}" / "trace" / "requests.jsonl"
    events = read_jsonl(root / "episode" / f"episode_{ep}.jsonl")
    tasks = root / "episode" / "live" / f"ep{ep:04d}" / "subtasks" / "events.jsonl"
    return {
        "subtask_events": read_jsonl(tasks),
        **a,
        "result": json.loads(a["result"]) if a.get("result") else None,
        "events": events,
        "requests": read_jsonl(journal),
        "log": (root / "worker.log").read_text()[-30000:] if (root / "worker.log").exists() else "",
    }


def metrics(a):
    r = json.loads(a["result"]) if a.get("result") else {}
    m = r.get("metrics", {})
    agent = r.get("agent", {})
    return {
        "attempt_id": a["id"],
        "episode": a["episode"],
        "attempt": a["ordinal"],
        "status": a["status"],
        "success": r.get(
            "task_success",
            int(bool(m.get("success") and m.get("stop_called") and not r.get("error"))) if m else None,
        ),
        "ne": m.get("distance_to_goal"),
        "spl": m.get("spl"),
        "ndtw": m.get("ndtw"),
        "distance_m": m.get("path_length"),
        "wall_s": r.get("wall_s"),
        "tool_calls": sum(agent.get("tool_calls", {}).values()),
        "turns": agent.get("turns"),
        "vla_calls": r.get("vla_calls"),
        "input_tokens": agent.get("usage", {}).get("input_tokens"),
        "cached_input_tokens": agent.get("usage", {}).get("cached_input_tokens"),
        "output_tokens": agent.get("usage", {}).get("output_tokens"),
        "cost_usd": agent.get("cost_usd"),
        "error": a.get("error") or r.get("error"),
        "scene_id": r.get("scene_id"),
        "episode_id": r.get("episode_id"),
        "training_status": r.get("training_status", "not_reviewed"),
    }


def csv_export(rows):
    values = [metrics(r) for r in rows]
    out = io.StringIO()
    if values:
        writer = csv.DictWriter(out, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    return out.getvalue()


def aggregate(rows):
    """Report all attempts and the first completed attempt per route separately."""
    first: dict[int, dict] = {}
    for row in rows:
        if row["status"] == "completed":
            first.setdefault(row["episode"], row)
    values = list(first.values())

    def average(key):
        numbers = [r[key] for r in values if isinstance(r.get(key), (int, float))]
        return sum(numbers) / len(numbers) if numbers else None

    return {
        "attempts": len(rows),
        "routes_completed": len(values),
        "sr": sum(bool(r["success"]) and not r.get("error") for r in values) / len(values)
        if values
        else None,
        "ne": average("ne"),
        "spl": average("spl"),
        "wall_s": average("wall_s"),
        "tool_calls": average("tool_calls"),
        "vla_calls": average("vla_calls"),
        "execution_errors": sum(bool(r.get("error")) for r in values),
        "cost_usd": sum(r["cost_usd"] for r in values)
        if values and all(r.get("cost_usd") is not None for r in values)
        else None,
        "selection_rule": "first completed attempt per episode; errors count as failures; interruptions shown separately",
    }
