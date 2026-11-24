"""Read-only live projection of one kernel episode bundle into item lifecycle events.

The wire journal is the spine: model turns and tool calls in exact order. Evidence records
in the same journal attach each leg's frames and poses, and each observed map, to the tool
call that produced them. The end comes from the episode's result file; after it, the goal
and dataset route from the evaluation are published for the map, and never before.
"""

from __future__ import annotations

import json
from pathlib import Path

from open_nav_harness.providers.responses import response_message

MOTION = ("delegate", "return_to")
ITEM_TYPES = {"update_plan": "plan", **{name: "motion" for name in MOTION}}
ENDINGS = ("result.json", "failed_or_uncertain.json", "failure.json")
READ_CAP = 8 << 20


class JsonlTail:
    """Byte-offset reader of an append-only JSONL file; an unterminated last line waits."""

    def __init__(self, path: Path):
        self.path, self.offset = path, 0

    def read(self) -> list[tuple[int, dict]] | None:
        """Complete new records with their end offsets, or None when the file shrank."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return []
        if size < self.offset:
            return None
        with self.path.open("rb") as f:
            f.seek(self.offset)
            chunk = f.read(min(size - self.offset, READ_CAP))
        records = []
        for line in chunk.split(b"\n")[:-1]:
            self.offset += len(line) + 1
            if line.strip():
                records.append((self.offset, json.loads(line)))
        return records


ERRORS = ("session_error", "model_transport_error", "tool_validation_error")


def common_event(r: dict) -> list[dict]:
    """Journal records shared by every harness: model turns and errors."""
    if r.get("kind") == "model_response":
        body = r.get("response") or {}
        message = (response_message(body) if body.get('object') == 'response' else
                   (body.get("choices") or [{}])[0].get("message") or {})
        return [{"type": "turn.completed", "t": r["time"], "text": message.get("content"),
                 "reasoning": message.get("reasoning") or message.get("reasoning_content"),
                 "usage": body.get("usage"), "elapsed_s": r.get("elapsed_s")}]
    return [{"type": "error", "t": r["time"], "kind": r["kind"], "error": r.get("error") or r.get("errors")}]


def _image_views(blocks: list[dict]) -> list[dict]:
    """Keep each recorded camera caption paired with its own image artifact."""
    out, caption = [], ""
    for block in blocks:
        if block.get("type") == "text":
            caption = block.get("text", "")
        url = (block.get("image_url") or {}).get("url")
        if isinstance(url, dict) and url.get("artifact"):
            out.append({"artifact": url["artifact"], "caption": caption})
            caption = ""
    return out


def plan_state(steps: list[dict]) -> dict:
    """The plan card's task shape; the first unfinished step is the one in progress."""
    current = next((i for i, step in enumerate(steps) if not step["done"]), None)
    return {"tasks": [{"instruction": step["text"],
                       "status": "completed" if step["done"] else
                       "in_progress" if i == current else "pending"}
                      for i, step in enumerate(steps)]}


class KernelFeed:
    """Rebuilds state from the start of the journal and suppresses events at or before ``cursor``."""

    def __init__(self, folder: Path, cursor: str = ""):
        self.folder = folder
        self.journal = JsonlTail(folder / "trace" / "requests.jsonl")
        self.after = int(cursor) if cursor.isdigit() else 0
        self.started = self.ended = False

    @property
    def cursor(self) -> str:
        return str(self.journal.offset)

    def poll(self) -> list[dict] | None:
        ending = next((self.folder / name for name in ENDINGS if (self.folder / name).exists()), None)
        records = self.journal.read()
        if records is None:
            return None
        out: list[dict] = []
        if not self.started and (self.folder / "session.json").exists():
            self.started = True
            session = json.loads((self.folder / "session.json").read_text())
            out.append({"type": "session.started", "instruction": session.get("instruction"),
                        "episode_id": session.get("episode_id")})
        for offset, record in records:
            events = self._journal(record)
            if offset > self.after:
                out += events
        if ending and self._drained() and not self.ended:
            self.ended = True
            out += self._reference()
            result = json.loads(ending.read_text())
            out.append({"type": "session.ended", "t": result.get("elapsed_s"),
                        "explicit_stop": result.get("stop") is not None, "status": result.get("status"),
                        "end_reason": result.get("end_reason"), "error": result.get("error"),
                        "usage": result.get("usage")})
        return out

    def _drained(self) -> bool:
        try:
            return self.journal.offset >= self.journal.path.stat().st_size
        except FileNotFoundError:
            return True

    def _journal(self, r: dict) -> list[dict]:
        kind = r.get("kind")
        if kind == "model_response" or kind in ERRORS:
            return common_event(r)
        if kind == "tool_dispatch":
            return [{"type": "item.started", "t": r["time"], "item": {
                "id": r["call_id"], "type": ITEM_TYPES.get(r["name"], "tool"), "name": r["name"],
                "arguments": r.get("arguments"), "status": "in_progress"}}]
        if kind == "tool_result":
            content = r["feedback"].get("content") or ""
            item = {"id": r["call_id"], "status": "failed" if r.get("is_error") else "completed",
                    "result": [content], "state": None,
                    "image_views": [{**view, "caption": view["caption"].rsplit(": ", 1)[-1]}
                                    for view in _image_views(r.get("images") or [])]}
            item["images"] = [view["artifact"] for view in item["image_views"]]
            payload = json.loads(content) if content.startswith("{") else {}
            if r["name"] == "update_plan" and "plan" in payload:
                item["state"] = plan_state(payload["plan"])
            if "leg" in payload:
                item["leg"] = payload["leg"]
            return [{"type": "item.completed", "t": r["time"], "item": item}]
        if kind == "leg_evidence":
            return [{"type": "item.updated", "item": {
                "id": r["call_id"], "poses": r["poses"],
                "frames": [{"policy": "trace/media/" + digest} for digest in r["frames"]]}}]
        if kind == "map_rendered":
            return [{"type": "item.updated", "item": {"id": r["call_id"], "map": {
                "artifact": r["artifact"], "size": r["size"], "places_px": {}, "overlay": "route",
                "start": r["start"],
                "transform": {"matrix": r["matrix"], "ppm": r["ppm"], "residual_px": 0, "nodes": 0}}}}]
        return []

    def _reference(self) -> list[dict]:
        try:
            reference = json.loads((self.folder / "evaluation.json").read_text()).get("reference")
        except FileNotFoundError:
            return []
        return [{"type": "session.reference", **reference}] if reference else []
