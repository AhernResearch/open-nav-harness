"""Read-only live projection of one attempt's trace into item lifecycle events.

The wire journal is the spine: model turns and tool calls in exact order. Motion progress
comes from the subtask event log, segmented by ``execution_memory`` (one per executed
movement), so segment k belongs to the k-th movement call that executed. Node positions
come from ``topology.json`` and only serve to place map overlays. The executed instruction
is the first line of the episode log; nothing after it (metrics, trajectories) is read.
No other file is opened.
"""

import cmath
import json
import struct
from pathlib import Path

MOTION = {"execute_subtask", "navigate_relative", "navigate_to_node"}
ITEM_TYPES = {"update_plan": "plan", "update_memory": "memory_write", "read_memory": "memory_read",
              "observe_map": "map", **{name: "motion" for name in MOTION}}
MAP_MPP = 0.02  # Environment default map resolution: magnified maps use k / MAP_MPP px per metre.
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


def fit_map(places_px: dict, nodes: dict) -> dict | None:
    """Similarity transform from world (x, z) to map pixels, fitted on labelled memory nodes.

    The renderer draws ``(world - origin) * ppm`` with ``ppm = k / MAP_MPP``; the fitted scale
    is snapped to that lattice. Returns None when fewer than two separated nodes are labelled
    or when three or more nodes disagree by more than 1.5 px.
    """
    pairs = [(complex(nodes[k][0], nodes[k][2]), complex(*places_px[k]))
             for k in places_px if k in nodes]
    if len(pairs) < 2:
        return None
    world, pixels = zip(*pairs)
    wm, pm = sum(world) / len(world), sum(pixels) / len(pixels)
    spread = sum(abs(w - wm) ** 2 for w in world)
    if spread < 0.25 ** 2 / 2:
        return None
    a = sum((p - pm) * (w - wm).conjugate() for w, p in pairs) / spread
    step = 1 / MAP_MPP
    a = cmath.rect(max(1, round(abs(a) / step)) * step, cmath.phase(a))
    b = pm - a * wm
    residual = max(abs(a * w + b - p) for w, p in pairs)
    if len(pairs) >= 3 and residual > 1.5:
        return None
    # px = m[0]*x + m[1]*z + m[2]; py = m[3]*x + m[4]*z + m[5]
    return {"matrix": [a.real, -a.imag, b.real, a.imag, a.real, b.imag],
            "ppm": abs(a), "residual_px": round(residual, 3), "nodes": len(pairs)}


def png_size(path: Path) -> list[int] | None:
    with path.open("rb") as f:
        head = f.read(24)
    return list(struct.unpack(">II", head[16:24])) if head[:8] == b"\x89PNG\r\n\x1a\n" else None


def episode_meta(output: Path, episode: int) -> dict | None:
    """The executed instruction and scene: only the first, complete line of the episode log."""
    try:
        with (output / "episode" / f"episode_{episode}.jsonl").open("rb") as f:
            line = f.readline()
    except FileNotFoundError:
        return None
    if not line.endswith(b"\n"):
        return None
    meta = json.loads(line)
    if meta.get("kind") != "episode_meta":
        return None
    return {k: meta.get(k) for k in ("instruction", "scene_id", "episode_id", "index")}


def _lines(content: str) -> tuple[list[str], dict | None]:
    """Tool feedback text lines, with the trailing task-state record split off."""
    lines, state = [], None
    for line in content.split("\n"):
        if line.startswith('{"subtask_state"'):
            state = json.loads(line)["subtask_state"]
        else:
            lines.append(line)
    return lines, state


class Feed:
    """Incremental projection of one attempt. Rebuilds state from the start of each file and
    suppresses everything at or before ``cursor`` (``"<journal_offset>.<events_offset>"``)."""

    def __init__(self, output: Path, episode: int, cursor: str = ""):
        live = output / "episode" / "live" / f"ep{episode:04d}"
        self.output, self.episode = output, episode
        self.meta: dict | None = None
        self.media = output / "episode" / "work" / f"ep{episode:04d}" / "trace" / "media"
        self.topology = live / "subtasks" / "topology.json"
        self.journal = JsonlTail(self.media.parent / "requests.jsonl")
        self.events = JsonlTail(live / "subtasks" / "events.jsonl")
        try:
            self.after = [int(x) for x in cursor.split(".")] if cursor else [0, 0]
            assert len(self.after) == 2
        except (ValueError, AssertionError):
            self.after = [0, 0]
        self.motions: list[str] = []  # executing or executed movement call IDs, in order
        self.moved: set[str] = set()
        self.open: str | None = None
        self.segment = 0
        self.pending: list[tuple[int, dict]] = []
        self.maps: list[dict] = []  # map items waiting for node positions

    @property
    def cursor(self) -> str:
        return f"{self.journal.offset}.{self.events.offset}"

    def poll(self) -> list[dict] | None:
        """New events in delivery order, or None when a source was truncated."""
        out: list[tuple[int, int, dict]] = []
        journal = self.journal.read()
        if journal is None:
            return None
        for offset, record in journal:
            out += [(0, offset, e) for e in self._journal(record)]
        events = self.events.read()
        if events is None:
            return None
        self.pending += events
        out += [(1, offset, e) for offset, e in self._motion()]
        fresh = [e for src, offset, e in out if offset > self.after[src]]
        if self.meta is None and (meta := episode_meta(self.output, self.episode)):
            self.meta = meta
            fresh.insert(0, {"type": "session.started", **meta})
        # Placement is an idempotent upsert that may resolve after its map was delivered.
        return fresh + self._place_maps()

    def _journal(self, r: dict) -> list[dict]:
        kind = r.get("kind")
        if kind == "model_response":
            body = r.get("response") or {}
            message = (body.get("choices") or [{}])[0].get("message") or {}
            return [{"type": "turn.completed", "t": r["time"], "text": message.get("content"),
                     "reasoning": message.get("reasoning") or message.get("reasoning_content"),
                     "usage": body.get("usage"), "elapsed_s": r.get("elapsed_s")}]
        if kind == "tool_dispatch":
            name = r["name"]
            if name in MOTION:
                self.motions.append(r["call_id"])
            self.open = r["call_id"]
            return [{"type": "item.started", "t": r["time"], "item": {
                "id": r["call_id"], "type": ITEM_TYPES.get(name, "tool"), "name": name,
                "arguments": r.get("arguments"), "status": "in_progress"}}]
        if kind == "tool_result":
            call = r["call_id"]
            self.open = None
            if r.get("is_error") and call in self.motions and call not in self.moved:
                self.motions.remove(call)  # rejected before moving: it owns no motion segment
            lines, state = _lines(r["feedback"].get("content") or "")
            item = {"id": call, "status": "failed" if r.get("is_error") else "completed",
                    "result": lines, "state": state,
                    "images": [i["image_url"]["url"]["artifact"] for i in r.get("images") or []
                               if isinstance((i.get("image_url") or {}).get("url"), dict)]}
            if r["name"] == "observe_map" and not r.get("is_error") and lines and item["images"]:
                head = json.loads(lines[0])
                item["map"] = {"artifact": item["images"][0], "places_px": head.get("places_px") or {},
                               "size": png_size(self.media / item["images"][0]), "transform": None}
                self.maps.append(item)
            return [{"type": "item.completed", "t": r["time"], "item": item}]
        if kind in ("session_error", "model_transport_error", "tool_validation_error"):
            return [{"type": "error", "t": r["time"], "kind": kind,
                     "error": r.get("error") or r.get("errors")}]
        if kind == "session_end":
            return [{"type": "session.ended", "t": r["time"], "explicit_stop": r.get("explicit_stop"),
                     "error": r.get("error"), "usage": r.get("usage")}]
        return []

    def _motion(self) -> list[tuple[int, dict]]:
        """Attach motion-log events to their movement call once that call is known."""
        out = []
        while self.pending:
            offset, e = self.pending[0]
            kind = e.get("kind")
            update = None
            if kind == "policy_execution_result":
                pose = (e.get("feedback") or {}).get("pose")
                if pose:
                    update = {"pose": pose, "step": e["feedback"].get("step_count")}
            elif kind == "observation" and e.get("context_key"):
                update = {"frame": {name: self._relative(v.get("path"))
                                    for name, v in (e.get("images") or {}).items()}, "pose": e.get("pose")}
            elif kind == "execution_memory":
                update = {"execution": e.get("current_execution"), "node": e.get("current_node_id")}
            if update is not None:
                if self.segment >= len(self.motions):
                    break  # its movement call is not in the journal yet
                call = self.motions[self.segment]
                self.moved.add(call)
                out.append((offset, {"type": "item.updated", "item": {"id": call, **update}}))
            if kind == "execution_memory":
                self.segment += 1
            self.pending.pop(0)
        return out

    def _place_maps(self) -> list[dict]:
        if not self.maps:
            return []
        try:
            nodes = {k: v["position"] for k, v in json.loads(self.topology.read_text())["nodes"].items()}
        except (OSError, ValueError, KeyError):
            return []
        out, waiting = [], []
        for item in self.maps:
            m = item["map"]
            if all(k in nodes for k in m["places_px"]):
                m["transform"] = fit_map(m["places_px"], nodes)
                m["nodes"] = {k: nodes[k] for k in m["places_px"]}
                out.append({"type": "item.updated", "item": {"id": item["id"], "map": m}})
            else:
                waiting.append(item)
        self.maps = waiting
        return out

    def _relative(self, path: str | None) -> str | None:
        try:
            return str(Path(path).relative_to(self.output)) if path else None
        except ValueError:
            return None
