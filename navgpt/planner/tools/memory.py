"""Execution-boundary places and directed episodic memory; no model or simulator imports.

Geometry/evidence are system-owned. Only a traversal's caption is model-authored.
The complete graph is persisted; context() is a bounded, deterministic projection.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path


class TopologyMemory:
    RECENT = 3
    RELATED = 5
    # UTF-8 bytes conservatively bound byte-tokenized model inputs without needing
    # a second tokenizer in the tool process. The runtime also counts full inputs.
    CONTEXT_BYTES = 6000
    PAGE_SIZE = 5
    IMAGE_LIMIT = 4
    INDEX_SIZE = 12

    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else None
        self.nodes = {}
        self.edges = []
        self.current_node_id = None
        self.active = None
        self.pending = None
        self.revision = 0

    def snapshot(self):
        return copy.deepcopy({"schema_version": 2, "revision": self.revision,
                              "current_node_id": self.current_node_id,
                              "active_edge_id": self.active, "pending_edge_id": self.pending,
                              "nodes": self.nodes, "edges": self.edges})

    def persist(self):
        self.revision += 1
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / "topology.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2) + "\n")
            temporary.replace(path)

    @staticmethod
    def position(pose):
        position = pose.get("position") if pose else None
        if not position or len(position) != 3 or not all(math.isfinite(float(v)) for v in position):
            raise ValueError("A confirmed three-dimensional pose is required")
        return list(map(float, position))

    def _node(self, pose, frame_id=None):
        position = self.position(pose)
        # Conservative place reuse, not room recognition: 25 cm in 3D. Larger
        # tolerances would merge opposite sides of doors and different floors.
        matches = [(math.dist(position, n["position"]), ident) for ident, n in self.nodes.items()
                   if math.dist(position, n["position"]) <= 0.25]
        ident = min(matches)[1] if matches else len(self.nodes)
        if ident not in self.nodes:
            self.nodes[ident] = {"id": ident, "position": position, "observation_refs": []}
        refs = self.nodes[ident]["observation_refs"]
        if frame_id and frame_id not in refs:
            refs.append(frame_id)
        self.current_node_id = ident
        return ident

    def initialize(self, pose, frame_id=None):
        if self.current_node_id is None:
            self._node(pose, frame_id)
            self.persist()

    def observe(self, pose, frame_id):
        self.initialize(pose, frame_id)
        if self.active is not None:
            refs = self.edges[self.active]["frame_refs"]
            if frame_id not in refs:
                refs.append(frame_id)
        if self.active is None and self.current_node_id is not None:
            node = self.nodes[self.current_node_id]
            if math.dist(self.position(pose), node["position"]) <= 0.25 and frame_id not in node["observation_refs"]:
                node["observation_refs"].append(frame_id)
                self.persist()

    def begin(self, pose, frame_id, tool, task=None, arguments=None):
        if self.active is not None:
            raise RuntimeError("Unresolved execution; do not dispatch another movement")
        self.initialize(pose, frame_id)
        self._previous_pending = self.pending
        self.pending = None
        ident = len(self.edges)
        self.edges.append({"id": ident, "start_node_id": self.current_node_id, "end_node_id": None,
                           "tool": tool, "task": copy.deepcopy(task), "arguments": copy.deepcopy(arguments or {}),
                           "start_pose": copy.deepcopy(pose), "end_pose": None,
                           "trajectory": [self.position(pose)], "frame_refs": [frame_id] if frame_id else [],
                           "termination_reason": None, "status": "executing", "caption": None})
        self.active = ident
        self.persist()

    def record(self, pose, frame_id=None):
        if self.active is None:
            return
        edge = self.edges[self.active]
        position = self.position(pose)
        if position != edge["trajectory"][-1]:
            edge["trajectory"].append(position)
        if frame_id and frame_id not in edge["frame_refs"]:
            edge["frame_refs"].append(frame_id)

    def finish(self, pose, frame_id, reason):
        if self.active is None:
            return
        self.record(pose, frame_id)
        edge = self.edges[self.active]
        edge.update(end_node_id=self._node(pose, frame_id), end_pose=copy.deepcopy(pose),
                    termination_reason=reason, status="recorded")
        self.pending, self.active = self.active, None
        self.persist()

    def cancel(self):
        """Validation failed before motion; restore the preceding writable segment."""
        if self.active is not None:
            self.edges.pop()
            self.active, self.pending = None, self._previous_pending
            self.persist()

    def uncertain(self, reason):
        if self.active is not None:
            self.edges[self.active].update(status="uncertain", termination_reason=reason)
            self.current_node_id = None
            self.persist()  # active remains fenced; no fabricated endpoint

    def remember(self, start, end, caption, evidence):
        if self.pending is None:
            raise ValueError("No completed execution is awaiting memory; execute a segment first")
        edge = self.edges[self.pending]
        if (start, end) != (edge["start_node_id"], edge["end_node_id"]):
            raise ValueError("Node IDs must match the current pending segment shown by observe_map")
        if not isinstance(caption, str) or not caption.strip() or len(caption) > 800:
            raise ValueError("caption must contain 1-800 characters")
        previous = edge["caption"]
        edge["caption"], edge["annotation_evidence"] = caption.strip(), copy.deepcopy(evidence)
        self.persist()
        return previous, self.brief(edge)

    @staticmethod
    def brief(edge):
        return {"sequence": edge["id"] + 1, "start_node_id": edge["start_node_id"],
                "end_node_id": edge["end_node_id"], "caption": edge["caption"],
                "task_id": (edge["task"] or {}).get("id"), "tool": edge["tool"],
                "termination_reason": edge["termination_reason"], "status": edge["status"]}

    def status(self):
        current = self.active if self.active is not None else self.pending
        return {"revision": self.revision, "current_node_id": self.current_node_id,
                "current_execution": self.brief(self.edges[current]) if current is not None else None}

    def context(self, task_id=None):
        closed = [e for e in self.edges if e["status"] == "recorded"]
        recent = closed[-self.RECENT:]
        selected = {e["id"] for e in recent}
        related = [e for e in closed if self.current_node_id in (e["start_node_id"], e["end_node_id"])
                   and e["id"] not in selected]
        related.sort(key=lambda e: ((e["task"] or {}).get("id") == task_id, e["id"]), reverse=True)
        pairs = {(e["start_node_id"], e["end_node_id"]) for e in recent}
        extra = []
        for e in related:
            pair = (e["start_node_id"], e["end_node_id"])
            if pair not in pairs:
                extra.append(e)
                pairs.add(pair)
            if len(extra) == self.RELATED:
                break
        result = {**self.status(), "recent": [self.brief(e) for e in recent],
                  "related": [self.brief(e) for e in extra]}
        # Drop whole older records, not fragments. Preserve current execution.
        while len(json.dumps(result, ensure_ascii=False).encode()) > self.CONTEXT_BYTES - 1200:
            if result["related"]:
                result["related"].pop()
            elif result["recent"]:
                result["recent"].pop(0)
            else:
                break
        shown = {e["sequence"] for e in result["recent"] + result["related"]}
        if result["current_execution"]:
            shown.add(result["current_execution"]["sequence"])
        counts = {}
        for e in reversed(closed):
            if e["id"] + 1 in shown:
                continue
            pair = (e["start_node_id"], e["end_node_id"])
            counts[pair] = counts.get(pair, 0) + 1
        index = [{"start_node_id": a, "end_node_id": b, "hidden_count": count}
                 for (a, b), count in counts.items()]
        result["history_index"] = index[:self.INDEX_SIZE]
        result["unlisted_pair_count"] = max(0, len(index) - self.INDEX_SIZE)
        result["retrieval"] = "read_memory(node_id, other_node_id?, include_images?, offset?)"
        # Reserve index space within the same budget.
        while len(json.dumps(result, ensure_ascii=False).encode()) > self.CONTEXT_BYTES:
            if result["history_index"]:
                result["history_index"].pop()
                result["unlisted_pair_count"] += 1
            elif result["related"] or result["recent"]:
                (result["related"] if result["related"] else result["recent"]).pop(0)
            else:
                break
        return result

    def read(self, node_id, other_node_id=None, offset=0):
        if node_id not in self.nodes or (other_node_id is not None and other_node_id not in self.nodes):
            raise ValueError("Use existing node IDs printed on observe_map")
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        matches = [e for e in reversed(self.edges)
                   if ((node_id in (e["start_node_id"], e["end_node_id"])) if other_node_id is None
                       else (e["start_node_id"] == node_id and e["end_node_id"] == other_node_id))]
        page = matches[offset:offset + self.PAGE_SIZE]
        return {"records": [{**self.brief(e), "task": e["task"],
                              "frame_refs": self.sample(e["frame_refs"], self.IMAGE_LIMIT),
                              "frame_count": len(e["frame_refs"]),
                              "distance_m": round(sum(math.dist(a, b) for a, b in
                                                       zip(e["trajectory"], e["trajectory"][1:])), 3)}
                             for e in page],
                "total": len(matches), "offset": offset,
                "next_offset": offset + len(page) if offset + len(page) < len(matches) else None}

    @staticmethod
    def sample(items, limit):
        if len(items) <= limit:
            return list(items)
        return [items[round(i * (len(items) - 1) / (limit - 1))] for i in range(limit)]
