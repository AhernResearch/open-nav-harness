"""Model-owned planning on the same causal observation/history store.

The strict SubtaskManager remains a reproducible baseline. This policy keeps its
frame provenance and bounded actor history, but makes review/ordering decisions
model choices rather than movement permissions.
"""

from __future__ import annotations

from .subtasks import SubtaskManager


class AgenticTasks(SubtaskManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_id = None
        self.completion_claim = None

    def snapshot(self):
        return {
            **super().snapshot(),
            "active_id": self.active_id,
            "completion_claim": self.completion_claim,
            "workflow": "model_owned",
        }

    def update_plan(self, subtasks, reason, reopen_ids=None):
        if not self.frames:
            raise ValueError("Observe the initial scene before planning")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Explain the plan revision")
        if not isinstance(subtasks, list) or not 1 <= len(subtasks) <= 128:
            raise ValueError("Plan must contain 1-128 tasks")
        old = {task["id"]: task for task in self.tasks}
        updated = []
        ids = set()
        for source in subtasks:
            ident = source.get("id")
            if type(ident) is not int or ident < 1 or ident in ids:
                raise ValueError("Task IDs must be distinct positive integers")
            ids.add(ident)
            task = {"id": ident}
            for key in ("instruction", "completion_criterion"):
                value = source.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Task needs " + key)
                task[key] = value.strip()
            task["source_quote"] = source.get("source_quote") or ""
            if not isinstance(task["source_quote"], str):
                raise ValueError("source_quote must be text when supplied")
            if task["source_quote"] and task["source_quote"] not in self.instruction:
                raise ValueError("An optional source_quote must match the original instruction")
            task["status"] = source.get("status", old.get(ident, {}).get("status", "pending"))
            if ident in (reopen_ids or []):
                task["status"] = "pending"
            if task["status"] not in ("pending", "active", "accepted"):
                raise ValueError("Task status must be pending, active or accepted")
            updated.append(task)
        self.tasks = updated
        self.revision += 1
        self.needs_replan = False
        self._event("plan", reason=reason, plan=self.snapshot())
        return self.snapshot()

    def current(self):
        if self.aborted:
            raise ValueError("Episode was aborted")
        task = next((t for t in self.tasks if t["id"] == self.active_id), None)
        if task is None or task["status"] == "accepted":
            task = next((t for t in self.tasks if t["status"] != "accepted"), None)
        if task is None:
            raise ValueError("Choose or add a remaining task before delegating to the VLA")
        return task

    def begin(self, task_id=None, instruction="", reset_context=False):
        if not self.frames:
            raise ValueError("Observe before delegating")
        if task_id is not None:
            if not any(t["id"] == task_id for t in self.tasks):
                raise ValueError("Unknown task ID")
            self.active_id = task_id
            task = next(t for t in self.tasks if t["id"] == task_id)
        else:
            task = self.current()
        self.active_id = task["id"]
        if instruction.strip():
            task["instruction"] = instruction.strip()
        if reset_context or self.anchor is None or self.anchor_task_id != task["id"]:
            self.anchor = len(self.frames) - 1
            self.anchor_task_id = task["id"]
            self.context_key = f"task-{task['id']}-frame-{self.anchor}"
        task["status"] = "active"
        self._event(
            "begin",
            task_id=task["id"],
            context_key=self.context_key,
            anchor_frame_id=self.frames[self.anchor]["id"],
        )
        return task

    def review(self, outcome, evidence_refs, summary, reason):
        if outcome not in ("accepted", "continue", "replan"):
            raise ValueError("outcome must be accepted, continue or replan")
        if not isinstance(summary, str) or not summary.strip() or len(summary) > 2400:
            raise ValueError("Write a nonempty work summary of at most 2400 characters")
        known = {f["id"] for f in self.frames} | {"map-" + f["id"].split("-", 1)[1] for f in self.frames}
        if not set(evidence_refs or []) <= known:
            raise ValueError("Unknown evidence reference")
        task = self.current()
        task["status"] = "accepted" if outcome == "accepted" else "active"
        self.needs_review = False
        self.needs_replan = outcome == "replan"
        self.summary = summary
        self.last_review = {
            "task_id": task["id"],
            "outcome": outcome,
            "evidence_refs": evidence_refs or [],
            "reason": reason,
            "camera_frame": self.frames[-1]["id"] if self.frames else None,
            "map_fresh": self.map_version == self.motion_version,
        }
        self._event("review", **self.last_review, summary=summary)
        return self.snapshot()

    def finish(self, status):
        if status not in ("complete", "abort"):
            raise ValueError("status must be complete or abort")
        self.aborted = status == "abort"
        self.completion_claim = status
        self._event("finish", status=status)


TOOL_DESCRIPTIONS = {
    "observe_forward": "Inspect the current forward RGB view at full resolution. Does not move.",
    "observe_panorama": (
        "Inspect four simultaneous camera views: ahead, right, behind, left. Does not move. "
        "Each image has direction, approximate clearance, a relative-bearing ruler and map orientation. "
        "For navigate_relative, positive degrees turn left. Read the ruler for object bearings; "
        "the whole image spans a field of view, not a single direction."
    ),
    "observe_map": (
        "Inspect the observed top-down colour map, travelled path, current pose and numbered visited nodes. "
        "Unknown cells are unobserved, not free space. Path colour runs blue (start) to red (recent); "
        "the red arrow is current pose. The map keeps a fixed world orientation but its crop can change. "
        "The result supplies scale, coordinates, bearings and node distances. Does not move or reveal the goal."
    ),
    "navigate_relative": (
        "Turn by turn_deg (positive left, negative right) and attempt distance_m metres of walking. "
        "Short obstacle detours use the simulator floor plan; long detours are refused. "
        "Walking charges roughly one action per 0.25 m; turning is included. "
        "Read actual walked_m, requested_m and blocked: sliding can reduce displacement even when not blocked. "
        "Returns endpoint views and observed node information. Clearance is a rough depth cue, not a guarantee."
    ),
    "navigate_to_node": (
        "Retrace already travelled ground to a visited node by ID or your recorded name. "
        "Node IDs remain stable. Use observe_map for node geometry and available destinations. "
        "This does not find unvisited goals."
    ),
    "annotate_node": (
        "Name and describe a visited place for later recall or retracing. By default attaches to the "
        "nearest current node; place can select a known node explicitly. caption should distinguish "
        "its visible surroundings. Optional clause associates an instruction clause. Does not move."
    ),
    "observe_node": "Retrieve the image recorded at a visited node by ID or name. Does not move.",
}
