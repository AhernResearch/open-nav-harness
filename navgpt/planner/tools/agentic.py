"""Model-owned planning on the same causal observation/history store.

The strict SubtaskManager remains a reproducible baseline. This policy keeps its
frame provenance and bounded actor history, but makes review/ordering decisions
model choices rather than movement permissions.
"""

from __future__ import annotations

from pydantic import ConfigDict
from typing import Literal
from typing_extensions import TypedDict

from .subtasks import SubtaskManager
from .memory import TopologyMemory


class PlanItem(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")

    id: int
    instruction: str
    completion_criterion: str
    status: Literal["pending", "in_progress", "completed"]


class AgenticTasks(SubtaskManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_id = None
        self.completion_claim = None
        self.memory = TopologyMemory(self.directory)

    def snapshot(self):
        state = super().snapshot()
        for key in ("needs_review", "needs_replan", "last_review"):
            state.pop(key)
        return {
            **state,
            "active_id": self.active_id,
            "completion_claim": self.completion_claim,
            "workflow": "model_owned",
            "tool_contract": "topology_memory_v2",
            "memory_context": self.memory.context(self.active_id),
        }

    def update_plan(self, subtasks, reason):
        if not self.frames:
            raise ValueError("Observe the initial scene before planning")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Explain the plan revision")
        if not isinstance(subtasks, list) or not 1 <= len(subtasks) <= 128:
            raise ValueError("Plan must contain 1-128 tasks")
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
            task["status"] = source.get("status", "pending")
            if task["status"] not in ("pending", "in_progress", "completed"):
                raise ValueError("Task status must be pending, in_progress or completed")
            updated.append(task)
        if sum(t["status"] == "in_progress" for t in updated) > 1:
            raise ValueError("At most one task may be in_progress")
        self.tasks = updated
        self.active_id = next((t["id"] for t in updated if t["status"] == "in_progress"), None)
        self.revision += 1
        self.needs_replan = False
        self._event("plan", reason=reason, plan=self.snapshot())
        return self.snapshot()

    def current(self):
        if self.aborted:
            raise ValueError("Episode was aborted")
        task = next((t for t in self.tasks if t["id"] == self.active_id), None)
        if task is None or task["status"] == "completed":
            task = next((t for t in self.tasks if t["status"] != "completed"), None)
        if task is None:
            raise ValueError("Choose or add a remaining task before delegating to the VLA")
        return task

    def begin(self, task_id=None, instruction="", reset_context=False):
        if not self.frames:
            raise ValueError("Observe before delegating")
        if task_id is not None:
            if not any(t["id"] == task_id for t in self.tasks):
                raise ValueError("Unknown task ID")
            task = next(t for t in self.tasks if t["id"] == task_id)
        else:
            task = self.current()
        if task["status"] == "completed":
            raise ValueError("Reopen the task with update_plan before executing it")
        self.active_id = task["id"]
        if instruction.strip():
            task["instruction"] = instruction.strip()
        if reset_context or self.anchor is None or self.anchor_task_id != task["id"]:
            self.anchor = len(self.frames) - 1
            self.anchor_task_id = task["id"]
            self.context_key = f"task-{task['id']}-frame-{self.anchor}"
        for other in self.tasks:
            if other["id"] != task["id"] and other["status"] == "in_progress":
                other["status"] = "pending"
        task["status"] = "in_progress"
        self._event(
            "begin",
            task_id=task["id"],
            context_key=self.context_key,
            anchor_frame_id=self.frames[self.anchor]["id"],
        )
        return task

    def returned(self, reason):
        self._event("return", reason=reason)

    def remember(self, start_node_id, end_node_id, caption, evidence):
        previous, entry = self.memory.remember(start_node_id, end_node_id, caption, evidence)
        self._event("memory_update", previous_caption=previous, memory=entry,
                    evidence=evidence, memory_revision=self.memory.revision)
        return entry

    def observe(self, views, pose, steps):
        frame_id = super().observe(views, pose, steps)
        if pose.get("position"):
            self.memory.observe(pose, frame_id)
        return frame_id

    def policy_instruction(self):
        task = self.current()
        latest = next((e for e in reversed(self.memory.edges) if e["caption"]), None)
        background = ("\nLatest executed-segment memory (model assessment, not an action): "
                      + f"{latest['start_node_id']} -> {latest['end_node_id']}: {latest['caption']}"
                      if latest else "")
        return (
            task["instruction"]
            + "\nStop this local operation when: "
            + task["completion_criterion"]
            + background
        )

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
        "IDs identify execution-boundary places, including revisits; captions belong to directed traversals. "
        "Returns the pending start -> end segment, recent memory and an index for read_memory. "
        "Read this map in a separate turn before update_memory. Does not move or reveal the goal."
    ),
    "navigate_relative": (
        "Turn by turn_deg (positive left, negative right) and attempt distance_m metres of walking. "
        "Short obstacle detours use the simulator floor plan; long detours are refused. "
        "Walking charges roughly one action per 0.25 m; turning is included. "
        "Read actual walked_m, requested_m and blocked: sliding can reduce displacement even when not blocked. "
        "Returns endpoint views and observed node information. Clearance is a rough depth cue, not a guarantee."
    ),
    "navigate_to_node": (
        "Retrace already travelled ground to a visited node by its map ID. "
        "Node IDs remain stable. Use observe_map for node geometry and available destinations. "
        "This does not find unvisited goals."
    ),
    "observe_node": "Retrieve the image recorded at a visited node by its map ID. Does not move.",
}
