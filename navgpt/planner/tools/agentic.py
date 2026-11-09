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
    status: Literal["pending", "in_progress", "completed"]


class AgenticTasks(SubtaskManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_id = None
        self.completion_claim = None
        self.completed_boundary = None
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
            "tool_contract": "topology_memory_v4",
            "memory_context": self.memory.context(self.active_id),
            "latest_memory": next((self.memory.brief(e) for e in reversed(self.memory.edges) if e["caption"]), None),
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
            value = source.get("instruction")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Task needs instruction")
            task["instruction"] = value.strip()
            task["status"] = source.get("status", "pending")
            if task["status"] not in ("pending", "in_progress", "completed"):
                raise ValueError("Task status must be pending, in_progress or completed")
            updated.append(task)
        if sum(t["status"] == "in_progress" for t in updated) > 1:
            raise ValueError("At most one task may be in_progress")
        # Only completion of the last executed task creates a boundary. Keep the
        # completion observation even if the planner later retraces before delegating.
        completed = {t['id'] for t in updated if t['status'] == 'completed'}
        previous = {t['id'] for t in self.tasks if t['status'] == 'completed'}
        if self.completed_boundary and self.completed_boundary['task_id'] not in completed:
            self.completed_boundary = None
        if self.anchor_task_id in completed - previous:
            self.completed_boundary = {
                'task_id': self.anchor_task_id, 'frame_index': len(self.frames) - 1,
                'frame_id': self.frames[-1]['id'],
            }
            self._event('task_completion_boundary', **self.completed_boundary)
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
        self.execution_instruction = instruction.strip() or task["instruction"]
        reset_reason = ("explicit" if reset_context else "initial" if self.anchor is None
                        else "task_change" if self.anchor_task_id != task["id"] else "continue")
        if reset_reason != "continue":
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
            reset_requested=reset_context,
            reset_reason=reset_reason,
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

    def policy_instruction(self, include_memory=True):
        task = self.current()
        latest = next((e for e in reversed(self.memory.edges) if e["caption"]), None)
        background = (f"\nHistorical memory (not current state; current node {self.memory.current_node_id}): "
                      + f"{latest['start_node_id']} -> {latest['end_node_id']}: {latest['caption']}"
                      if latest and include_memory else "")
        return (
            getattr(self, "execution_instruction", task["instruction"])
            + background
        )

    def finish(self, status):
        if status not in ("complete", "abort"):
            raise ValueError("status must be complete or abort")
        self.aborted = status == "abort"
        self.completion_claim = status
        self._event("finish", status=status)


TOOL_DESCRIPTIONS = {
    "observe_map": (
        "View the observed map, visited node IDs and current pose. Path: blue to red over time; "
        "red arrow: current pose. World orientation is fixed; crop may change. Unknown cells "
        "are unobserved. Returns the pending start -> end segment, recent memory and history "
        "index. Does not move or reveal the goal."
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
}
