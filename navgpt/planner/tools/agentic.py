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


class PlanPatch(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")

    id: int
    status: Literal["pending", "in_progress", "completed"]


class StopEvidence(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")

    element: str
    node_id: int
    seen: str


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
            "tool_contract": "topology_memory_v7",
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

    def patch_plan(self, patch, reason):
        """Status-only revision; the full-plan path validates and records it."""
        changes = {}
        for item in patch:
            if item["id"] in changes:
                raise ValueError("Each task ID may appear once in a patch")
            changes[item["id"]] = item["status"]
        unknown = set(changes) - {t["id"] for t in self.tasks}
        if not changes or unknown:
            raise ValueError("Patch must name existing task IDs" + (f"; unknown: {sorted(unknown)}" if unknown else ""))
        # Promoting a task demotes the previous in_progress one, as a full plan would require.
        promoted = any(s == "in_progress" for s in changes.values())
        return self.update_plan([{**t, "status": changes.get(t["id"], "pending" if promoted and t["status"] == "in_progress"
                                                         else t["status"])} for t in self.tasks], reason)

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
        # Executing a task makes it the in_progress one. This is a plan revision the model
        # did not write, so it is recorded like one rather than changed silently.
        if task["status"] != "in_progress":
            for other in self.tasks:
                if other["status"] == "in_progress":
                    other["status"] = "pending"
            task["status"] = "in_progress"
            self.revision += 1
            self._event("plan", reason=f"execute_subtask started task {task['id']}", plan=self.snapshot())
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


class StallGuard:
    """Hands control back after repeated delegations that do not move the robot.

    A zero move has net displacement below ZERO_M. A short move ends by actor stop or
    blockage below SHORT_FRACTION of the Planner's own distance cap; reaching that cap
    is the Planner's choice and never counts. After ZERO_LIMIT consecutive zero moves or
    SHORT_LIMIT consecutive short moves, further delegation is refused until the robot
    returns to an allowed node or the episode ends. ``policy`` selects which nodes are
    allowed: ``previous`` (the last node it moved away from) or ``any`` other visited node.
    """

    ZERO_M = 0.1
    SHORT_FRACTION = 0.25
    ZERO_LIMIT = 2
    SHORT_LIMIT = 3
    POLICIES = ("previous", "any")

    def __init__(self, policy="previous"):
        if policy not in self.POLICIES:
            raise ValueError("stall_return must be previous or any")
        self.policy = policy
        self.zero = self.short = 0
        self.allowed = None  # node IDs that release the lock; None means unlocked

    def record(self, net_m, cap_m, reason, memory):
        """Account one delegation; return model-facing feedback, or None after progress."""
        zero = net_m < self.ZERO_M
        short = zero or (reason in ("vla_stop", "blocked") and net_m < self.SHORT_FRACTION * cap_m)
        self.zero = self.zero + 1 if zero else 0
        self.short = self.short + 1 if short else 0
        if not short:
            return None
        if self.zero >= self.ZERO_LIMIT or self.short >= self.SHORT_LIMIT:
            self.allowed = self.return_nodes(memory) or None
        feedback = {"vla_moved": not zero, "net_displacement_m": round(net_m, 2),
                    "consecutive_no_progress": self.short,
                    "note": "The actor made little or no progress. Rephrasing the same move from "
                            "the same pose usually stops again."}
        if self.allowed:
            feedback.update(locked=True, return_options=self.allowed,
                            note="Delegation is paused after repeated no-progress attempts. Call "
                                 "navigate_to_node with one of return_options, or terminate_episode.")
        return feedback

    def return_nodes(self, memory):
        current = memory.current_node_id
        if self.policy == "any":
            return sorted(n for n in memory.nodes if n != current)
        moved = (e for e in reversed(memory.edges)
                 if e["status"] == "recorded" and e["end_node_id"] != e["start_node_id"]
                 and e["start_node_id"] is not None and e["start_node_id"] != current)
        edge = next(moved, None)
        return [edge["start_node_id"]] if edge else []

    def check_delegation(self):
        if self.allowed:
            raise ValueError("Delegation is paused after repeated no-progress attempts: call "
                             f"navigate_to_node with one of {self.allowed}, or terminate_episode")

    def check_return(self, node_id):
        if self.allowed and node_id not in self.allowed:
            raise ValueError(f"After repeated no-progress attempts, return to one of {self.allowed}")

    def released(self):
        self.allowed, self.zero, self.short = None, 0, 0


class StopEvidenceGate:
    """Requires route evidence before a ``complete`` STOP.

    Each entry names a route element from the instruction, the visited node where it was satisfied
    and what was seen there. One entry must be at the current node, so the stop location itself is
    claimed and named. Nothing is checked against the plan: plan granularity is the model's choice
    and tying evidence to it produced omissions. The check is structural only and never uses goals
    or reference paths. After MAX_REJECTIONS refusals the next complete STOP passes and is reported
    as bypassed. ``abort`` is always allowed.
    """

    MAX_REJECTIONS = 2

    def __init__(self):
        self.rejections = 0

    def problems(self, evidence, memory):
        entries = evidence or []
        found = ["give at least one route element and where you saw it"] if not entries else []
        found += [f"describe the {e['element'] or 'unnamed'} element you saw" for e in entries if not e["element"].strip()]
        found += [f"describe what you saw for {e['element'] or 'an element'}" for e in entries if not e["seen"].strip()]
        found += [f"node {e['node_id']} was never visited" for e in entries if e["node_id"] not in memory.nodes]
        if entries and all(e["node_id"] != memory.current_node_id for e in entries):
            found.append(f"say what you see here, at node {memory.current_node_id}, where you are stopping")
        return found

    def check(self, status, evidence, memory):
        """Raise ValueError to refuse STOP; return True when a deficient STOP is let through."""
        if status != "complete":
            return False
        found = self.problems(evidence, memory)
        if not found:
            return False
        if self.rejections >= self.MAX_REJECTIONS:
            return True
        self.rejections += 1
        raise ValueError("STOP refused, nothing moved: " + "; ".join(found) + ". Name each route element from "
                         "the instruction you satisfied, the visited node where it happened and what you saw "
                         "there, including the place where you are stopping now; or terminate with status=abort.")
