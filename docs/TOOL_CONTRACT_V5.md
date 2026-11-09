# Agentic tool contract v5 (`topology_memory_v5`)

v5 supersedes `TOOL_CONTRACT_V3.md` and the v4 instruction-only plan for the tool surface. Its aim is fewer
bookkeeping turns. Baseline profiles and completed attempts remain unchanged.

The default agentic surface has eight tools, or seven when `navigate_relative` is ablated:

| Tool | Arguments | Result |
| --- | --- | --- |
| observe | views: distinct front/right/back/left directions | Requested simultaneous current views; no motion |
| observe_map | none | Observed map, node IDs, pending segment, bounded memory context; grants the caption credential |
| update_memory | start_node_id, end_node_id, caption | Write confirmation and memory revision |
| update_plan | reason plus exactly one of subtasks (full plan) or patch ([{id, status}]) | Plan revision confirmation |
| execute_subtask | task_id, instruction?, reset_context? | Execution evidence, endpoint views, then the current map |
| navigate_relative | turn_deg, distance_m | Movement result and endpoint views, then the current map |
| navigate_to_node | node_id | Return result, then the current map |
| terminate_episode | status=complete/abort, reason | Actual STOP result; the declaration is written only after STOP is confirmed |

`read_memory` is still implemented. It is registered only when an experiment's `planner.tools` lists it
explicitly (`AGENTIC_TOOLS_WITH_RECALL`). In the v4 traces it was called once in 16 episodes.

## Changes from v4

- **The map comes with every movement.** Each recorded movement ends with the current map and its
  `memory_context`. That map grants the caption credential, exactly as `observe_map` does. Any later
  movement revokes it. If the map cannot be rendered, the result says so and no credential is granted.
- **A caption may come before a map in the same response, never after one.** The runtime rejects a batch
  in which `update_memory` follows `observe_map` or a movement, because the model has not yet seen that
  map. A typical segment is therefore two turns:
  `execute_subtask` → `[update_memory, update_plan(patch), execute_subtask]`.
  Before this change it took four turns, and 17% of `update_memory` calls failed their precondition.
- **Plan patches.** `update_plan(patch=[{id, status}], reason)` changes only the statuses of existing tasks.
  Promoting a task demotes the previous `in_progress` task. Patches go through the same validation and
  completion-boundary logic as a full plan.
- **Implicit activation is recorded.** When `execute_subtask` makes a task `in_progress`, the revision is
  incremented and a `plan` event is emitted. A plan revision the model did not write never happens silently.
- **No-progress guard.** After three consecutive identical `navigate_relative` calls that each walked less
  than 0.1 m, with no other movement in between, the fourth identical call is refused before it moves.
  The refusal is repairable.
- **Movement rejection is distinct from an uncertain outcome.** If the environment rejects a movement
  before its first step (a 4xx or an `error` output), the result is a repairable `precondition_failed`.
  The memory edge is cancelled and the map credential stays valid. A transport loss or a 5xx is still
  treated as uncertain and fenced.
- **Prompt.** The model writes at most one short sentence before tool calls.

## Context (unchanged contract, new pruning)

The managed context puts the projected current state last. Pruning, once required, cuts down to 75% of
the input budget. Image omission, once required, cuts down to half of `max_images`, oldest images first,
and keeps their text. Both keep the request prefix stable for prefix caching. If the newest turn alone
exceeds the image budget, the run fails with `ImageBudget` rather than a token overflow.

## Validation

Only local checks have been run: pytest, ruff, mypy, the frontend build and Playwright. No GPU run or A/B
comparison has been done yet. Every change above alters the release digest, so it requires an owned-worker
release before any queued experiment can use it.
