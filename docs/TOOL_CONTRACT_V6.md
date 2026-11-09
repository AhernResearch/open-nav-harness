# Agentic tool contract v6 (`topology_memory_v6`)

v6 is a delta on [v5](TOOL_CONTRACT_V5.md). The tool set is unchanged.

## Planner-chosen distance cap

`execute_subtask(task_id, max_distance_m, instruction?, reset_context?)` requires `max_distance_m`.
The value must be a positive, finite number of metres; there is deliberately no lower bound in this
version. The actor's walked path length is checked after every step. When it reaches the cap, the
call returns with `subtask_return=distance_budget`, and the status echoes `max_distance_m`. There is
still no per-call turn cap.

The 2026-11-09 A/B smoke run motivated the cap. Without one, 13 of 109 delegations overshot their
closest point to the goal by more than 2 m, and 4 of those had already been within 3 m of the goal.
The removed turn cap was the harmful one: 37 of 69 handoffs ended on model-chosen turn caps.

## No-progress pause

`StallGuard` (in `navgpt/planner/tools/agentic.py`) accounts for every finished `execute_subtask`:

- **zero move:** net displacement below 0.1 m;
- **short move:** a zero move, or a stop by the actor or a blockage below 25% of the call's
  `max_distance_m`. Reaching the Planner's own cap never counts.

After 2 consecutive zero moves or 3 consecutive short moves, `execute_subtask` and
`navigate_relative` are refused before they move. They stay refused until `navigate_to_node`
reaches an allowed node, or until the episode ends. `observe`, `update_plan` and `update_memory`
remain available. Each no-progress result carries a `stall` object with `vla_moved`, the
consecutive count and, once paused, `return_options`.

`harness.stall_return` selects which nodes are allowed. It is frozen with the experiment.

| Value | Allowed nodes |
| --- | --- |
| `previous` (default) | the start node of the most recent recorded movement that changed node |
| `any` | every visited node except the current one |

If no other node exists, the guard warns but does not pause. The previous v5 smoke run showed why
the pause is needed: after two consecutive zero moves, 44 of 48 further delegations still did not
move.

## Validation

Only local checks have been run: pytest, ruff, mypy, the frontend build and Playwright. The paired
`previous` and `any` runs are pending.
