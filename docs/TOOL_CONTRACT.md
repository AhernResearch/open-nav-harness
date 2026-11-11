# Agentic navigation tool contract

The current state schema is `topology_memory_v7`. `NavigationSession` implements
every navigation tool; direct, RPC and MCP transports share its registry.
[Tool profiles](../navgpt/planner/tool_profiles.py) define the inventories used by
configuration, preflight and session registration. Frozen attempts keep their
recorded tool inventory, source digest and evidence.

## Default tools

New agentic runs expose seven tools:

| Tool | Arguments | Result |
| --- | --- | --- |
| `observe` | `views` | Selected simultaneous front/right/back/left views; no motion |
| `observe_map` | none | Observed map, visited node IDs, pending segment and bounded memory |
| `update_plan` | `reason`, exactly one of `subtasks` or `patch` | Plan revision |
| `execute_subtask` | `task_id`, `max_distance_m`, optional `instruction`, actor-dependent `reset_context` | Execution evidence, endpoint views and map |
| `navigate_to_node` | `node_id` | Retrace over visited ground, actual arrival and map |
| `update_memory` | `start_node_id`, `end_node_id`, `caption` | Caption and memory revision |
| `terminate_episode` | `status`, `reason`, optional `evidence` | Confirmed STOP and model declaration |

Explicit historical profiles may add `navigate_relative(turn_deg, distance_m)`
or `read_memory(node_id, other_node_id?, images?, views?, offset?)`. Omitted tools
are not registered and cannot be dispatched. Baseline `global` and strict
`subtask` profiles retain their separate schemas.

## Plan and execution

Full plans contain `{id, instruction, status}` entries with distinct positive IDs
and at most one `in_progress` task. Status is `pending`, `in_progress` or
`completed`. A patch contains `{id, status}` entries for existing tasks. Plan
updates own completion; actor STOP does not complete a task or end an episode.
Implicit task activation increments the revision and records a plan event.
An execution instruction override applies only to that call.

`max_distance_m` must be positive and finite. The actor's actual walked path is
checked after each step; reaching the cap returns `subtask_return=distance_budget`.
Actor STOP, blockage, episode and runtime budgets also return control to the
Planner. Handoff is not arrival verification.

`harness.vla_instruction_style: r2r` replaces the optional instruction/reset
arguments with required `instruction` and `stop_at`. The adapter records both
fields and compiles the route followed by `Stop <stop_at>.` before invoking the
same executor. Empty fields fail before movement. The default `natural` style
retains its schema. Actor history and reset support follow the actor profile;
see [LightNav](LIGHTNAV.md).

## Memory and causality

Each public movement records one directed segment, including zero-motion and
unsuccessful executions. Nodes use stable IDs and observed 3D positions; captions
cannot alter topology. Missing movement responses leave an uncertain open edge
and retain resource ownership. A confirmed rejection before movement cancels the
tentative edge and returns repairable `precondition_failed` feedback.

The map returned by a movement or successful `observe_map` grants a caption
receipt for the pending segment. Both endpoints must match the displayed IDs.
Any later movement revokes the receipt. Repeated `update_memory` calls revise
only this pending segment. Captions contain 1–800 characters; earlier revisions
and receipts remain in the journal.

The Planner must have received the map before writing. The managed loop rejects
a batch where `update_memory` follows a map-producing call; writing before the
next movement in a later response is allowed. `read_memory` never grants a map
receipt or moves the robot. Its optional saved evidence is bounded and paginated.

The managed context projects the current plan, node, pending segment, map receipt,
remaining budget and memory once. Raw tool results, images and requests remain
in the trace. Context pruning retains a cursor, cuts token usage toward 75% of
the input budget and removes oldest images toward half the image limit. The
serving counter checks the final request; an oversized newest turn fails closed.

## No-progress handling

`StallGuard` counts finished delegations with net displacement below 0.1 m as
zero moves. A zero move, or actor STOP/blockage before 25% of the requested
distance, is a short move. Reaching the Planner's own distance cap is not short.
After two consecutive zero moves or three consecutive short moves, delegation
and direct relative movement pause until an allowed node return or episode end.
If no other node exists the guard warns without pausing.

`harness.stall_return` is frozen as `previous` (the previous distinct segment
start) or `any` (any other visited node). Feedback lists `return_options`.
Observations, plan and memory updates remain available. Explicit relative-movement
profiles also refuse a fourth identical call after three calls walking below
0.1 m each.

## STOP evidence

`status=complete` requires at least one `{element, node_id, seen}` entry.
`element` and `seen` are non-empty, every node is visited, and one entry describes
the current stopping node. Evidence describes instruction elements independently
of the model's chosen plan granularity. This is structural validation; no goals,
reference paths or evaluator labels are consulted.

After two evidence refusals, the next complete STOP may pass with
`evidence_gate_bypassed: true`. The completion event records the evidence,
bypass flag and refusal count. `status=abort` needs no evidence. The declaration
is committed only after actual STOP is confirmed. Movement is never retried.

## Validation evidence

Historical development measurements are retained under [evidence](evidence/).
They are not current validation counts or a benchmark performance claim. Run the
checks in [AGENTS.md](../AGENTS.md) after changes. GPU validation uses a separate
deployment and an explicit train/development manifest.
