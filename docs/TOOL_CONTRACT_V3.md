# Agentic tool contract v3

The agentic profile exposes nine tools. Baseline profiles and completed attempts remain unchanged.

| Tool | Arguments | Result |
| --- | --- | --- |
| observe | views: distinct front/right/back/left directions | Requested simultaneous current views, orientation and clearance; no motion |
| observe_map | none | Observed map, current node/segment, bounded related memory and history index |
| read_memory | node_id, other_node_id?, images=none/endpoint/trajectory, views?, offset=0 | Five newest matching experiences per page and optional saved evidence |
| update_memory | start_node_id, end_node_id, caption | Write confirmation and memory revision |
| update_plan | subtasks, reason | Plan revision confirmation |
| execute_subtask | task_id, instruction?, max_actions, max_distance_m, max_turn_deg, reset_context | Local execution evidence and results |
| navigate_relative | turn_deg, distance_m | Actual movement results and endpoint views; positive angles turn left |
| navigate_to_node | node_id | Return along visited ground and report actual arrival |
| terminate_episode | status=complete/abort, reason | Actual STOP result and separate model declaration |

`observe` never rotates the robot. Movement returns endpoint views automatically, but no map.
Writing memory requires an explicit successful `observe_map` after the latest movement in a preceding model turn.
A new movement invalidates this receipt and closes the previous writable segment.

`read_memory(node_id)` selects incoming and outgoing experiences. Adding `other_node_id` selects only
node_id -> other_node_id. `images=endpoint` selects one saved snapshot: the latest node observation,
or the newest matching execution endpoint on this page. `views` defaults to front and is legal only
for endpoint images. Missing directions are reported, not filled from another visit. Trajectory mode
returns at most four ordered process frames across this page. No historical read moves or re-renders the robot.
Planner evidence uses the common environment cameras independently of actor input images.

Execution instruction overrides last only for that call; update_plan owns task definitions and completion.
Actor frame budgets live in harness.history_frames (default 16) and harness.boundary_frames (default 2),
are included in RunConfig snapshots, and are not model tool arguments.

The managed planner context projects one current plan, active task, node, pending segment, map-read
validity, remaining budget and latest annotated segment. Earlier history is retrieved on demand.
The transport journal retains full state; tool messages in the model input omit repeated state payloads.
Internal map receipts remain in audit events. Lost movement responses remain fenced and are never retried.

## Validation on 2026-11-08

91 tests and 4 subtests pass; mypy and required ruff checks pass; frontend builds.
Local v3 NavigationSession was exercised against both existing owned A100 VLA workers and Habitat environments,
using direct local tools for LightNav and local MCP stdio for NavGPT VLA. Each probe made 20 calls, including
four expected recoverable rejections, two actual model predictions, node return, memory revision and image retrieval.
Each short VLA execution returned two process frames plus four endpoint views, without a map.

These are scripted integration probes, not autonomous planner traces or navigation success measurements.
Planner tokens are therefore not measured. Full raw probes are under state/probes/tools-v3-20261108/.
Both owned A100 session workers were subsequently upgraded to the frozen tools-v3 release.
Two autonomous Qwen27B runs on train episode 0 completed successfully through remote session execution.
LightNav: 12 turns, 198.1 seconds, NE 1.2633 m, 105373 input tokens.
Official VLA: 17 turns, 343.4 seconds, NE 0.3844 m, 175688 input tokens.
Each encountered one recoverable map-before-memory rejection. Neither called read_memory or requested
a selective view after the initial four-view observation. LightNav captioned 2/3 segments; official VLA
captioned 4/4 but stopped with its final plan task still in_progress. These two paired runs do not establish
a benchmark success rate or causal improvement. See evidence/tools-v3-episodes-20261108.json.
No formal 50-episode benchmark was started.
