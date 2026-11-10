# Agentic tool contract v7 (`topology_memory_v7`)

v7 is a delta on [v6](TOOL_CONTRACT_V6.md). The tool set is unchanged.

## Evidence for a complete STOP

`terminate_episode(status, reason, evidence?)` takes an optional `evidence` list, and
`status=complete` requires it. Each entry is `{element, node_id, seen}`, naming one route
element of the instruction and where it was met:

- at least one entry is present;
- every `element` and `seen` is non-empty;
- every `node_id` is a visited topology node;
- one entry is at the current node, so the stopping place itself is named and described.

Evidence is deliberately not tied to plan tasks. The first release required one entry per plan
task, and the model responded by omitting whole tasks from the plan instead of evidencing them,
so the gate refused STOPs on routes it had actually walked.

`StopEvidenceGate` (in `navgpt/planner/tools/agentic.py`) checks these rules before STOP is sent.
A refusal is a repairable `precondition_failed`, and nothing moves. `status=abort` needs no
evidence. The check is structural only. It cannot judge whether the evidence is true, and it never
uses goals or reference paths.

After two refusals in one episode, the next complete STOP passes. Its result carries
`evidence_gate_bypassed: true`, so a model that cannot satisfy the format is never trapped. The
`completion_reason` plan event records the evidence, the bypass flag and the refusal count.

The 2026-11-10 val_unseen run (50 routes, v6) motivated the gate. 13 of 23 failures were
`complete` STOPs at routes the agent never came near, and their reasons restated the instruction.
The per-task form was measured on the 16 routes v6 stopped wrongly: 2 succeeded, 6 were gate
refusals, all of them missing plan tasks. Plan granularity is the model's choice, so it is not a
valid evidence key.

## Validation

Only local checks have been run: pytest, ruff, mypy, the frontend build and Playwright. The paired
GPU comparison against v6 is pending.
