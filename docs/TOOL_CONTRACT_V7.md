# Agentic tool contract v7 (`topology_memory_v7`)

v7 is a delta on [v6](TOOL_CONTRACT_V6.md). The tool set is unchanged.

## Evidence for a complete STOP

`terminate_episode(status, reason, evidence?)` takes an optional `evidence` list, and
`status=complete` requires it. Each entry is `{task_id, node_id, seen}`:

- every plan task has at least one entry, and no entry names a task outside the plan;
- every `node_id` is a visited topology node;
- every `seen` is non-empty;
- the last plan task is satisfied at the current node, where STOP executes.

`StopEvidenceGate` (in `navgpt/planner/tools/agentic.py`) checks these rules before STOP is sent.
A refusal is a repairable `precondition_failed`, and nothing moves. `status=abort` needs no
evidence. The check is structural only. It cannot judge whether the evidence is true, and it never
uses goals or reference paths.

After two refusals in one episode, the next complete STOP passes. Its result carries
`evidence_gate_bypassed: true`, so a model that cannot satisfy the format is never trapped. The
`completion_reason` plan event records the evidence, the bypass flag and the refusal count.

The 2026-11-10 val_unseen run (50 routes, v6) motivated the gate. 13 of 23 failures were
`complete` STOPs at routes the agent never came near, and their reasons restated the instruction.

## Validation

Only local checks have been run: pytest, ruff, mypy, the frontend build and Playwright. The paired
GPU comparison against v6 is pending.
