# Instruction-only agentic plan experiment

The agentic plan schema is now `{id, instruction, status}`. `completion_criterion` is removed;
`assessment` was not added. IDs and statuses retain bookkeeping semantics. Execution overrides
remain local to a call and do not mutate the plan. The actor receives the local instruction and
historical memory, without a separately generated stopping condition. Strict baseline profiles
and completed traces retain their original contracts.

This changes the agentic contract to topology_memory_v4. The managed context projection supports
v2/v3 historical snapshots and the v4 current snapshot. No additional recovery rules, memory gates,
movement restrictions or evaluator signals were added for this experiment.

Validation: 94 tests and 4 subtests pass; required ruff, mypy and frontend build checks pass.
The paired development experiment uses Qwen27B + LightNav, train episodes 0 and 6907, unchanged
500-action / 100-turn / 900-second budgets. Episode 0 previously succeeded; 6907 previously timed
out after prolonged manual movement. Frozen experiment and job IDs are in
state/deployment/instruction-only-job.json. Single runs cannot establish causal performance gains.
