# Short-movement tool ablation

The planner can use `planner.tools` to select the existing agentic surface with
`navigate_relative` omitted. This is a supported explicit configuration, not a
new navigation implementation. The default tool surface is unchanged.

The interactive Harness launcher and Workbench's agentic mode explicitly freeze
the eight-tool list below for new runs. Reusing a past configuration in the
Harness does not re-enable `navigate_relative`. Generic CLI/API configurations
retain their original defaults for comparisons; already frozen experiments and
active attempts keep their recorded inventory.

The same frozen list reaches local tools and the execution worker. An omitted
tool is not registered and cannot be dispatched. Preflight verifies that exact
surface. The agentic briefing replaces the direct-movement sentence with guidance
to use `execute_subtask` for progress/repairs and `navigate_to_node` for returning
to visited places. Map/memory semantics and VLA context handling are unchanged.

Example planner tool list, with `harness.mode: agentic`:

```yaml
planner:
  tools:
    - observe
    - observe_map
    - navigate_to_node
    - terminate_episode
    - update_plan
    - execute_subtask
    - update_memory
    - read_memory
```

## Initial paired development experiment

Use the first nine routes already attempted in baseline job
`de3fce1a2f8e47f9a72e011b56ef6ed7`, in the same manifest order. Preserve the baseline,
including seven timeout outcomes and the interrupted-motion cleanup record.
The remaining 41 baseline routes are not part of this paired comparison.

Hold LightNav checkpoint, Qwen27B planner profile/server configuration, eight-lane
pool, 900-second timeout, 100 planner turns and 500 environment actions fixed.
Do not enable caching or change decoding performance in this experiment. Freeze
a separate experiment and source release; record the tool list and prompt in
its trace. No automatic movement retry is permitted.

Compare route success, endpoint error, explicit stop versus timeout, model turns,
planner HTTP time, tool counts/time, VLA prediction count, path length and repeated
subtask/return behavior. Verify zero `navigate_relative` calls in the actual wire
requests and worker dispatch journal. Lower tool counts alone are not a success:
repeated ineffective VLA calls could replace repeated short moves. This is a small,
single-pass development comparison with stochastic inference, not a definitive
benchmark or a measurement of the effect of planner serving optimizations.

The first deployment attempt (`1d825eaa10cf4d61a305194a7292f36a`)
was rejected before any model request or movement by an additional provider-loop
contract check still using the default inventory. All nine records are retained
as startup errors, excluded from navigation comparisons. Provider adapters now
resolve the same explicit frozen inventory; a model-loop regression test covers
the admission and rejection of an omitted tool. The corrected run has a separate
versioned manifest (`lightnav-no-relative-paired-v2-20261108.json`).
