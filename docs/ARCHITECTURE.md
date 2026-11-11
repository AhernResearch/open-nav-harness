# Architecture

## Ownership

| Layer | Responsibility | Must not own |
|---|---|---|
| `frontend/src/pages` | Forms, status and evidence presentation | Navigation or scheduling decisions |
| `frontend/src/harness` | Live, read-only run view (`/harness`): chat thread, plan, memory, map; may pause/cancel a job | Queue or experiment rules |
| `open_nav_harness/platform/harness.py` | Incremental projection of one attempt's trace into item events | Any write to run files; evaluation data |
| `open_nav_harness/platform` | Auth, immutable experiments, queue, leases, service lifecycle, exports | Model-specific prompting logic |
| `open_nav_harness/core` | Validated contracts, artifacts, context budget, recovery evidence | GPU model loading |
| `open_nav_harness/providers` | Wire protocol and managed model/tool loop | Environment state or navigation tools |
| `navgpt/planner/tools/session.py` | Episode-local tool state and shared navigation capability | Global mutable episode state |
| `navgpt/planner/tools/subtasks.py` | Plan/review lifecycle and causal actor frame selection | Hidden goal or success labels |
| `navgpt/environment` | Simulator execution, observations and evaluator metrics | Planner/provider selection |
| `navgpt/vla` | Actor inference and history replacement | Experiment orchestration |

```mermaid
flowchart LR
    UI[React console] --> API[Holo API]
    API --> DB[(SQLite WAL)]
    Worker[Single Holo scheduler] --> DB
    Worker --> Episode[Isolated episode process]
    Episode --> Planner[Planner adapter]
    Planner --> Loop[Managed API loop or SDK]
    Loop --> Tools[Shared navigation tools]
    Tools --> Env[A100 Habitat]
    Tools --> VLA[A100 actor]
    Loop --> Trace[Requests + content-addressed images]
    Episode --> Results[Evaluation artifacts]
    API --> Replay[Read-only replay and export]
```

## Extension boundaries

`navgpt.planner.contracts` is provider-neutral. The runtime factory is the single
selection point for CLI and platform. API-compatible local models share
`CompatibleClient` and `ManagedRuntime`; model IDs never branch the navigation logic.
A different API protocol should add a client adapter and corresponding validation.
Existing Codex/Claude runtimes retain their SDK-driven tool transport and local login.
They are marked `sdk_observable_events_only`; SDK messages cannot be reconstructed
as exact model input. Managed API sessions record `exact_wire_requests`.

The MCP file is an optional transport entry point over the same tool registry.
Navigation rules exist on the session instance. Two instances do not share mutable
plan, observation, map, budget or VLA state. Per-episode subprocess isolation additionally
keeps SDK state and crash handling separate.

## Context and causality

The managed context consists of the original instruction/system briefing and complete
assistant/tool-feedback blocks. Tool images follow their text results in a user image
message, with explicit originating call IDs. This mapping is in the recorded wire input.
Before each request the serving processor counts the actual messages and tool schemas,
including image tokens. Output and safety reserves are subtracted from the context cap.
The image cap is checked before tokenization.

`full` refuses overflow. `bounded` removes oldest whole feedback blocks, retains the
latest block and inserts the already-written task state/work summary. Without persisted
task state, pruning fails explicitly. It never synthesizes a summary from future events.
The full audit journal survives pruning. The context-selection record and model-request
record identify precisely what the model was given.

Actor input contracts are selected by `navgpt/vla/profiles.py`, independently of
Planner context pruning. NavGPT VLA uses bounded replacement at task boundaries.
LightNav retains its native continuous episode buffer and SlowFast sampler; task
changes only revise instruction text. `policy_context` events preserve source frame
IDs. The client appends unseen frames, and re-inference at a stationary endpoint
reuses the last observation instead of inventing another timestep.

Simulator metrics, goals and reference paths are evaluation artifacts. They are never
added to Planner or actor requests. The UI labels evaluator trajectory plots separately.
Map observations exposed by NavGPT-3 retain their original simulated geometry support;
this is a harness capability, not a learned model capability.

## Durable execution

SQLite uses WAL, foreign keys and `BEGIN IMMEDIATE` for claims and lifecycle fences.
An attempt owns its environment, actor and one Planner slot. One active attempt per job
preserves route order; independent jobs may run concurrently when resources permit.
Completed episodes are not automatically repeated. Failure recovery receives a new
attempt ordinal and directory. Configuration, launch-template snapshots and execution-source fingerprints are frozen.
The execution fingerprint covers navigation, providers, context and the episode executor;
control-plane/UI changes do not invalidate navigation configurations.

The worker lock prevents multiple schedulers. Each episode writes a process receipt
before execution. After a restart the worker verifies PID, Linux start ticks and the
exact input path, attaches to a surviving process or consumes its committed result.
An unexplained death retains leases. Even a result file cannot release a resource if
its trace contains an unacknowledged/errored movement request.

Managed service operations fence scheduler claims in the same database transaction.
An owned service is claimable only after readiness. Startup reservations include other
starting services; insufficient memory never evicts existing processes. Remote units
carry a generated `OPEN_NAV_HARNESS_OWNER` token and use systemd process-group cleanup.
Only trusted configuration files contain executable argv; the browser cannot submit
shell commands. Register each physical stateful endpoint once; aliases are unsupported.

## Scope of v0.1

Single administrator, Linux control host, one scheduler, SQLite and systemd user services
are deliberate choices. There is no multi-tenant RBAC, cluster autoscaler, arbitrary
plugin execution, automatic trace training admission, or model-quality guarantee.
External endpoints can participate in experiments, but the platform cannot safely
reconcile their uncertain effects through process ownership it does not hold.

## Lightweight execution path

`ToolRegistry` owns schemas, strict argument validation and text/image results without
importing MCP. `ManagedRuntime` calls it directly; Codex/Claude use the thin MCP adapter.
`open_tools` selects direct Python or the leased actor worker's RPC transport. There is
one `NavigationSession`, not separate implementations for each transport or model.

```mermaid
flowchart LR
    Local[Local Planner API] --> Loop[ManagedRuntime]
    Codex[Codex SDK / local login] --> MCP[Thin MCP adapter]
    Loop --> Tools[Shared tool registry]
    MCP --> Tools
    Tools --> RPC[One tool invocation over SSH-forwarded HTTP]
    RPC --> Session[A100 NavigationSession]
    Session --> Actor[In-process actor]
    Session --> Habitat[Loopback Habitat service]
    Actor --> Light[LightNav pinned inference library + vLLM]
    Actor --> Native[NavGPT custom action head + PyTorch]
```

The actor worker owns weights for its process lifetime, and one episode at a time.
A session ID fences other clients and legacy actor mutations. Duplicate request IDs
are rejected. Dispatch/result receipts are flushed to disk before responding. Unknown
movement outcomes retain ownership; no transport retries movement. Artifacts synchronize
at tool boundaries with an acknowledgment cursor; local paths are rewritten for replay.
Codex credentials remain on Holo. Worker ports bind to A100 loopback and use direct-LAN
SSH forwarding. Habitat remains a separate interpreter because its dependencies differ.

An actor's executable source fingerprint and configured environment endpoint must match
preflight. Deploy immutable release directories; changing code requires new experiments
and a worker restart after current episodes finish. A failed session requires explicit
owned-service recovery, not an implicit model reset.

## Model-owned workflow

`harness.mode: agentic` keeps plan/state/history provenance but removes strict review
permissions. The model selects the task and remaining instruction; execution and
history budgets belong to runtime configuration. NavGPT VLA retains a bounded
window (1–16 groups with 0–4 boundary groups). LightNav uses its native episode
history. The current frame is always retained. It can repair or revise without first calling a review tool, retrieve node
images, and stop independently of plan checkboxes. Initial observation/decomposition
and evidence-based handoff review are prompting guidance. `subtask` preserves the
previous strict workflow for comparisons; `global` preserves full-route delegation.

### Topology memory

The [current tool contract](TOOL_CONTRACT.md) defines the seven default agentic
tools, plan schema, execution caps, map receipts and STOP evidence. Explicit
historical inventories remain supported. All profiles use the same session and
memory implementation; completed traces retain their recorded contract.

One `TopologyMemory` owns the graph in `live/subtasks/topology.json`, independent of the
legacy dense geometric graph. Nodes contain stable IDs, actual 3D positions and frame
references; a new boundary reuses the closest node within 0.25 m in 3D, or creates a node.
This is conservative geometric matching, not visual room recognition. Observe calls do
not create new nodes after initialization. Every public movement (`execute_subtask`,
`navigate_relative`, `navigate_to_node`) records one directed execution edge, including
zero-displacement/self-loop and unsuccessful attempts. Parallel and reverse traversals
remain distinct. The dense actual trajectory is retained for execution and retracing,
not exposed as additional memory nodes. Return uses a 3D match on the recorded path;
it never infers a shortcut through unknown ground.

System-owned edge fields are internal ID/order, actual start/end nodes and poses, tool,
arguments, task snapshot, trajectory, ordered frame references, execution status and stop
reason. The **only model-authored memory field is caption**. Failure/exploration/return
interpretations belong in that caption, not a system-invented semantic classification.
A missing motion response leaves an uncertain, unclosed edge and retains ownership;
it cannot invent an endpoint or retry. Validation before movement cancels the tentative
edge. Backend STOP and budget expiry never establish task completion.

`update_memory(start_node_id, end_node_id, caption)` has exactly three required fields.
IDs are the numbers on the map, not task IDs or internal edge IDs. It writes only the
pending execution; repeat calls revise that execution, never earlier trips with the
same endpoints. Beginning a new execution closes the prior write window. Captions are
1–800 characters; storage has no old 4000-character global cap. Full previous captions
and map-read evidence remain in the event journal. No parallel node-caption store exists.

A successful `observe_map` or movement-returned map grants write access. The receipt records image
SHA-256, shown IDs, frame, motion epoch and pending execution. Both endpoints must be
shown and match the pending edge. Every later motion or VLA call invalidates it;
`read_memory` and failed map reads do not grant access. The managed loop rejects
a batch where a memory write follows a map-producing call before the model sees its result.
MCP enforces completed tool-read-before-write but cannot independently certify SDK
model-request boundaries.

The same map renderer draws the memory node layer for agentic mode (with a capability
check preventing old dense IDs from being mistaken for memory IDs). The latest actual
execution path is cyan; numbered nodes and the current heading remain visible. VLA
keyframes and endpoint views are still returned automatically. `observe_map` also returns
current execution, selected captions and an omitted-history index through task state.

Context selection is deterministic: current execution always retained; latest three
recorded executions; up to five additional current-node-adjacent records, prioritizing
same-task association then recency, with directed-pair deduplication. No hidden semantic
classifier claims to recognize all failures. Whole older records are removed to fit a
6000 UTF-8-byte memory budget (a conservative byte-tokenizer bound); up to twelve hidden
pair counts advertise retrieval. Complete history stays persisted. The serving tokenizer
still checks the entire request. Uncaptioned executions are explicit, not omitted as if
they never happened. The managed loop replaces repeated task/map summary snapshots with
one current persisted state, while preserving original tool responses in the trace.
SDK-managed conversation compaction is external; do not claim identical prompt replacement
there. The most recent caption is bounded background for the VLA, not a new instruction.

`read_memory(node_id, other_node_id=None, images="none", views=None, offset=0)` selects incident
history, or only the specified directed pair. It returns five executions newest-first,
explicit pagination and system facts plus model captions. Optional images are capped at
four sampled chronological frames across the page; references/counts disclose omissions.
The memory read does not move, write, or grant map-read permission.

Malformed tool arguments reject the whole unexecuted batch and return repair feedback.
The next model call can correct them. A repeated call ID, uncertain movement, truncated
response or missing explicit STOP remains an execution error. This distinction prevents
a formatting mistake from being confused with failed navigation or unsafe retry.

Bounded Planner context retains a pruning cursor and uses exponential advancement on
overflow, followed by an authoritative tokenizer check. Discarded turns are not repeatedly
resent to the counter. The complete trace remains available for supervision and replay.

Actor supervision records preserve native horizons: each `policy_context`,
`policy_prediction`, `policy_execution_dispatch`, and `policy_execution_result` shares
an episode-local `prediction_id`. Predictions contain the original waypoints and
controller advice; execution results contain observed pose, step count and navigation
status. An unacknowledged dispatch has no fabricated result. LightNav's 10 points are
not silently converted to the native VLA's 8 points or relabelled with a time interval.
Dataset resampling and training admission are separate, reviewed operations.

## Harness view and Workbench

The console has two views in one app. `/workbench` is the management console (services,
experiments, queue, comparison, legacy replay). `/harness` is a live, read-only view of one
attempt: a session list, a chat-like thread with a docked plan, a map and the robot's view.
`/` opens the Harness; `/?attempt=ID` redirects to `/harness?attempt=ID`.

`GET /api/attempts/{id}/stream` is the only live source. It tails exactly three files:
`trace/requests.jsonl` (model turns and tool calls, in exact order), `subtasks/events.jsonl`
(movement progress) and `subtasks/topology.json` (node positions for the map). Events use item
lifecycle types (`item.started/updated/completed`, `turn.completed`, `attempt.status`); a
finished attempt replays through the same stream. The cursor is `<journal_offset>.<events_offset>`
and is sent as the SSE `id`. Movement progress is assigned by construction: each executed movement
call owns one segment that ends with `execution_memory`; rejected calls own none. Evaluation
data (goals, reference paths, metrics) is never read. Only the first line of `episode_<n>.jsonl`
(the executed instruction) is used from the episode log.

The map image is the `observe_map` result the model saw. Live overlays are drawn in that image's
pixels using a similarity fitted from the labelled memory nodes (`fit_map`); with fewer than two
nodes, or inconsistent labels, no overlay is drawn. The instruction picker reads text-only copies
of environment episode files from `<state>/catalog/<dataset>_<split>.json.gz`, whose list position
is the environment's episode index. Starting a run freezes a one-episode `development` experiment
and enqueues it through the normal queue and leases. Custom instruction text is not supported yet:
it would change the executed instruction and therefore the implementation fingerprint.

The composer reads `GET /api/harness/services` every ten seconds. Read-only probes check
endpoint health and verify that an API Planner advertises its registered model ID. SDK model
access is checked at preflight. Managed services must also be `ready`; probing never adopts
processes or changes service states. The model menu retains unavailable entries, and an explicit
selection is preserved if its endpoint goes offline. Sending requires an available Planner,
environment and VLA. Model choice is the registered service ID: normal experiment freezing
copies that service's complete Planner profile, including endpoint and context budgets.
For shared VLA inference the composer limits environment choices to registered execution
bindings for that VLA, selects a matching worker, and freezes a one-lane `execution_pool`.
Legacy owned actors continue using `environment_service`; pool admission and leases remain
owned by the control plane.

Pending Harness runs keep their job ID in `?job=ID`. The read-only
`GET /api/harness/jobs/{id}` reports the latest attempt and distinguishes occupied
resources, services awaiting startup, and retained leases from interrupted attempts.
The view also displays paused, terminal and connection-error states instead of an
unconditional waiting message. Polling never releases leases or resubmits a job;
recovery uses the existing ownership-checked stop and reconciliation workflow.
