"""Causal pruning of atomic tool-feedback turns; no hidden summarizer."""

import copy
import json


class ContextOverflow(RuntimeError):
    pass


class CounterLimit(RuntimeError):
    """The serving processor rejected an input size before returning a count."""


class ContextWindow:
    def __init__(
        self, system: str, initial: str, policy: str, limit: int, reserve: int, max_images: int = 32
    ):
        self.prefix = [{"role": "system", "content": system}, {"role": "user", "content": initial}]
        self.blocks: list[list[dict]] = []
        self.policy, self.limit, self.reserve = policy, limit, reserve
        self.state = None
        self.max_images = max_images
        self.start = 0

    def append(self, messages: list[dict], state=None):
        self.blocks.append(copy.deepcopy(messages))
        if state is not None:
            self.state = copy.deepcopy(state)

    def _projection(self):
        state: dict = self.state or {}
        if state.get("tool_contract") not in ("topology_memory_v3", "topology_memory_v4"):
            return state
        memory = state.get("memory_context", {})
        candidates = [memory.get("current_execution")] + list(reversed(memory.get("recent", [])))
        latest = state.get("latest_memory") or next((r for r in candidates if r and r.get("caption")), None)
        current = memory.get("current_execution")
        return {"plan": state.get("tasks"), "active_task_id": state.get("active_id"),
                "current_node_id": memory.get("current_node_id"),
                "pending_segment": {k: v for k, v in (current or {}).items() if k != "caption"},
                "latest_memory": latest, "map_read_valid": state.get("map_read_valid", False),
                "steps_remaining": state.get("steps_remaining"),
                "completion_claim": state.get("completion_claim")}

    async def build(self, counter, tools, journal):
        start = self.start
        stride = 1
        while True:
            messages = list(self.prefix)
            topology = bool(self.state and self.state.get("tool_contract") in ("topology_memory_v2", "topology_memory_v3", "topology_memory_v4"))
            state_message = None
            if self.state and (start or topology):
                label = "Current persisted state"
                state_message = {
                    "role": "user",
                    "content": label + ":\n" + json.dumps(self._projection(), ensure_ascii=False),
                }
                if not topology:
                    messages.append(state_message)
            map_snapshots = []
            for block in self.blocks[start:]:
                for original in block:
                    message = original
                    if topology and original.get("role") == "tool":
                        # The wire journal keeps the full response. Model input has
                        # one authoritative current state, not repeated map summaries.
                        message = copy.deepcopy(original)
                        lines: list[str] = []
                        for line in message.get("content", "").splitlines():
                            try:
                                value = json.loads(line)
                            except ValueError:
                                value = None
                            if isinstance(value, dict) and "subtask_state" in value:
                                pass
                            else:
                                if isinstance(value, dict) and "memory_context" in value:
                                    map_snapshots.append((message, len(lines), value))
                                lines.append(line)
                        message["content"] = "\n".join(lines)
                    messages.append(message)
            # Keep the newest requested map summary. Older map images/geometry remain
            # historical evidence; omitted memories can still be read from the store.
            # Only edit copied messages, preserving the complete audit history.
            for message, index, value in map_snapshots[:-1]:
                del value["memory_context"]
                value["historical_map"] = True
                lines = message["content"].splitlines()
                lines[index] = json.dumps(value, ensure_ascii=False)
                message["content"] = "\n".join(lines)
            if topology and state_message:
                # Latest state follows historical claims and observations, without
                # splitting assistant/tool pairs or duplicating the snapshot.
                messages.append(state_message)
            image_count = 0
            for message in messages:
                content = message.get("content")
                if isinstance(content, list):
                    image_count += len([item for item in content if item.get("type") == "image_url"])
            try:
                n = await counter(messages, tools) if image_count <= self.max_images else self.limit
            except CounterLimit:
                n = self.limit
            if n + self.reserve + 256 <= self.limit:
                self.start = start
                journal.emit(
                    "context_selection",
                    policy=self.policy,
                    token_count=n,
                    output_reserve=self.reserve,
                    dropped_blocks=start,
                    kept_blocks=len(self.blocks) - start,
                    task_state=self.state,
                )
                return messages
            # Never discard the latest complete tool-feedback block. If no task
            # state exists, there is no safe replacement for removed history.
            if self.policy == "full" or not self.state or start >= len(self.blocks) - 1:
                raise ContextOverflow(
                    f"Essential context ({n}) exceeds {self.limit - self.reserve - 256} input tokens"
                )
            # Advance exponentially on overflow; never reintroduce discarded turns.
            # The serving tokenizer remains authoritative for every selected input.
            start = min(len(self.blocks) - 1, start + stride)
            stride *= 2
