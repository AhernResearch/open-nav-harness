"""Causal pruning of atomic tool-feedback turns; no hidden summarizer."""

import copy
import json


class ContextOverflow(RuntimeError):
    pass


class ImageBudget(ContextOverflow):
    """The newest tool-feedback block alone carries more images than the input allows."""


class CounterLimit(RuntimeError):
    """The serving processor rejected an input size before returning a count."""


class ContextWindow:
    # Once history must be removed, remove enough that the next turns append without
    # pruning again. Every prune or newly omitted image rewrites the prompt after the
    # system prefix, so trimming the minimum each turn would defeat prefix caching.
    TOKEN_LOW_WATER = 0.75
    IMAGE_LOW_WATER = 0.5

    def __init__(
        self, system: str, initial: str, policy: str, limit: int, reserve: int, max_images: int = 32
    ):
        self.prefix = [{"role": "system", "content": system}, {"role": "user", "content": initial}]
        self.blocks: list[list[dict]] = []
        self.policy, self.limit, self.reserve = policy, limit, reserve
        self.state = None
        self.max_images = max_images
        self.start = 0
        # Images before this global index (counted over all blocks) are omitted, oldest
        # first. It only grows, so an omitted image never reappears.
        self.image_floor = 0

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
        budget = self.limit - self.reserve - 256
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
            floor = self._image_floor(start)
            index = sum(self._images(m) for block in self.blocks[:start] for m in block)
            for block in self.blocks[start:]:
                for original in block:
                    message = original
                    count = self._images(original)
                    if count and index < floor:
                        message = self._omit_images(original, floor - index)
                    index += count
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
            try:
                n = await counter(messages, tools)
            except CounterLimit:
                n = self.limit
            # The current selection only needs to fit; a new prune must leave headroom.
            # The newest block alone is accepted whenever it fits at all.
            target = budget if start == self.start or start >= len(self.blocks) - 1 else budget * self.TOKEN_LOW_WATER
            if n <= target:
                self.start, self.image_floor = start, floor
                journal.emit(
                    "context_selection",
                    policy=self.policy,
                    token_count=n,
                    output_reserve=self.reserve,
                    dropped_blocks=start,
                    kept_blocks=len(self.blocks) - start,
                    omitted_images=floor,
                    task_state=self.state,
                )
                return messages
            # Never discard the latest complete tool-feedback block. If no task
            # state exists, there is no safe replacement for removed history.
            if self.policy == "full" or not self.state or start >= len(self.blocks) - 1:
                raise ContextOverflow(
                    f"Essential context ({n}) exceeds {budget} input tokens"
                )
            # Advance exponentially on overflow; never reintroduce discarded turns.
            # The serving tokenizer remains authoritative for every selected input.
            start = min(len(self.blocks) - 1, start + stride)
            stride *= 2

    @staticmethod
    def _images(message: dict) -> int:
        content = message.get("content")
        return sum(item.get("type") == "image_url" for item in content) if isinstance(content, list) else 0

    def _image_floor(self, start: int) -> int:
        """Image floor keeping at most ``max_images`` images in ``blocks[start:]``.

        Older images are omitted before older turns are dropped, because their text
        (calls, results, image labels) is cheap and still explains what happened.
        Exceeding the budget omits down to the low-water mark, never into the newest block.
        """
        before = sum(self._images(m) for block in self.blocks[:start] for m in block)
        kept = sum(self._images(m) for block in self.blocks[start:] for m in block)
        floor = max(self.image_floor, before)
        if before + kept - floor <= self.max_images:
            return floor
        newest = sum(self._images(m) for m in self.blocks[-1])
        if self.policy == "full" or newest > self.max_images:
            raise ImageBudget(f"{kept} images exceed the {self.max_images}-image input budget"
                              + ("" if self.policy == "full" else f"; the newest turn alone has {newest}"))
        return before + kept - max(newest, int(self.max_images * self.IMAGE_LOW_WATER))

    @staticmethod
    def _omit_images(message: dict, count: int) -> dict:
        """A copy whose first ``count`` images become a text marker; labels remain."""
        content = []
        for item in message["content"]:
            if item.get("type") == "image_url" and count > 0:
                count -= 1
                item = {"type": "text", "text": "[image omitted from context to fit the image budget]"}
            content.append(item)
        return {**message, "content": content}
