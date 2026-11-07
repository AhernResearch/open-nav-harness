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
        self.blocks.append(messages)
        if state is not None:
            self.state = copy.deepcopy(state)

    async def build(self, counter, tools, journal):
        start = self.start
        stride = 1
        while True:
            messages = list(self.prefix)
            if start and self.state:
                messages.append(
                    {
                        "role": "user",
                        "content": "Current persisted task state and previously written work summary:\n"
                        + json.dumps(self.state, ensure_ascii=False),
                    }
                )
            messages += [m for block in self.blocks[start:] for m in block]
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
