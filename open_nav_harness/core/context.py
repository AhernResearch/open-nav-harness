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

    async def build(self, counter, tools, journal):
        start = self.start
        stride = 1
        budget = self.limit - self.reserve - 256
        while True:
            messages = list(self.prefix)
            if self.state and start:
                # Pruned history is replaced by the latest persisted state, after the prefix.
                messages.append({"role": "user", "content": "Current persisted state:\n"
                                 + json.dumps(self.state, ensure_ascii=False)})
            floor = self._image_floor(start)
            index = sum(self._images(m) for block in self.blocks[:start] for m in block)
            for block in self.blocks[start:]:
                for message in block:
                    count = self._images(message)
                    if count and index < floor:
                        message = self._omit_images(message, floor - index)
                    index += count
                    messages.append(message)
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
