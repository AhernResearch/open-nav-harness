"""Server-sent events over an incremental feed; a replay is the same stream read from zero."""

import asyncio
import json
from collections.abc import Callable

from fastapi import Request
from fastapi.responses import StreamingResponse


def follow(request: Request, feed, status: Callable[[], dict], finished: Callable[[dict], bool]):
    """Stream ``feed.poll()`` batches with status changes until the source is finished and quiet."""

    async def stream():
        yield "retry: 3000\n\n"
        last, quiet = None, 0
        while not await request.is_disconnected():
            events = await asyncio.to_thread(feed.poll)
            if events is None:
                yield "event: reset\ndata: {}\n\n"
                return
            quiet = 0 if events else quiet + 1
            current = await asyncio.to_thread(status)
            if current != last:
                last = current
                events.append({"type": "attempt.status", **current})
            if events:
                yield f"id: {feed.cursor}\ndata: {json.dumps(events, ensure_ascii=False)}\n\n"
            elif finished(current) and quiet >= 2:
                yield f"event: end\ndata: {json.dumps(current)}\n\n"
                return
            else:
                yield ": heartbeat\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})
