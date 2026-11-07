"""Determine whether interrupted tool calls may still own remote motion."""

MOTION_TOOLS = frozenset(
    {
        "execute_subtask",
        "navigate_by_instruction",
        "navigate_relative",
        "navigate_to_node",
        "terminate_episode",
        "navigate_primitive",
    }
)


def resource_state(events: list[dict], *, wire: bool, worker_info: dict | None = None) -> str:
    # RPC ownership can survive even a failed observation or lost session-open reply.
    if worker_info is not None and (
        worker_info.get("protocol_version") != 1 or worker_info.get("session_id") or worker_info.get("failed")
    ):
        return "unknown"
    pending = set()
    for e in events:
        if e["kind"] == ("tool_dispatch" if wire else "tool_use"):
            name = str(e.get("name", "")).rsplit("__", 1)[-1]
            if name in MOTION_TOOLS:
                pending.add(e.get("call_id") if wire else e.get("id"))
        elif e["kind"] == "tool_result":
            key = e.get("call_id") if wire else e.get("tool_use_id")
            if not e.get("is_error"):
                pending.discard(key)
    return "unknown" if pending else "idle"
