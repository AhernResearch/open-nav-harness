"""Tool inventories shared by settings, preflight and navigation transports.

Keep historical profiles explicit so frozen configurations retain their surface.
This module has no runner, provider or simulator dependencies.
"""

ALL_TOOLS = [
    "observe_forward", "observe_panorama", "observe_map", "navigate_by_instruction",
    "navigate_relative", "navigate_to_node", "annotate_node", "terminate_episode",
]
SUBTASK_TOOLS = [tool for tool in ALL_TOOLS if tool != "navigate_by_instruction"] + [
    "update_plan", "execute_subtask", "review_subtask",
]
AGENTIC_TOOLS = [
    "observe", "observe_map", "navigate_to_node", "terminate_episode",
    "update_plan", "execute_subtask", "update_memory",
]
AGENTIC_TOOLS_WITH_DIRECT_MOVEMENT = AGENTIC_TOOLS + ["navigate_relative"]
AGENTIC_TOOLS_WITH_RECALL = AGENTIC_TOOLS + ["read_memory"]

CAPABILITIES = {
    "core": ("navigate_by_instruction", "observe_forward", "observe_panorama", "terminate_episode"),
    "spatial_memory": ("navigate_by_instruction", "observe_forward", "observe_panorama", "terminate_episode",
                       "observe_map", "navigate_to_node", "annotate_node"),
    "full": tuple(ALL_TOOLS),
}

EXPECTED_TOOLS = {
    "planner_agentic": AGENTIC_TOOLS,
    "planner_subtask": SUBTASK_TOOLS,
    "planner_basic": ["navigate_primitive", "observe_forward"],
    "planner_motion": ["navigate_primitive", "navigate_relative", "observe_forward",
                       "observe_map", "observe_panorama", "terminate_episode"],
    "planner_memory": ["annotate_node", "navigate_primitive", "navigate_relative",
                       "navigate_to_node", "observe_forward", "observe_map", "observe_node",
                       "observe_panorama", "terminate_episode"],
    "planner_vla_memory": ALL_TOOLS,
}


def expected_tools(condition, bundle="full"):
    if condition == "planner_vla_memory":
        return list(CAPABILITIES[bundle])
    tools = EXPECTED_TOOLS.get(condition)
    return list(tools) if tools is not None else None
