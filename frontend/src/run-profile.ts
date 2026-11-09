import type { Row } from "./api";

/** The interactive agentic profile delegates forward movement to the actor. */
export function agenticPlanner(config: Row = {}): Row {
  return {
    ...config,
    tools: [
      "observe",
      "observe_map",
      "navigate_to_node",
      "terminate_episode",
      "update_plan",
      "execute_subtask",
      "update_memory",
    ],
  };
}
