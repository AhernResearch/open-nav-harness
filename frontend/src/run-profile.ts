import type { Row } from "./api";

/** Exploration and local repairs use the actor; visited-node retracing remains available. */
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
