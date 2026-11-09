"""Episode-local navigation tools shared by direct, RPC and MCP transports."""

from __future__ import annotations
import base64
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Literal
import requests
from .registry import Image, TextContent, ToolRegistry, ToolResult
from .agentic import PlanItem, PlanPatch
from navgpt.vla.profiles import input_profile


def _sibling(module_name, attr):
    try:
        mod = __import__("navgpt.planner.tools." + module_name, fromlist=[attr])
    except ImportError:
        import importlib.util

        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(here, module_name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(module_name, mod)
        spec.loader.exec_module(mod)
    return getattr(mod, attr)


save_frame = _sibling("frame_store", "save_frame")
read_frame = _sibling("frame_store", "read_frame")
frame_exists = _sibling("frame_store", "frame_exists")


# Environment verbs that can change the pose; CAPTURE_VERBS also record passed frames.
MOTION_VERBS = frozenset(("step_hightolow", "retrace_to", "retrace_path", "step_discrete", "step_velocity", "teleport"))
CAPTURE_VERBS = frozenset(("step_hightolow", "retrace_to", "retrace_path", "step_discrete"))


class MotionRejected(RuntimeError):
    """The environment refused a movement before its first step: nothing moved."""


class NavigationSession:
    """Episode-local navigation tool state. One instance per episode; no mutable module state."""

    # After this many consecutive identical navigate_relative calls each walking less
    # than RELATIVE_NO_PROGRESS_M, the next identical call is refused before moving.
    RELATIVE_NO_PROGRESS_LIMIT = 3
    RELATIVE_NO_PROGRESS_M = 0.1

    def __init__(self, env=None, actor=None, cancelled=None):
        self.env = dict(os.environ if env is None else env)
        self._actor = actor
        self._cancelled = cancelled
        self.SERVER_URL = self.env.get("NAVGPT_SERVER_URL", "http://127.0.0.1:9200")
        self.VERB_PREFIX = self.env.get("NAVGPT_VERB_PREFIX", "r2rce")
        self.STEP_BUDGET = int(self.env.get("NAVGPT_STEP_BUDGET", "500"))
        self.LIVE_DIR = Path(self.env["NAVGPT_LIVE_DIR"]) if self.env.get("NAVGPT_LIVE_DIR") else None
        self.CONDITION = self.env.get("NAVGPT_CONDITION", "planner_basic")
        self._enabled_tools = json.loads(self.env["NAVGPT_TOOLS"]) if "NAVGPT_TOOLS" in self.env else None
        if self._enabled_tools is not None and (
            not isinstance(self._enabled_tools, list) or not self._enabled_tools
            or any(not isinstance(name, str) for name in self._enabled_tools)
        ):
            raise ValueError("NAVGPT_TOOLS must be a non-empty list of tool names")
        self._AGENTIC = self.CONDITION == "planner_agentic"
        self._SUBTASK = self.CONDITION in ("planner_subtask", "planner_agentic")
        self._rollout_limits = None
        self.VLA_URL = self.env.get("NAVGPT_VLA_URL", "http://127.0.0.1:8000")
        self.VLA_BACKEND = self.env.get("NAVGPT_VLA_BACKEND", "navgpt")
        self._input_profile = input_profile(self.VLA_BACKEND)
        self.VLA_MAX_STEPS = int(self.env.get("NAVGPT_VLA_MAX_STEPS", "200"))
        self.VLA_MAP = self.env.get(
            "NAVGPT_VLA_MAP", "1" if self.CONDITION in ("planner_vla_memory", "planner_subtask", "planner_agentic") else "0"
        ) not in ("0", "", "false")
        self.PLACE_LISTING_MAX = int(self.env.get("NAVGPT_PLACE_LISTING_MAX", "24"))
        self.MOVE_ROUTES = self.env.get("NAVGPT_MOVE_ROUTES", "1") not in ("0", "", "false")
        self.MOVE_VIEW = self.env.get("NAVGPT_MOVE_VIEW", "1") not in ("0", "", "false")
        self.MAX_KEYFRAMES = int(self.env.get("NAVGPT_MAX_KEYFRAMES", "8"))
        self._HAS_MOVE = self.CONDITION == "planner_motion"
        self._HAS_VLA = self.CONDITION in ("planner_vla_memory", "planner_subtask", "planner_agentic")
        self._HAS_PLACES = self.CONDITION in ("planner_memory", "planner_vla_memory", "planner_subtask", "planner_agentic")
        self._HAS_MOVE = self._HAS_MOVE or self._HAS_PLACES
        self._HAS_PANO = self._HAS_MOVE or self._HAS_VLA
        self.INTERFACE = self.env.get("NAVGPT_INTERFACE_VARIANT", "standard")
        self.BUNDLE = self.env.get("NAVGPT_CAPABILITY_BUNDLE", "full")
        self.REVIEW_EVERY = int(self.env.get("NAVGPT_REVIEW_EVERY", "0"))
        self._ABLATION = self.INTERFACE != "standard"
        self._CAPABILITIES = _sibling("ablation", "CAPABILITIES")
        _sibling("ablation", "validate")(self.INTERFACE, self.BUNDLE, self.REVIEW_EVERY)
        self._scheduler = _sibling("ablation", "ReviewScheduler")(self.REVIEW_EVERY)
        self.tools = ToolRegistry()
        self._t0 = time.time()
        self._obs_count = 0
        self._steps_taken = 0
        self._motion_epoch = 0
        self._memory_map_read = None
        # Consecutive identical navigate_relative calls that barely moved (agentic only).
        self._relative_streak = {"args": None, "epoch": -1, "count": 0}
        self._episode_over = False
        self._end_reason = ""
        self._OBSERVE_DESC = "Look through the robot's forward-facing camera and return the RGB view."
        self._STEP_DESC = "Execute movement actions in order.\nActions: 0 = STOP (permanently ENDS the episode — issue it only when you believe you have reached the instruction's endpoint), 1 = move forward 0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.\nExecutes sequentially and halts early if the episode ends. Returns how many low-level steps were taken and whether the episode is over."
        self._recent_frames: list[str] = []
        self.LIVE_FORMAT = (self.env.get("NAVGPT_LIVE_FORMAT") or "jpg").lower()
        self.LIVE_JPEG_QUALITY = int(self.env.get("NAVGPT_LIVE_JPEG_QUALITY") or "88")
        self._tool_ctx: dict[str, Any] = {"tool": None, "call": 0}
        self._tool_call_counts: dict[str, int] = {}
        self._vla = None
        self._history_pending = None
        self._actor_history_start = 0
        self._actor_history_policy = self.env.get("NAVGPT_ACTOR_HISTORY_POLICY", "episode")
        if self._actor_history_policy not in ("episode", "completed_subtask"):
            raise ValueError("Unknown actor history policy")
        if self._actor_history_policy == "completed_subtask" and (
            not self._AGENTIC or self._input_profile.history != "episode"
        ):
            raise ValueError("completed_subtask requires an agentic episode-history actor")
        self._last_policy_inference_id = None
        self._last_pose = None
        self._last_scan = None
        self._last_revisit = None
        self._instruction = self.env.get("NAVGPT_INSTRUCTION", "")
        self._tasks = (
            (_sibling("agentic", "AgenticTasks") if self._AGENTIC else _sibling("subtasks", "SubtaskManager"))(
                self._instruction,
                self.env.get("NAVGPT_SUBTASK_DIR")
                or (self.LIVE_DIR / "subtasks" if self.LIVE_DIR else None),
            )
            if self._SUBTASK
            else None
        )
        self._vla_calls = 0
        self._policy_step = 0
        self._presentation_by_hash = {}
        self._LOOK_DESC = "Scan the surroundings in ONE call. Returns four camera views — ahead, right, behind, left.\n\nEVERY LABEL IS PAINTED ON THE PICTURE ITSELF, so you never have to work out which text belongs to which view:\n  * the direction word on each view, with how far you would actually get walking that way (measured on the floor plan, sliding along walls exactly as a real move does — so a small number means genuinely blocked, not just 'something is close');\n  * a ruler along the bottom of each view giving the BEARING of each column, written as the exact number navigate_relative() takes (positive = left). This is how you turn 'the doorway is in the left third of the AHEAD view' into a move: read the tick under it. A 120-degree view has no single direction, so the word AHEAD alone cannot tell you where in it to go;\n  * VIEW n, your position as X and Y in metres from where the episode started, and MAP-UP — the turn that faces the top of the map. LEFT and AHEAD mean different world directions after you turn, so MAP-UP is what lets a view you saw twenty turns ago still be placed;\n\nThe cameras render simultaneously, so this does not move the robot and does not change where you stand or which way you face. Use it freely at any junction or whenever you are unsure — turning to look is never necessary.\n\nFor IDENTIFYING a particular object — is that a fire extinguisher, is that chair the one described — use observe_forward() instead: it returns the forward view alone at full size, and detail is what it is for."
        self._MOVE_DESC = "Turn to a relative bearing and walk. turn_deg: 0 = straight ahead, POSITIVE = LEFT, negative = right. distance_m: how far to try to walk.\nThe turn does not use the movement budget; the walking does, one step per 0.25 m budget, so it costs the same as stepping that far, not less.\nIt walks the floor-plan route to the point you asked for when that route is nearly the straight line — so a chair or a door frame in the way is walked around rather than stopped at, and detoured tells you when that happened. If the only route is a long way round, it does NOT take it: it walks straight and stops, because that means the way you asked for is genuinely blocked and you need to know.\nThe result ENDS WITH THE FOUR VIEWS from where you arrive, so you never need a observe_panorama straight after moving.\nReturns walked_m against requested_m, and blocked. walked_m is your ACTUAL displacement: it is often less than you asked for even when blocked is false, because the robot slides along walls and furniture rather than stopping dead. blocked=true means genuinely obstructed, never out-of-budget.\nnearest_obstacle_before_m is a depth measurement along that bearing, and it is a ROUGH GUIDE ONLY — it is the nearest surface in a 20-degree cone, so a single chair leg or door frame in that cone reads as a tiny clearance even though you can walk straight past it; and it is measured at camera height, so a low step or a bed frame does not appear at all. Do not let a small value stop you asking for a longer distance: ask, then read walked_m to learn what actually happened."
        self._MAP_DESC = "A top-down photograph of everywhere you have looked, with the path you have already walked drawn on it.\n\nCells carry the real colours your cameras measured there, seen from above — so a red rug reads red, a wooden floor reads brown. The flat slate checker is territory you have never observed: unknown, not empty — anything your cameras did see is drawn lighter than it, however dark the surface was.\n\nYour route is drawn over it as a line that ages from BLUE to RED: the blue dot is where the episode started, red is where you have just been, and the RED ARROW is you — it points the way you currently face. Reading the colour tells you which way round you walked a loop, not just that you walked one.\n\nPlaces you can return to are numbered AMBER BADGES, and the place you are standing on is an amber ring around you — the same numbers the place list underneath the map uses, so a mark on the picture and a line of the list are the same thing. That list comes with every map: each place's coordinates in metres from your start, your own name for it, how far away it is, and what far walking back to it would be.\n\nThe map is never rotated to your heading, so a loop looks like a loop across calls — the arrow moves, the picture does not. You do not need to work out which way the picture is pointing: the result tells you which way to TURN to face each place, in the same degrees navigate_relative() takes, and where up-the-map is relative to you if you want to read the image directly. It also gives the pixel coordinates of you and of your start, how far you are from the start, and how big the frame is. The frame is cropped to your route and what you have seen around it, so it grows as you explore: check frame_extent_m before comparing two calls by eye.\n\nIt does not move the robot, and it refreshes the surroundings as it draws, so you never need a observe_panorama just to update it.\n\nUse it to answer the question a forward camera cannot: have I been here before, and which parts of this floor have I still never looked at? Going in circles is one of the most common ways this task is failed, and it is invisible from the camera alone — the result also reports revisiting_earlier_position outright."
        self._VLA_DESC = (
            "Give a navigation instruction to NavGPT VLA, which drives the robot along the route for you.\n\nWHAT TO PASS: on the first call, the full original episode instruction. Before a later corrective call, compare the returned views and route evidence with the original goal. Pass a self-contained instruction for the remaining route from the observed current position, including the next supported landmark or direction and the original stopping condition. Omit completed clauses; do not invent landmarks or change the goal. Reuse the previous instruction only for intentional continuation when it still fits. With no argument, a scheduled pause resumes its active instruction; otherwise the episode's original instruction is used. Always pass corrective instructions explicitly.\n\nWHAT IT DOES: drives the robot until it judges the route finished, or until its step allowance for this call runs out. It moves the robot for real — those steps come out of your movement budget.\n\nWHAT YOU GET BACK: photographs taken along the way, then four views from wherever the robot ended up (ahead / right / behind / left, each with measured clearance in metres), then a summary: how many steps it took, how far it travelled, how far it ended from where it started, how often it hit something, and vla_suggests_arrival.\n"
            + (
                "\nAND A MAP, last: a top-down view of everywhere you have looked, your route ageing from blue at the start to red at the present, and a red arrow where NavGPT VLA left you, pointing the way you now face. The photographs say what is around you; the map says where that is relative to where you began.\n"
                if self.VLA_MAP
                else ""
            )
            + "\nWHAT IT CANNOT DO: end the episode, and recognise arrival. vla_suggests_arrival only means it stopped producing movement — it is wrong in both directions, so never treat it as confirmation. Only your own STOP ends the episode, and judging whether you are actually at the endpoint the instruction describes is the part you cannot hand over.\n\nHOW TO USE IT: normally one call at the start. Then read the four returned views against the instruction, walk the last few metres yourself with navigate_relative() if it finished short or overshot, and STOP when the view matches the endpoint. The four views it returns are exactly what observe_panorama() gives, so do not spend a turn on observe_panorama straight afterwards."
        )
        self._place_labels: dict[int, dict[str, Any]] = {}
        self.LABELS_PATH = self.env.get("NAVGPT_PLACE_LABELS") or ""
        self._place_frames: dict[int, str] = {}
        self.PLACE_FRAME_NEAR_M = float(self.env.get("NAVGPT_PLACE_FRAME_NEAR_M", "1.5"))
        self.LABEL_NEAR_M = float(self.env.get("NAVGPT_LABEL_NEAR_M", "1.5"))
        self._ADD_PLACE_DESC = 'Record where you are as a named place, so you can come back to it.\n\nname is a short handle you will use later — "kitchen doorway", "pool corner". caption is one sentence that will let you recognise it when you have walked away and forgotten: what is around it, which way the exits go, what made you notice it. Keep the caption under about 120 characters.\n\nNew places appear on their own as you cover new ground, and the result of every move tells you when one has appeared without a name. Name it then, while you are standing there — a place called "place 7" is no use to you later, and the caption is the whole difference between a map and a memory.\n\nYou can also name somewhere the moment you decide it matters, without waiting to be asked: if you think you have arrived at the endpoint the instruction describes, name it BEFORE you look any further. Then, if the search turns out to be wrong, you can walk back to it and stop there.\n\nThe label describes WHERE YOU ARE STANDING. It attaches to the place nearest you, and the result says which and how far away it was.\n\nIt does not move the robot.'
        self._GO_DESC = 'Walk to somewhere you recorded earlier: navigate_to_node(2) by its number, or navigate_to_node("kitchen doorway") by its name. The robot retraces ground it has already walked, so the route is one it has proved is walkable.\nThis is for GOING BACK. To cover new ground use navigate_relative(turn_deg, distance_m), reading the bearing off the ruler burned under whatever you want to walk towards.\nNumbers last the whole episode. observe_map() lists every place with its coordinates, your name for it, and how far walking back would be.'
        self._trace: list[dict[str, Any]] = []
        self._trace_keys: list[int] = []
        self._ARRIVE_DESC = 'END THE EPISODE BY SAYING WHERE YOU ARE STOPPING. This is how you finish; it is permanent.\n\nterminate_episode("here") stops where you stand. terminate_episode(3) walks back to place 3 first and stops there — the retrace walks back over ground you already covered.\n\nWHY THIS TOOL EXISTS instead of a plain STOP. Ending the episode is a CHOICE BETWEEN CANDIDATES — where you stand, or somewhere you recorded — rather than a reflex from wherever the robot happens to be. Both errors cost the same episode and neither is the safe default: always preferring the place you bound to the last clause loses the episodes where you were already at the end, and always preferring where you stand loses the episodes where the route plainly continued past you. So name your target and say why it beats the other.\n\nYou do not need to be certain, see the landmark, or be able to name the room. You need to be within 3 metres of where the last clause points.'
        self._RECALL_DESC = "Show the picture that was taken at a place you recorded — observe_node(3).\n\nUse it when a caption is not enough to tell two places apart, which is the one moment an image of a place is worth asking for. It is deliberately NOT attached to the place list: an image added to this conversation is re-sent on every following turn, so images per place quickly become expensive. So the list carries words, the map carries where everything is, and this carries the pixels only when you ask.\nIt does not move the robot."
        if self._AGENTIC:
            self._register("observe", self.observe_views.__doc__, self.observe_views)
        else:
            self._register("observe_forward", self._OBSERVE_DESC, self.observe)
        if self.CONDITION != "planner_basic":
            self._register(
                "terminate_episode",
                self.finish_agentic_episode.__doc__ if self._AGENTIC else self.finish_subtask_episode.__doc__ if self._SUBTASK else self._ARRIVE_DESC,
                self.finish_agentic_episode if self._AGENTIC else self.finish_subtask_episode if self._SUBTASK else self.arrive,
            )
        if not self._HAS_VLA:
            self._register("navigate_primitive", self._STEP_DESC, self.step)
        if self._HAS_PANO and not self._AGENTIC:
            self._register("observe_panorama", self._LOOK_DESC, self.look_around)
        if self._HAS_MOVE:
            self._register("observe_map", self._MAP_DESC, self.local_map)
            self._register("navigate_relative", self._MOVE_DESC, self.move)
        if self._HAS_PLACES:
            self._register("navigate_to_node", self._GO_DESC, self.return_node if self._AGENTIC else self.go)
            if not self._AGENTIC:
                self._register("annotate_node", self._ADD_PLACE_DESC, self.add_place)
        if self._HAS_PLACES and not self._HAS_VLA and not self._AGENTIC:
            self._register("observe_node", self._RECALL_DESC, self.recall)
        if self._AGENTIC:
            self._register("update_plan", self.update_agentic_plan.__doc__, self.update_agentic_plan)
            execute = self.execute_agentic_task
            description = execute.__doc__
            if self.env.get("NAVGPT_ALLOW_CONTEXT_RESET", "1") == "0" or not self._input_profile.manual_reset:
                execute = self.execute_agentic_task_without_reset
                description = description.replace(
                    "reset_context restarts visual history for this task.\n        ",
                    "",
                )
            self._register("execute_subtask", description, execute)
            self._register("update_memory", self.update_memory.__doc__, self.update_memory)
            # Optional recall: registered only when an experiment explicitly selects it.
            if self._enabled_tools is not None:
                self._register("read_memory", self.read_memory.__doc__, self.read_memory)
        elif self._SUBTASK:
            self._register("update_plan", self.update_plan.__doc__, self.update_plan)
            self._register("execute_subtask", self.execute_subtask.__doc__, self.execute_subtask)
            self._register("review_subtask", self.review_subtask.__doc__, self.review_subtask)
        elif self._HAS_VLA:
            self._register("navigate_by_instruction", self._VLA_DESC, self.vla_navigate)
        if self._enabled_tools is not None and set(self._enabled_tools) != set(self.tools.definitions):
            raise ValueError("Configured tools do not match the condition's registered surface")

    def _tool_enter(self, name: str) -> None:
        self._tool_call_counts[name] = self._tool_call_counts.get(name, 0) + 1
        self._tool_ctx["tool"] = name
        self._tool_ctx["call"] = self._tool_call_counts[name]

    def _encode_live(self, png: bytes) -> tuple[bytes, str]:
        """(bytes, extension) for the on-disk copy of a frame."""
        if self.LIVE_FORMAT == "png":
            return (png, "png")
        try:
            import io
            from PIL import Image as _PILImage

            im = _PILImage.open(io.BytesIO(png))
            im.load()
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=self.LIVE_JPEG_QUALITY, optimize=True)
            return (buf.getvalue(), "jpg")
        except Exception:
            return (png, "png")

    def _live_frame(self, png: bytes, label: str | None = None, kind: str = "view") -> None:
        """Dump a frame for spectating, and remember where it went.

        Images are written to LIVE_DIR (a run with live output); without it there
        are no stored frames for the captioner or node recall.

        Every frame also gets a line in ``frames.jsonl`` saying which tool drew it and
        what it shows, so a viewer never has to infer the attribution from the tool's
        text output.
        """
        if self.LIVE_DIR is None:
            return
        self.LIVE_DIR.mkdir(parents=True, exist_ok=True)
        (data, ext) = self._encode_live(png)
        f = self.LIVE_DIR / "obs_{:04d}_step{:03d}.{}".format(self._obs_count, self._steps_taken, ext)
        location = save_frame(f, data)
        save_frame(self.LIVE_DIR / ("latest." + ext), data)
        self._recent_frames.append(location)
        del self._recent_frames[:-8]
        try:
            with (self.LIVE_DIR / "frames.jsonl").open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "obs": self._obs_count,
                            "file": f.name,
                            "step": self._steps_taken,
                            "tool": self._tool_ctx["tool"],
                            "call": self._tool_ctx["call"],
                            "label": label or kind,
                            "kind": kind,
                            "t": round(time.time() - self._t0, 1),
                        }
                    )
                    + "\n"
                )
        except OSError:
            pass

    def _live_log(self, entry: dict[str, Any]) -> None:
        if self.LIVE_DIR is None:
            return
        self.LIVE_DIR.mkdir(parents=True, exist_ok=True)
        with (self.LIVE_DIR / "actions.log").open("a") as fh:
            fh.write(json.dumps({"t": round(time.time() - self._t0, 1), **entry}) + "\n")

    def _policy_event(self, kind, **fields):
        payload = {"prediction_id": self._policy_step, "policy_backend": self.VLA_BACKEND, **fields}
        self._live_log({"kind": kind, **payload})
        if self._tasks is not None:
            self._tasks._event(kind, **payload)

    def _feed_passive_history(self, frames):
        for frame in frames:
            if self._SUBTASK:
                self._tasks.observe(
                    self._policy_views(frame["views"]),
                    frame.get("pose") or {},
                    frame.get("step_count", self._steps_taken),
                )
            elif self.VLA_BACKEND == "lightnav":
                if self._history_pending is not None:
                    self._lazy_vla().observe_history(self._history_pending["views"])
                self._history_pending = frame
            else:
                self._lazy_vla().observe_history(frame["views"])

    def _prepare_policy_observation(self, pano):
        if self._SUBTASK:
            return
        if self.VLA_BACKEND == "lightnav" and self._history_pending is not None:
            if self._history_pending.get("policy_frame_id") != pano.get("policy_frame_id"):
                self._lazy_vla().observe_history(self._history_pending["views"])
            self._history_pending = None

    def _stage_policy_observation(self, pano):
        """Keep an unseen endpoint until inference or subsequent corrective motion."""
        if self._SUBTASK:
            return
        if self.VLA_BACKEND != "lightnav" or pano.get("policy_frame_id") == self._last_policy_inference_id:
            return
        image = (pano.get("raw_views") or {}).get("policy_front")
        if not image:
            raise RuntimeError("LightNav requires its unannotated policy camera")
        if self._history_pending is not None:
            if self._history_pending.get("policy_frame_id") == pano.get("policy_frame_id"):
                return
            self._lazy_vla().observe_history(self._history_pending["views"])
        self._history_pending = {
            "policy_frame_id": pano.get("policy_frame_id"),
            "views": {"policy_front": image},
        }

    def _call(self, function_name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        motion = function_name.rsplit("__", 1)[-1]
        moving = motion in MOTION_VERBS
        capture = (self._ABLATION or self.VLA_BACKEND == "lightnav" or self._SUBTASK) and motion in CAPTURE_VERBS
        if capture:
            inputs = dict(inputs, capture_history=True)
        if self._AGENTIC and moving:
            inputs = dict(inputs, capture_path=True)
        if self._cancelled is not None and self._cancelled.is_set():
            raise RuntimeError("Execution cancelled before environment call")
        # The environment reports every `error` output and 4xx before its first step, so
        # those are clean rejections. A lost reply or a 5xx may follow motion: uncertain.
        try:
            resp = requests.post(
                "{}/call/{}".format(self.SERVER_URL, function_name), json={"inputs": inputs}, timeout=300
            )
        except Exception:
            if moving:
                self._motion_dispatched()
            raise
        if resp.status_code >= 400:
            detail = ""
            try:
                detail = str((resp.json() or {}).get("error") or "")
            except ValueError:
                detail = resp.text[:200]
            detail = detail or "{} {}".format(resp.status_code, resp.reason)
            if moving and resp.status_code < 500:
                raise MotionRejected(detail)
            if moving:
                self._motion_dispatched()
            raise RuntimeError(detail)
        output = resp.json()["outputs"]
        if output.get("error") and self._AGENTIC:
            raise (MotionRejected if moving else RuntimeError)(str(output["error"]))
        if moving and not output.get("error"):
            self._motion_dispatched()
        if (
            self._SUBTASK
            and moving
            and (not output.get("error"))
            and (not (motion == "step_discrete" and inputs.get("action") == 0))
        ):
            self._tasks.mark_motion()
        if capture:
            self._feed_passive_history(output.pop("motion_history", []))
        if self._AGENTIC:
            for position in output.pop("motion_path", []):
                self._tasks.memory.record({"position": position})
            if output.get("pose", {}).get("position"):
                self._last_pose = output["pose"]
                self._tasks.memory.record(output["pose"])
        return output

    def _motion_dispatched(self):
        """The robot may have moved: older map reads no longer authorize memory writes."""
        self._motion_epoch += 1
        self._memory_map_read = None

    def _policy_views(self, views):
        names = ("policy_front",) if self.VLA_BACKEND == "lightnav" else ("front", "right", "back", "left")
        return {name: views[name] for name in names}

    def _lazy_vla(self):
        if self._actor is not None:
            return self._actor
        if self._vla is None:
            self._vla = _sibling("vla_client", "NavGPTVLAClient")(self.VLA_URL, backend=self.VLA_BACKEND)
        return self._vla

    def _pano(self) -> dict[str, Any]:
        """Fetch the 4-view panorama + range scan.

        This is also what grows the top-down map: the NavGPT Environment service unprojects these
        four RGB-D views into its fixed colour grid on every call, so looking is what
        fills the map in.

        Costs zero simulator steps — the four cameras render alongside the agent's
        forward camera in one observation.
        """
        self._recent_frames.clear()
        out = self._call("{}__observe_pano".format(self.VERB_PREFIX), {})
        if self._ABLATION:
            for name, value in (out.get("views") or {}).items():
                self._presentation_by_hash[hashlib.sha256(base64.b64decode(value)).hexdigest()] = (
                    out.get("presentation") or {}
                ).get(name, {})
        self._last_pose = out.get("pose")
        self._last_scan = out.get("scan")
        if "revisiting_earlier_position" in out:
            self._last_revisit = out["revisiting_earlier_position"]
        self._stage_policy_observation(out)
        if self._SUBTASK:
            self._tasks.observe(
                self._policy_views(out.get("raw_views") or {}), out.get("pose") or {}, self._steps_taken
            )
        if self._AGENTIC:
            frame = self._tasks.frames[-1]
            frame["planner_views"] = dict(out.get("views") or {})
            if self._tasks.directory:
                folder = self._tasks.directory / "observations" / frame["id"] / "planner"
                folder.mkdir(parents=True, exist_ok=True)
                for direction, value in frame["planner_views"].items():
                    if direction in ("front", "right", "back", "left"):
                        (folder / (direction + ".png")).write_bytes(base64.b64decode(value))
            self._tasks._event("planner_observation", frame_id=frame["id"], pose=frame["pose"],
                               directions=list(frame["planner_views"]))
        return out

    def _clearance_summary(self, scan, payload=None) -> dict[str, Any]:
        """Prefer the server's own figures when it sent them.

        The server BURNS these numbers into the pictures, so a second computation here
        could disagree with the painted label — the exact failure burning them in was
        meant to remove. It also reports can_walk_m (how far a move would actually get,
        simulated on the navmesh) which is a better answer to "can I go that way" than
        cone clearance is; cone clearance answers "am I about to scrape something".
        """
        if payload:
            walk = payload.get("can_walk_m")
            if walk:
                if self.INTERFACE == "absolute_bearings":
                    heading = payload["absolute_heading_deg"]
                    offsets = {"ahead_m": 0, "right_m": 90, "behind_m": 180, "left_m": 270}
                    return {"bearing_%03d_m" % ((heading + offsets[k]) % 360): v for (k, v) in walk.items()}
                return dict(walk)
            clear = payload.get("clearance_m")
            if clear:
                return dict(clear)
        return self._clearance_summary_from_scan(scan)

    def _clearance_summary_from_scan(self, scan) -> dict[str, Any]:
        """Calibrated clearance in the four cardinal directions, in metres.

        Depth-derived, so these are measurements rather than impressions — the point
        of the `move` surface. None means that bearing was never observed.
        """

        def wedge(centre_deg):
            if not scan:
                return None
            n = len(scan)
            half = 10
            vals = [scan[(int(centre_deg) + k) % n] for k in range(-half, half + 1)]
            vals = [v for v in vals if v is not None]
            return round(min(vals), 2) if vals else None

        return {"ahead_m": wedge(0), "left_m": wedge(90), "behind_m": wedge(180), "right_m": wedge(270)}

    def _pano_frames(self, pano, clear=None, prefix="") -> list[tuple[str, bytes]]:
        """The look, as (caption, png) pairs, one per view.

        ONE renderer for every place a look is shown — observe_panorama, the end of a
        VLA leg, arriving at a place, returning to a waypoint — so their captions
        cannot drift apart.

        `prefix` is what the caption says about WHY you are seeing this ("back at
        place 3", "where NavGPT VLA stopped").
        """
        views = pano.get("views") or {}
        if self._ABLATION:
            out = []
            for name in ("front", "right", "back", "left"):
                if not views.get(name):
                    continue
                cap = prefix
                out.append((cap.strip(), base64.b64decode(views[name])))
            return out
        clear = clear if clear is not None else self._clearance_summary(pano.get("scan"), pano)
        out = []
        for name, label, key in (
            ("front", "ahead", "ahead_m"),
            ("right", "right (+90 deg right)", "right_m"),
            ("back", "behind (180 deg)", "behind_m"),
            ("left", "left (+90 deg left)", "left_m"),
        ):
            b64 = views.get(name)
            if not b64:
                continue
            dist = (clear or {}).get(key)
            cap = "{} — clearance {}".format(label, "{} m".format(dist) if dist is not None else "unknown")
            out.append((prefix + " — " + cap if prefix else cap, base64.b64decode(b64)))
        return out

    def _pano_content(self, pano, clear=None, prefix="") -> list[Any]:
        """_pano_frames as MCP content, captions dropped when they would be empty."""
        content: list[Any] = []
        for cap, png in self._pano_frames(pano, clear, prefix):
            self._bump_obs(png, cap.split(" — ")[0] if cap else "view")
            if cap:
                content.append(cap)
            content.append(Image(data=png, format="png"))
        return content

    def _capture_view(self) -> bytes:
        """Render the current egocentric RGB. Pure read — advances the live-frame
        counter, never the simulator."""
        self._recent_frames.clear()
        outputs = self._call("{}__observe_egocentric".format(self.VERB_PREFIX), {"stamp": self._HAS_PANO})
        _forward_presentation = outputs.get("presentation", {})
        png = base64.b64decode(outputs["rgb"])
        if self._ABLATION:
            self._presentation_by_hash[hashlib.sha256(png).hexdigest()] = _forward_presentation
        self._obs_count += 1
        self._live_frame(png, "forward view")
        return png

    @staticmethod
    def _validate_views(views):
        if not views or len(views) != len(set(views)) or any(
                v not in ("front", "right", "back", "left") for v in views):
            raise ValueError("views must select distinct front/right/back/left directions")

    def observe_views(self, views: list[Literal["front", "right", "back", "left"]]) -> list:
        """Inspect selected simultaneous camera directions without moving or rotating.

        Movement already returns endpoint views; request fresh views only when needed.
        Each image identifies its direction, capture orientation and approximate clearance.
        """
        self._validate_views(views)
        if self._episode_over:
            raise ValueError("Episode is already over")
        self._tool_enter("observe")
        pano = self._pano()
        selected = dict(pano, views={v: pano.get("views", {}).get(v) for v in views})
        frames = dict(zip((v for v in ("front", "right", "back", "left")
                           if selected["views"].get(v)), self._pano_frames(selected)))
        result = [json.dumps({"orientation": (pano.get("pose") or {}).get("rotation_wxyz"),
                              "missing_views": [v for v in views if v not in frames]})]
        for direction in views:
            if direction in frames:
                label, png = frames[direction]
                self._bump_obs(png, direction)
                result.extend([label, Image(data=png, format="png")])
        return result

    def return_node(self, node_id: int) -> list:
        return self._return_memory_node(node_id)

    def observe(self) -> list:
        self._tool_enter("observe_forward")
        if self._episode_over:
            return ["episode already over ({}); no image".format(self._end_reason)]
        png = self._capture_view()
        return [Image(data=png, format="png")]

    def step(self, actions: list[int]) -> list:
        self._tool_enter("navigate_primitive")
        if self._episode_over:
            return ["episode already over ({}); no more steps possible".format(self._end_reason)]
        if not actions:
            return ["no actions given — pass a non-empty list, e.g. navigate_primitive([1])"]
        executed: list[int] = []
        for action in actions:
            try:
                action = int(action)
            except (TypeError, ValueError):
                return ["invalid action {!r} — must be an integer 0-3".format(action)]
            if action not in (0, 1, 2, 3):
                return ["invalid action {} — must be 0, 1, 2 or 3".format(action)]
            outputs = self._call("{}__step_discrete".format(self.VERB_PREFIX), {"action": action})
            self._steps_taken += 1
            executed.append(action)
            if action == 0:
                self._episode_over = True
                self._end_reason = "stop_called"
                break
            if outputs.get("terminated") or outputs.get("truncated"):
                self._episode_over = True
                self._end_reason = "step_budget_exhausted"
                break
        status = {
            "actions_executed": executed,
            "steps_taken_total": self._steps_taken,
            "steps_remaining_approx": max(0, self.STEP_BUDGET - self._steps_taken),
            "episode_over": self._episode_over,
        }
        if self._episode_over:
            status["end_reason"] = self._end_reason
        if self._HAS_PLACES and (not self._episode_over):
            line = self._places_line()
            if line:
                status["nearest_place"] = line
        self._live_log({"step": executed, **status})
        return [json.dumps(status)]

    def look_around(self) -> list:
        self._tool_enter("observe_panorama")
        if self._episode_over:
            return ["episode already over ({}); no more looking".format(self._end_reason)]
        out = self._pano()
        clear = self._clearance_summary(out.get("scan"), out)
        content: list[Any] = self._pano_content(out, clear)
        status = {
            "can_walk_m": clear,
            "view": out.get("view_index"),
            "you_are_at_xy": out.get("you_are_at_xy"),
            "steps_taken_total": self._steps_taken,
            "steps_remaining_approx": max(0, self.STEP_BUDGET - self._steps_taken),
        }
        if self._HAS_MOVE and self._last_revisit is not None:
            status["revisiting_earlier_position"] = self._last_revisit
        content.append(json.dumps(status))
        if self._HAS_PLACES:
            cand = self._candidates_block()
            if cand:
                content.append(cand)
        self._live_log({"observe_panorama": True, "can_walk_m": status.get("can_walk_m")})
        self._frame_here()
        self._auto_label_start()
        return content

    def _auto_label_start(self) -> None:
        """Node 0 is where the episode began; say so without spending a turn on it."""
        if self._AGENTIC:
            return None
        if not self._HAS_PLACES or 0 in self._place_labels:
            return
        try:
            pos = (self._call("{}__agent_state".format(self.VERB_PREFIX), {}) or {}).get("position")
        except Exception:
            pos = None
        self._place_labels[0] = {
            "name": "start",
            "caption": "where the episode began",
            "clause": None,
            "anchor": list(pos) if pos else None,
            "frame": self._recent_frames[-1] if self._recent_frames else None,
            "auto": True,
        }
        self._dump_labels()

    def _detour_m(self, out: dict[str, Any]) -> float:
        try:
            return round(
                max(0.0, float(out.get("walked_m") or 0.0) - float(out.get("net_displacement_m") or 0.0)), 2
            )
        except (TypeError, ValueError):
            return 0.0

    def move(self, turn_deg: float, distance_m: float) -> list:
        self._tool_enter("navigate_relative")
        if self._episode_over:
            return ["episode already over ({}); no more moves".format(self._end_reason)]
        try:
            bearing = float(turn_deg)
            dist = float(distance_m)
        except (TypeError, ValueError):
            return ["invalid arguments — turn_deg and distance_m must be numbers"]
        if not math.isfinite(bearing) or not math.isfinite(dist) or dist < 0:
            raise ValueError("Finite turn_deg and nonnegative distance_m are required")
        streak = self._relative_streak
        if (self._AGENTIC and streak["args"] == (bearing, dist) and streak["epoch"] == self._motion_epoch
                and streak["count"] >= self.RELATIVE_NO_PROGRESS_LIMIT):
            raise ValueError("The same navigate_relative made no progress {} times; reassess the views "
                             "or choose another approach".format(streak["count"]))
        predicted = None
        try:
            fresh = self._pano().get("scan")
        except Exception:
            fresh = self._last_scan
        if fresh:
            n = len(fresh)
            half = 10
            centre = int(round(bearing)) % n
            vals = [fresh[(centre + k) % n] for k in range(-half, half + 1)]
            vals = [v for v in vals if v is not None]
            predicted = round(min(vals), 2) if vals else None
        epoch_before = self._motion_epoch
        out = self._call(
            "{}__step_hightolow".format(self.VERB_PREFIX),
            {"angle_rad": math.radians(bearing), "distance_m": dist, "route": self.MOVE_ROUTES},
        )
        self._steps_taken = int(out.get("step_count") or self._steps_taken)
        stalled = float(out.get("walked_m") or 0.0) < self.RELATIVE_NO_PROGRESS_M
        repeated = streak["args"] == (bearing, dist) and streak["epoch"] == epoch_before
        self._relative_streak = {"args": (bearing, dist), "epoch": self._motion_epoch,
                                 "count": (streak["count"] + 1 if repeated else 1) if stalled else 0}
        if out.get("terminated") or out.get("truncated"):
            self._episode_over = True
            self._end_reason = "step_budget_exhausted"
        status = {
            "turned_deg": out.get("turned_deg"),
            "requested_m": out.get("requested_m"),
            "walked_m": out.get("walked_m"),
            "blocked": out.get("blocked"),
            "detour_m": self._detour_m(out),
            "detoured": bool(out.get("routed")) and self._detour_m(out) > 0.2,
            "nearest_obstacle_before_m": predicted,
            "steps_taken_total": self._steps_taken,
            "steps_remaining_approx": max(0, self.STEP_BUDGET - self._steps_taken),
            "episode_over": self._episode_over,
        }
        if self._episode_over:
            status["end_reason"] = self._end_reason
        if self._HAS_PLACES and (not self._episode_over):
            line = self._places_line()
            if line:
                status["nearest_place"] = line
        self._live_log({"move": [bearing, dist], **status})
        self._frame_here()
        content: list[Any] = [json.dumps(status)]
        if self.MOVE_VIEW and self._HAS_PANO and (not self._episode_over):
            try:
                content += self._pano_content(self._pano())
            except Exception as exc:
                content.append("(view unavailable after the move: {!r})".format(exc))
        if self._HAS_PLACES and (not self._episode_over):
            cand = self._candidates_block()
            if cand:
                content.append(cand)
        return content

    def _fetch_map(self) -> tuple[bytes, dict[str, Any]] | tuple[None, str]:
        """The top-down map plus the fields worth reporting, or None if it failed.

        Shared by ``local_map`` and the tail of ``navigate_by_instruction`` so the two cannot
        drift into describing the same picture differently.
        """
        try:
            inputs = {}
            if self._AGENTIC:
                memory = self._tasks.memory
                edge = memory.edges[memory.pending] if memory.pending is not None else None
                inputs = {"memory_nodes": [{"id": n["id"], "position": n["position"]}
                                           for n in memory.nodes.values()],
                          "segment_path": edge["trajectory"] if edge else []}
            out = self._call("{}__observed_map".format(self.VERB_PREFIX), inputs)
            if self._AGENTIC and out.get("node_layer") != "memory":
                raise RuntimeError("Environment lacks topology memory map support; update the owned environment")
            png = base64.b64decode(out["png"])
            if self._SUBTASK:
                self._tasks.mark_map()
        except Exception as exc:
            return (None, "{}: {}".format(type(exc).__name__, exc))
        fields: dict[str, Any] = {}
        for key, src in (
            ("facing", "facing"),
            ("you_are_at_px", "you_are_at_px"),
            ("started_at_px", "started_at_px"),
            ("straight_line_from_start_m", "straight_line_from_start_m"),
            ("walked_m", "walked_m"),
            ("revisiting_earlier_position", "revisiting_earlier_position"),
            ("seen_area_m2", "seen_area_m2"),
            ("unobserved_fraction_of_frame", "unobserved_fraction_of_frame"),
            ("map_extent_m", "frame_extent_m"),
            ("metres_per_pixel", "meters_per_pixel"),
            ("frame", "frame"),
            ("legend", "legend"),
        ):
            if out.get(src) is not None:
                fields[key] = out[src]
        if self._AGENTIC:
            fields["places_px"] = out.get("places_px", {})
            fields["segment_legend"] = out.get("segment_legend")
        if self._ABLATION:
            for key in (
                "places_px",
                "presentation_transform",
                "absolute_heading_deg",
                "up_on_the_map_is",
                "caption",
                "direction_cues_px",
            ):
                if out.get(key) is not None:
                    fields[key] = out[key]
        if self._ABLATION:
            self._presentation_by_hash[hashlib.sha256(png).hexdigest()] = fields
        return (png, fields)

    def _dump_labels(self) -> None:
        """Mirror the labels to disk. Never raise: telemetry must not end an episode."""
        if not self.LABELS_PATH:
            return
        try:
            with open(self.LABELS_PATH, "w") as fh:
                json.dump(
                    {
                        str(k): {kk: vv for (kk, vv) in v.items() if kk != "frame"}
                        for (k, v) in self._place_labels.items()
                    },
                    fh,
                )
        except Exception as exc:
            self._live_log({"dump_labels_error": str(exc)[:120]})

    def _fetch_places(self) -> dict[str, Any] | None:
        try:
            return self._call("{}__places".format(self.VERB_PREFIX), {})
        except Exception:
            return None

    def _frames_from_trace(self, trace) -> None:
        """Give every place the SPECIALIST drove through a picture. Free.

        The right stopping place is typically mid-route, driven through by NavGPT VLA
        during one rollout and never looked at by the agent, so frames captured only
        where the agent looked would leave it without a picture. But the rollout DID
        photograph it: every trace entry carries a front view and the pose it was taken
        from. Matching those poses to places costs nothing and covers the whole route,
        which is the part the endgame decision is actually about.

        Keeps, per place, the frame whose pose is CLOSEST to it, so a picture is never
        attributed to a place the camera was two metres from when a nearer one exists.
        """
        if self._AGENTIC:
            return None
        if not (self._HAS_PLACES and trace) or self.LIVE_DIR is None:
            return
        payload = self._fetch_places()
        if not payload:
            return
        places = [pl for pl in payload.get("places") or [] if pl.get("position")]
        if not places:
            return
        best: dict[int, tuple[float, bytes]] = {}
        for fr in trace:
            pos = (fr.get("pose") or {}).get("position")
            png = fr.get("png")
            if not pos or not png:
                continue
            for pl in places:
                wp = pl["position"]
                d = math.hypot(float(pos[0]) - float(wp[0]), float(pos[2]) - float(wp[2]))
                if d > self.PLACE_FRAME_NEAR_M:
                    continue
                pid = int(pl["id"])
                if pid not in best or d < best[pid][0]:
                    best[pid] = (d, png)
        if not best:
            return
        try:
            self.LIVE_DIR.mkdir(parents=True, exist_ok=True)
            for pid, (d, png) in best.items():
                if (self._place_labels.get(pid) or {}).get("frame"):
                    continue
                f = self.LIVE_DIR / "place_{:03d}.png".format(pid)
                self._place_frames[pid] = save_frame(f, png)
        except OSError as exc:
            self._live_log({"frames_from_trace_error": str(exc)[:120]})
        else:
            self._live_log({"frames_from_trace": sorted(best)})

    def _frame_here(self) -> None:
        """Remember the picture taken at the place the robot is standing on. Free."""
        if self._AGENTIC:
            return None
        if not (self._HAS_PLACES and self._recent_frames):
            return
        payload = self._fetch_places()
        if not payload:
            return
        here = payload.get("nearest_place")
        if here is None:
            return
        pl = next((p for p in payload.get("places") or [] if p["id"] == here), None)
        if pl is None or (pl.get("straight_m") or 99.0) > self.LABEL_NEAR_M:
            return
        self._place_frames[int(here)] = self._recent_frames[-1]

    def _pending_place(self, payload: dict[str, Any]) -> int | None:
        """The newest place the agent has not captioned yet, if any.

        Derived by comparing ids against the labels the MCP tool server holds, so it needs no
        server state: the graph proposes, the agent names, and "which one still needs a
        name" is just set arithmetic.

        Used only for the NAG ("place 4 still has no name"). It must NOT be used to pick
        what a bare ``annotate_node("name")`` labels — see ``_label_target``.
        """
        ids = [p["id"] for p in payload.get("places") or []]
        unnamed = [i for i in ids if i not in self._place_labels]
        return max(unnamed) if unnamed else None

    def _label_target(self, payload: dict[str, Any]) -> int | None:
        """Which place a bare ``annotate_node("name", "caption")`` describes.

        Not ``_pending_place`` (the HIGHEST-NUMBERED uncaptioned place, wherever it
        is). Under `planner_vla_memory` NavGPT VLA drives the whole route in one call,
        so the graph gains places 0..N at once; the agent then walks back to the spot
        the last clause describes and names it, and labelling the newest place would
        attach the name to NavGPT VLA's OVERSHOOT node instead.

        A label describes WHERE THE ROBOT IS. So: among the places within
        ``LABEL_NEAR_M`` of the robot prefer an uncaptioned one (which keeps the
        "caption the place you just made" intent), else the nearest of them. Only when
        nothing at all is within reach does it fall back to the nearest place, and the
        caller's ``note`` already says so out loud.
        """
        places = payload.get("places") or []
        near = [
            p
            for p in places
            if (p.get("straight_m") if p.get("straight_m") is not None else 99.0) <= self.LABEL_NEAR_M
        ]
        if near:
            uncaptioned = [p for p in near if p["id"] not in self._place_labels]
            pool = uncaptioned or near
            return min(pool, key=lambda p: p.get("straight_m") or 0.0)["id"]
        return payload.get("nearest_place")

    def _place_line(self, pl: dict[str, Any], nearest: int | None) -> list[str]:
        label = self._place_labels.get(pl["id"])
        name = '"{}"'.format(label["name"]) if label else "(awaiting a caption)"
        here = " <- you are here" if pl["id"] == nearest else ""
        out = [
            "{:>3}  ({:6.2f},{:6.2f})  {:22} {:5.1f} m away, {} · walk back {} · visits {} · links {}{}".format(
                pl["id"],
                pl["xy"][0],
                pl["xy"][1],
                name,
                pl["straight_m"],
                pl.get("turn", "?"),
                "{:.1f} m".format(pl["route_m"]) if pl["route_m"] is not None else "no walked route",
                pl["visits"],
                ",".join((str(i) for i in pl["links"])) or "-",
                here,
            )
        ]
        if label and label.get("clause") is not None:
            out[0] += " · clause {}".format(label["clause"])
        if label and label.get("caption"):
            out.append("       {}".format(label["caption"]))
        return out

    def _format_places(self, payload: dict[str, Any]) -> str:
        """The place table, in one place so no two callers can print it differently."""
        places = payload.get("places") or []
        nearest = payload.get("nearest_place")
        head = "places ({}) · you are at ({:.2f}, {:.2f}) · x/y are metres along the map's axes from where you started · every direction below is a turn in the same degrees navigate_relative() takes, positive = left".format(
            len(places), *(payload.get("you_are_at_xy") or [0.0, 0.0])
        )
        if self._ABLATION:
            head = "places: x/y metres from start in fixed map axes; relative turns positive left"
        if self.INTERFACE == "absolute_bearings":
            head = "places: x/y metres from start in fixed map axes; bearings clockwise from map-north"
        ordered = sorted(places, key=lambda pl: pl["straight_m"])
        keep = ordered[: self.PLACE_LISTING_MAX]
        lines = [head]
        for pl in keep:
            lines += self._place_line(pl, nearest)
        if len(ordered) > len(keep):
            rest = [pl["id"] for pl in ordered[len(keep) :]]
            lines.append(
                "       ... and {} more, all drawn on the map: {}".format(
                    len(rest), ",".join((str(i) for i in rest))
                )
            )
        if payload.get("note"):
            lines.append("       {}".format(payload["note"]))
        pending = self._pending_place(payload)
        if self._AGENTIC:
            return "\n".join(lines)
        if pending is not None and (not self._ABLATION):
            lines.append(
                '       place {} still has no name — annotate_node("name", "caption") records it'.format(
                    pending
                )
            )
        return "\n".join(lines)

    def _places_line(self) -> str | None:
        """One line for a movement result: the nearest place and what a return costs.

        This rides on every navigate_primitive() and navigate_relative() on purpose: a
        fact the agent must choose to fetch tends to be ignored, and the moment that
        matters is precisely while it is walking away from a place it named.
        """
        if self._AGENTIC:
            return None
        payload = self._fetch_places()
        if not payload or not payload.get("places"):
            return None
        nearest = payload.get("nearest_place")
        pl = next((p for p in payload["places"] if p["id"] == nearest), None)
        if pl is None:
            return None
        label = self._place_labels.get(pl["id"])
        name = '"{}"'.format(label["name"]) if label else "unnamed"
        if pl["route_m"] is None:
            cost = "no walked route back"
        else:
            cost = "{} steps to return".format(int(math.ceil(max(pl["route_m"], pl["straight_m"]) / 0.25)))
        line = "{} {} at ({:.1f}, {:.1f}) — {:.1f} m away, {}, {}".format(
            pl["id"], name, pl["xy"][0], pl["xy"][1], pl["straight_m"], pl.get("turn", "?"), cost
        )
        pending = self._pending_place(payload)
        if pending is not None and pending != pl["id"]:
            line += " · place {} is new and unnamed".format(pending)
        elif pending is not None and not self._AGENTIC:
            line += ' · unnamed: annotate_node("name", "caption")'
        return line

    def local_map(self) -> list:
        """Top-down colour map of what the cameras have actually seen. See _MAP_DESC.

        Asks the NavGPT Environment service for ``observed_map``, which unprojects the
        four RGB-D pano views into a top-down grid: a cell is painted only where a
        measurement from a pose the agent actually occupied landed in it.

        That keeps the map honest — nothing is disclosed that a depth camera at those
        poses could not have measured, and rooms never looked into stay unknown —
        while surfaces come out in their real colours, which is far easier to match
        against an instruction ("the red rug") than grey blobs.

        Every call takes a fresh pano first. That costs no movement steps and is what
        makes the tool answer about *now*: without it the map would show only what
        the last observe_panorama saw, so calling it after walking would place the
        agent outside its own coverage.
        """
        self._tool_enter("observe_map")
        self._memory_map_read = None
        self._pano()
        if self._last_pose is None:
            return ["no pose available yet"]
        got = self._fetch_map()
        if got[0] is None:
            return [
                "map unavailable ({}) — nothing was drawn. Navigate from the views for now.".format(got[1])
            ]
        (png, status) = got
        self._bump_obs(png, "map", kind="map")
        table = None
        if self._HAS_PLACES and not self._AGENTIC:
            payload = self._fetch_places()
            if payload and payload.get("places"):
                table = self._format_places(payload)
        self._live_log(
            {
                "local_map": True,
                "seen_area_m2": status["seen_area_m2"],
                "revisiting": status["revisiting_earlier_position"],
            }
        )
        if self._AGENTIC:
            status = self._grant_map_read(png, status, "observe_map")
        out: list[Any] = [Image(data=png, format="png"), json.dumps(status)]
        if table:
            out.append(table)
        return out

    def _grant_map_read(self, png: bytes, status: dict, source: str) -> dict:
        """Record a returned map as the memory-write credential; return its model view.

        The model reads it in the result, so a later update_memory may caption the
        pending segment. Any movement after this map revokes it.
        """
        self._memory_map_read = {
            "motion_epoch": self._motion_epoch,
            "source": source,
            "map_call": self._tool_call_counts.get(source, 0),
            "map_sha256": hashlib.sha256(png).hexdigest(),
            "node_ids": sorted(int(n) for n in status.get("places_px", {})),
            "frame_id": self._tasks.frames[-1]["id"] if self._tasks.frames else None,
            "pending_edge_id": self._tasks.memory.pending,
        }
        self._tasks._event("memory_map_read", **self._memory_map_read)
        return {"facing": status.get("facing"), "places_px": status.get("places_px"),
                "memory_context": self._tasks.memory.context(self._tasks.active_id)}

    def _segment_map(self) -> list:
        """The map after a recorded movement, so its caption needs no extra observe_map turn."""
        if self._episode_over:
            return []
        got = self._fetch_map()
        if got[0] is None:
            return ["map unavailable after this movement ({}); call observe_map before update_memory".format(got[1])]
        png, status = got
        self._bump_obs(png, "map", kind="map")
        return [Image(data=png, format="png"), json.dumps(self._grant_map_read(png, status, "movement"))]

    def vla_navigate(self, instruction: str = "") -> list:
        """Delegate a route to NavGPT VLA. See _VLA_DESC.

        The first call uses the full episode instruction. The Planner may explicitly
        revise later instructions using observed progress while preserving the goal.
        An omitted instruction resumes the active route at a scheduled pause, and
        otherwise defaults to the original episode instruction. STOP remains the
        Planner's decision.
        """
        self._tool_enter("navigate_by_instruction")
        self._vla_calls += 1
        if self._episode_over:
            return ["episode already over ({}); NavGPT VLA cannot run".format(self._end_reason)]
        active_route = self._scheduler.route if self._ABLATION and self._scheduler.suspended else ""
        route = (instruction or "").strip() or active_route or (self._instruction or "").strip()
        if not route:
            return ["no instruction available — pass the episode's full instruction text"]
        limits = self._rollout_limits or {
            "max_actions": int(self.env.get("NAVGPT_SUBTASK_MAX_ACTIONS", "80")),
            "max_distance_m": float(self.env.get("NAVGPT_SUBTASK_MAX_DISTANCE", "3")),
            "max_turn_deg": float(self.env.get("NAVGPT_SUBTASK_MAX_TURN", "180")),
        }
        remaining = max(1, self.STEP_BUDGET - self._steps_taken)
        budget = remaining if self.VLA_MAX_STEPS <= 0 else max(1, min(self.VLA_MAX_STEPS, remaining))
        if self._SUBTASK and not self._AGENTIC:
            budget = min(budget, limits["max_actions"])
        spec = self._lazy_vla()
        grant = self._scheduler.begin(route) if self._ABLATION else None
        if self._ABLATION and self.VLA_MAX_STEPS > 0:
            budget = max(1, min(budget, self.VLA_MAX_STEPS - self._scheduler.used))
        trace: list[dict[str, Any]] = self._scheduler.trace if self._ABLATION else []
        handoff = None
        start_pose = trace[0]["pose"] if trace else None
        travelled = self._scheduler.travelled if self._ABLATION else 0.0
        blocked_count = self._scheduler.blocked_streak if self._ABLATION else 0
        pose_unchanged = _sibling("vla_client", "pose_unchanged")
        (prev_pose, stuck) = (None, False)
        arrival = False
        executed = 0
        turned = 0.0
        subtask_return = "action_budget"
        for i in range(budget):
            pano = self._pano()
            views = pano.get("views") or {}
            pose = pano.get("pose") or {}
            if start_pose is None:
                start_pose = pose
            front = views.get("front")
            if front:
                trace.append(
                    {
                        "step": self._scheduler.used if self._ABLATION else executed,
                        "png": base64.b64decode(front),
                        "travelled": travelled,
                        "pose": pose,
                        "presentation": (pano.get("presentation") or {}).get("front", {}),
                        "source_sha256": {
                            k: hashlib.sha256(base64.b64decode(v)).hexdigest()
                            for (k, v) in (pano.get("raw_views") or {}).items()
                        }
                        if self._ABLATION
                        else {},
                    }
                )
            self._policy_step += 1
            inference_started = time.monotonic()
            try:
                self._prepare_policy_observation(pano)
                if self._SUBTASK:
                    selected = (self._tasks.frames[self._actor_history_start:] if self._input_profile.history == "episode"
                                else self._tasks.selected_frames())
                    spec.prepare_context(self._tasks.context_key, selected)
                    self._tasks.record_context(selected, route, prediction_id=self._policy_step)
                advice = spec.act(
                    views=pano.get("raw_views") or {},
                    instruction=route,
                    episode_id=str(self._instruction_id()),
                    position=pose.get("position") or [0.0, 0.0, 0.0],
                    rotation_wxyz=pose.get("rotation_wxyz") or [1.0, 0.0, 0.0, 0.0],
                    is_stuck=stuck,
                    **{"frame_id": selected[-1]["id"]} if self._SUBTASK else {},
                )
                self._last_policy_inference_id = pano.get("policy_frame_id")
                self._policy_event(
                    "policy_prediction",
                    frame_id=selected[-1]["id"] if self._SUBTASK else None,
                    context_key=self._tasks.context_key if self._SUBTASK else None,
                    instruction=route, advice=advice,
                    waypoint_frame="agent_relative_forward_left_ccw",
                    waypoint_units=["metres", "metres", "radians"],
                    waypoint_count=len(advice.get("navigation") or []),
                    resampled=False,
                    inference_and_context_s=round(time.monotonic() - inference_started, 6),
                )
            except Exception as exc:
                if self._SUBTASK:
                    self._tasks.aborted = True
                    self._tasks._event("backend_error", error=str(exc))
                    raise RuntimeError("Subtask backend failed; abort this episode") from exc
                if self._ABLATION:
                    self._scheduler.fail()
                    raise RuntimeError(
                        "VLA inference failed; history state uncertain, episode must be excluded"
                    ) from exc
                return [
                    json.dumps(
                        {
                            "error": "NavGPT VLA unavailable: {}".format(exc),
                            "steps_the_vla_took": executed,
                            "advice": "NavGPT VLA is down — walk the route yourself with observe_panorama(), observe_forward() and navigate_relative()",
                        }
                    )
                ]
            arrival = bool(advice.get("should_stop"))
            velocity = advice.get("velocity")
            if self.VLA_BACKEND == "lightnav" and arrival:
                subtask_return = "decode_error" if advice.get('stop_reason') == 'decode_error' else "vla_stop"
                if self._ABLATION:
                    handoff = self._scheduler.finish(grant)
                break
            (target, rot) = (advice.get("position"), advice.get("rotation_xyzw"))
            if velocity is None and (target is None or rot is None):
                if self._ABLATION:
                    if not arrival:
                        raise RuntimeError("VLA returned neither STOP nor a complete movement")
                    handoff = self._scheduler.finish(grant)
                break
            before = pose.get("position") or [0.0, 0.0, 0.0]
            self._policy_event("policy_execution_dispatch", pose_before=pose,
                               velocity=velocity, target_position=target, target_rotation_xyzw=rot)
            if velocity is not None:
                res = self._call("{}__step_velocity".format(self.VERB_PREFIX), velocity)
            else:
                res = self._call(
                    "{}__teleport".format(self.VERB_PREFIX),
                    {
                        "position": target,
                        "rotation": rot,
                        "theta": advice.get("theta", 0.0),
                        "is_stuck": stuck,
                    },
                )
            self._policy_event("policy_execution_result",
                               feedback={k: res.get(k) for k in ("pose", "step_count", "navigable",
                                                                 "terminated", "truncated", "error")})
            stuck = pose_unchanged(prev_pose, res.get("pose"))
            prev_pose = res.get("pose")
            executed += 1
            self._steps_taken = int(res.get("step_count") or self._steps_taken)
            after = (res.get("pose") or {}).get("position") or before
            travelled += math.dist(before, after)
            if self._SUBTASK:

                def heading(p):
                    (w, x, y, z) = p.get("rotation_wxyz", [1.0, 0.0, 0.0, 0.0])
                    return math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + x * x))

                delta = heading(res.get("pose") or pose) - heading(pose)
                turned += abs(math.degrees(math.atan2(math.sin(delta), math.cos(delta))))
            if not res.get("navigable"):
                blocked_count += 1
            else:
                blocked_count = 0
            if self._ABLATION:
                terminal = bool(res.get("terminated") or res.get("truncated"))
                handoff = self._scheduler.commit(
                    grant, blocked=not res.get("navigable"), natural=terminal or arrival or i + 1 >= budget
                )
                self._scheduler.travelled = travelled
            if res.get("terminated") or res.get("truncated"):
                self._episode_over = True
                self._end_reason = "step_budget_exhausted"
                break
            if arrival or handoff == "scheduled":
                subtask_return = "vla_stop" if arrival else "scheduled"
                break
            if self._SUBTASK and (
                blocked_count >= 3
                or (not self._AGENTIC and (
                    travelled >= limits["max_distance_m"]
                    or turned >= limits["max_turn_deg"]
                ))
            ):
                subtask_return = (
                    "blocked"
                    if blocked_count >= 3
                    else "distance_budget"
                    if travelled >= limits["max_distance_m"]
                    else "turn_budget"
                )
                break
        keyframes: list[tuple[str, bytes]] = []
        if trace:
            k = max(1, min(self.MAX_KEYFRAMES, len(trace)))
            picks = [round(j * (len(trace) - 1) / max(1, k - 1)) for j in range(k)] if k > 1 else [0]
            seen = set()
            for j in picks:
                if j in seen:
                    continue
                seen.add(j)
                fr = trace[j]
                keyframes.append(
                    (
                        "Process frame {n}/{tot}: step {s}, travelled {inm:.1f} m".format(
                            n=len(keyframes) + 1, tot=k, s=fr["step"], inm=fr["travelled"]
                        ),
                        fr["png"],
                    )
                )
        self._set_trace(trace)
        self._frames_from_trace(trace)
        pano = self._pano()
        end_clear = self._clearance_summary(pano.get("scan"), pano)
        end_frames = self._pano_frames(
            pano, end_clear, "Endpoint after {} steps".format(executed)
        )
        end_pose = pano.get("pose") or {}
        net = 0.0
        if start_pose and end_pose:
            try:
                net = math.dist(
                    start_pose.get("position") or [0, 0, 0], end_pose.get("position") or [0, 0, 0]
                )
            except (TypeError, ValueError):
                net = 0.0
        content: list[Any] = []
        for label, png in keyframes:
            self._bump_obs(png, "VLA " + label.split(":")[0])
            content.append(label)
            content.append(Image(data=png, format="png"))
        for label, png in end_frames:
            parts = [x.strip() for x in (label or "").split(" — ")]
            self._bump_obs(png, "stopped: " + (parts[1] if len(parts) > 1 else parts[0] or "view"))
            content.append(label)
            content.append(Image(data=png, format="png"))
        map_fields: dict[str, Any] = {}
        if self.VLA_MAP and not self._AGENTIC:
            got = self._fetch_map()
            if got[0] is not None:
                (map_png, map_fields) = got
                self._bump_obs(map_png, "map", kind="map")
                content.append(
                    "the map after this leg — your route so far, ageing blue (start) to red (now); the red arrow is where NavGPT VLA left you, pointing the way you face"
                )
                content.append(Image(data=map_png, format="png"))
        status = {
            "instruction_given": route,
            "steps_the_vla_took": executed,
            "step_budget_for_this_call": budget,
            "distance_travelled_m": round(travelled, 2),
            "net_displacement_m": round(net, 2),
            "blocked_events": blocked_count,
            "vla_suggests_arrival": arrival,
            "arrival_note": "This means NavGPT VLA stopped producing new movement — NOT that you have arrived. It is unreliable about arrival, in both directions. Check the four views against the instruction yourself before you STOP.",
            "steps_taken_total": self._steps_taken,
            "steps_remaining_approx": max(0, self.STEP_BUDGET - self._steps_taken),
            "episode_over": self._episode_over,
            "reminder": "the episode is still running — only your own stop ends it",
        }
        if self._SUBTASK:
            status["subtask_return"] = subtask_return
            status["turn_travelled_deg"] = round(turned, 2)
        if self._ABLATION:
            status["handoff"] = handoff or "natural"
            status["review"] = self._scheduler.snapshot()
            if self._scheduler.delegation == 1 and self._vla_calls == 1:
                canonical = {
                    "trace": [{"pose": f["pose"], "source": f["source_sha256"]} for f in trace],
                    "end_pose": end_pose,
                    "vla_state": spec.audit_state(),
                    "nodes": [
                        {k: n.get(k) for k in ("id", "position", "xy", "visits", "links", "route")}
                        for n in (self._fetch_places() or {}).get("places", [])
                    ],
                    "travelled": travelled,
                }
                status["initial_rollout"] = hashlib.sha256(
                    json.dumps(canonical, sort_keys=True).encode()
                ).hexdigest()
            self._live_log({"review": status["review"], "handoff": status["handoff"]})
        if map_fields:
            for key in (
                "caption",
                "direction_cues_px",
                "places_px",
                "presentation_transform",
                "absolute_heading_deg",
                "up_on_the_map_is",
                "frame",
                "facing",
                "you_are_at_px",
                "started_at_px",
                "straight_line_from_start_m",
                "revisiting_earlier_position",
                "seen_area_m2",
                "legend",
            ):
                if key in map_fields:
                    status[key] = map_fields[key]
        elif self._HAS_MOVE and self._last_revisit is not None:
            status["revisiting_earlier_position"] = self._last_revisit
        if self._episode_over:
            status["end_reason"] = self._end_reason
        content.append(json.dumps(status))
        if self._ABLATION:
            places = self._fetch_places()
            if places is None:
                raise RuntimeError("Required observed graph unavailable")
            content.append(self._format_places(places))
        self._live_log(
            {"navigate_by_instruction": route, "steps": executed, "arrival": arrival, "map": bool(map_fields)}
        )
        self._frame_here()
        return content

    def go(self, target: Any) -> list:
        if self._AGENTIC:
            return self._return_memory_node(target)
        """Walk back to a recorded place, or to a waypoint of the last VLA rollout
        ("wp N"). See _GO_DESC."""
        self._tool_enter("navigate_to_node")
        if self._episode_over:
            return ["episode already over ({}); no more moves".format(self._end_reason)]
        text = str(target).strip()
        m_wp = re.match("^wp\\s*(\\d+)$", text, flags=re.I)
        if m_wp:
            return self.go_back_to_waypoint(int(m_wp.group(1)))
        if text and (not text.lstrip("-").isdigit()) and (len(text) <= 2) and text.isalpha():
            return [
                "there are no lettered spots on this run — travel with navigate_relative(turn_deg, distance_m), reading the bearing off the ruler under whatever you want to walk towards. navigate_to_node() returns to a numbered place you recorded, or one by name."
            ]
        return self._go_place(target)

    def add_place(self, name: str, caption: str = "", place: Any = None, clause: Any = None) -> list:
        """Attach the agent's own name and caption to a place. See _ADD_PLACE_DESC."""
        self._tool_enter("annotate_node")
        if not str(name).strip():
            return ['a place needs a name — something you will recognise later, like "kitchen doorway"']
        payload = self._fetch_places()
        if not payload or not payload.get("places"):
            return ["no places yet — they appear as you travel"]
        target = None
        if place is None:
            target = self._label_target(payload)
        elif isinstance(place, (list, tuple)) and len(place) == 2:
            (best, best_d) = (None, None)
            for pl in payload["places"]:
                d = math.dist(pl["xy"], [float(place[0]), float(place[1])])
                if best_d is None or d < best_d:
                    (best, best_d) = (pl["id"], d)
            target = best
        else:
            try:
                target = int(place)
            except (TypeError, ValueError):
                return ["place must be a number, [x, y], or omitted for where you are"]
        known = [pl["id"] for pl in payload["places"]]
        if target not in known:
            return ["no place {} — places are {}".format(target, known)]
        anchor = None
        try:
            anchor = (self._call("{}__agent_state".format(self.VERB_PREFIX), {}) or {}).get("position")
        except Exception:
            anchor = (self._last_pose or {}).get("position")
        clause_n = None
        if clause is not None:
            try:
                clause_n = int(clause)
            except (TypeError, ValueError):
                clause_n = None
        replaced = self._place_labels.get(int(target))
        self._place_labels[int(target)] = {
            "name": str(name)[:60],
            "caption": str(caption)[:140],
            "clause": clause_n,
            "anchor": list(anchor) if anchor else None,
            "frame": self._recent_frames[-1] if self._recent_frames else None,
        }
        self._dump_labels()
        pl = next((p for p in payload["places"] if p["id"] == target))
        status = {
            "recorded_place": target,
            "at_xy": pl["xy"],
            "distance_from_you_m": pl["straight_m"],
            "note": "recorded where you are standing"
            if pl["straight_m"] < 0.75
            else "the nearest place is {:.1f} m from you; your caption was attached to it, and going back will return you to where you stand now".format(
                pl["straight_m"]
            ),
            **({"clause": clause_n} if clause_n is not None else {}),
            **({"replaced_the_name": replaced.get("name")} if replaced else {}),
        }
        self._live_log({"annotate_node": target, "name": str(name)[:60], "clause": clause_n})
        return [json.dumps(status), self._format_places(payload)]

    def _go_place(self, place: Any) -> list:
        """Retrace to a numbered/named place along ground already walked."""
        if self._episode_over:
            return ["episode already over ({}); cannot move".format(self._end_reason)]
        payload = self._fetch_places()
        if not payload or not payload.get("places"):
            return ["no places yet — they appear as you travel"]
        known = [pl["id"] for pl in payload["places"]]
        inputs: dict[str, Any] = {}
        if place is None:
            if self._AGENTIC:
                return ["navigate_to_node requires a node ID from observe_map"]
            return [
                "navigate_to_node needs a place: a number ({}), a name you gave with annotate_node, or [x, y] in metres from your start".format(
                    ", ".join((str(i) for i in known))
                )
            ]
        if isinstance(place, (list, tuple)):
            if len(place) != 2:
                return ["coordinates must be [x, y] in metres from your start"]
            try:
                inputs["xy"] = [float(place[0]), float(place[1])]
            except (TypeError, ValueError):
                return ["coordinates must be two numbers, [x, y]"]
        else:
            target = None
            try:
                target = int(place)
            except (TypeError, ValueError):
                want = str(place).strip().lower()
                hits = [
                    pid for (pid, lab) in self._place_labels.items() if lab["name"].strip().lower() == want
                ]
                if not hits:
                    return [
                        'no place named "{}" — names so far: {}'.format(
                            place,
                            ", ".join(sorted((l["name"] for l in self._place_labels.values())))
                            or "(none yet)",
                        )
                    ]
                by_id = {pl["id"]: pl for pl in payload["places"]}
                target = min(hits, key=lambda pid: by_id[pid]["straight_m"] if pid in by_id else 1000000000.0)
            if target not in known:
                return [
                    "no place {} — the places you have are {}".format(
                        target, ", ".join((str(i) for i in known))
                    )
                ]
            inputs["place"] = target
        try:
            res = self._call("{}__retrace_to".format(self.VERB_PREFIX), inputs)
        except Exception as exc:
            return ["cannot go there ({})".format(exc)]
        self._steps_taken = int(res.get("step_count") or self._steps_taken)
        if res.get("terminated") or res.get("truncated"):
            self._episode_over = True
            self._end_reason = "step_budget_exhausted"
        approached = None
        label = self._place_labels.get(res.get("place"))
        goal_pos = (label or {}).get("anchor")
        if goal_pos is None:
            goal_pos = next(
                (pl.get("position") for pl in payload["places"] if pl["id"] == res.get("place")), None
            )
        if goal_pos and (not self._episode_over):
            pose = (res.get("pose") or {}).get("position")
            if pose:
                dx = float(goal_pos[0]) - float(pose[0])
                dz = float(goal_pos[2]) - float(pose[2])
                gap = math.hypot(dx, dz)
                if gap > 0.4:
                    heading = float((res.get("pose") or {}).get("heading_rad") or 0.0)
                    bearing = math.atan2(-dx, -dz) - heading
                    bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
                    out = self._call(
                        "{}__step_hightolow".format(self.VERB_PREFIX),
                        {"angle_rad": bearing, "distance_m": gap},
                    )
                    self._steps_taken = int(out.get("step_count") or self._steps_taken)
                    approached = round(float(out.get("walked_m") or 0.0), 2)
                    if out.get("terminated") or out.get("truncated"):
                        self._episode_over = True
                        self._end_reason = "step_budget_exhausted"
        pano = self._pano()
        clear = self._clearance_summary(pano.get("scan"), pano)
        content: list[Any] = []
        content += self._pano_content(pano, clear, "back at place {}".format(res.get("place")))
        got = self._fetch_map()
        if got[0] is not None:
            self._bump_obs(got[0], "map", kind="map")
            content.append("the map, with your route so far and every place on it")
            content.append(Image(data=got[0], format="png"))
        status = {
            "went_to_place": res.get("place"),
            "place_xy": res.get("place_xy"),
            "arrived": res.get("arrived"),
            "route": res.get("route"),
            "retraced_m": res.get("retraced_m"),
            "steps_charged": res.get("steps_charged"),
            "steps_taken_total": self._steps_taken,
            "steps_remaining_approx": max(0, self.STEP_BUDGET - self._steps_taken),
            "episode_over": self._episode_over,
        }
        if approached is not None:
            status["final_approach_m"] = approached
        if res.get("stopped_short_m") is not None:
            status["stopped_short_m"] = res["stopped_short_m"]
        if self._episode_over:
            status["end_reason"] = self._end_reason
        content.append(json.dumps(status))
        after = self._fetch_places()
        if after:
            content.append(self._format_places(after))
        self._live_log({"navigate_to_node": res.get("place"), "steps_charged": res.get("steps_charged")})
        return content

    def _set_trace(self, trace):
        """Keep the last rollout so the agent can return to a waypoint it saw."""
        self._trace = list(trace or [])
        k = max(1, min(self.MAX_KEYFRAMES, len(self._trace)))
        self._trace_keys = (
            [round(j * (len(self._trace) - 1) / max(1, k - 1)) for j in range(k)] if k > 1 else [0]
        )

    def go_back_to_waypoint(self, n: Any) -> list:
        """Retrace to waypoint n of the last VLA rollout (reached via navigate_to_node("wp N"))."""
        m_wp = re.match("^wp\\s*(\\d+)$", str(n).strip(), flags=re.I)
        if m_wp:
            n = int(m_wp.group(1))
        if self._episode_over:
            return ["episode already over ({}); cannot move".format(self._end_reason)]
        if not self._trace or not self._trace_keys:
            return ["no route recorded yet — call navigate_by_instruction first"]
        try:
            idx = int(n)
        except (TypeError, ValueError):
            return ["n must be a waypoint number, 1..{}".format(len(self._trace_keys))]
        if not 1 <= idx <= len(self._trace_keys):
            return ["waypoint {} does not exist; there are {}".format(idx, len(self._trace_keys))]
        target = self._trace[self._trace_keys[idx - 1]]
        pose = target["pose"] or {}
        if not pose.get("position"):
            return ["waypoint {} has no recorded pose".format(idx)]
        try:
            res = self._call(
                "{}__retrace_path".format(self.VERB_PREFIX),
                {"position": list(pose["position"]), "rotation_wxyz": pose.get("rotation_wxyz")},
            )
        except Exception as exc:
            return ["cannot return to waypoint {} ({})".format(idx, exc)]
        self._steps_taken = int(res.get("step_count") or self._steps_taken)
        if res.get("terminated") or res.get("truncated"):
            self._episode_over = True
            self._end_reason = "step_budget_exhausted"
        out = self._pano()
        clear = self._clearance_summary(out.get("scan"), out)
        content: list[Any] = []
        content += self._pano_content(out, clear, "at waypoint {}".format(idx))
        status = {
            "returned_to_waypoint": idx,
            "arrived": res.get("arrived"),
            "retraced_m": res.get("retraced_m"),
            "steps_charged": res.get("steps_charged"),
            "steps_taken_total": self._steps_taken,
            "steps_remaining_approx": max(0, self.STEP_BUDGET - self._steps_taken),
            "episode_over": self._episode_over,
        }
        content.append(json.dumps(status))
        self._live_log({"navigate_to_node": idx, **status})
        return content

    def arrive(self, where: Any = "here") -> list:
        """Terminal action for the place-carrying conditions: stop, having named where.

        Profiles without a node graph can only stop where they stand. For
        `planner_memory`/`planner_vla_memory`, ending the episode is a CHOICE BETWEEN
        CANDIDATES rather than a reflex from wherever the robot happens to stand: the
        agent may name a recorded place to retrace to before stopping.

        Telemetry is deliberately identical to a primitive STOP: the same `steps_taken_total`
        key the runner's EventSink keys on, the same `end_reason`, so a run using
        terminate_episode() is scored by exactly the same path as one using action 0.
        """
        self._tool_enter("terminate_episode")
        if self._episode_over:
            return ["episode already over ({}); nothing left to do".format(self._end_reason)]
        content: list[Any] = []
        text = str(where).strip()
        here = text.lower() in ("", "here", "now", "this", "this spot", "current")
        if not here and (not self._HAS_PLACES):
            return [
                'this profile has no node graph, so there is nowhere to retrace to — terminate_episode("here") stops where you stand'
            ]
        if not here:
            got = self._go_place(where)
            content += got
            blob = " ".join((str(x) for x in got if isinstance(x, str)))
            if self._episode_over:
                return content
            if '"went_to_place"' not in blob or '"arrived": false' in blob:
                content.append(
                    'did NOT stop: the retrace to {!r} did not arrive, so the episode is still running. Fix the target or call terminate_episode("here").'.format(
                        text
                    )
                )
                return content
        outputs = self._call("{}__step_discrete".format(self.VERB_PREFIX), {"action": 0})
        self._steps_taken += 1
        self._episode_over = True
        self._end_reason = "stop_called"
        status = {
            "stopped_at": "here" if here else text,
            "actions_executed": [0],
            "steps_taken_total": self._steps_taken,
            "steps_remaining_approx": max(0, self.STEP_BUDGET - self._steps_taken),
            "episode_over": True,
            "end_reason": self._end_reason,
            "terminated": bool(outputs.get("terminated")),
        }
        content.append(json.dumps(status))
        self._live_log({"arrive": status["stopped_at"], "steps": self._steps_taken})
        return content

    def _candidates_block(self, limit: int = 12) -> str | None:
        """EVERY place, on the tail of every look, in route order.

        Otherwise the place table is reachable only by calling observe_map(), so at the
        moment the navigator decides where to stop the graph would be off screen. This
        puts it where the decision is made.

        ALL of them, not a shortlist: the right stopping place is typically mid-route,
        neither the newest nor the closest, so a top-3 shortlist would often hide it.
        Graphs are small, so listing all of them costs a few short lines.

        Route order (by id) rather than by distance, because ids are chronological: the
        list then reads as the journey, which is the frame the instruction is written in.
        Numbers lead, because retrieval is mostly by number.
        """
        if self._AGENTIC:
            return None
        payload = self._fetch_places()
        if not payload:
            return None
        places = payload.get("places") or []
        if not places:
            return None
        (nearest, nearest_d) = (None, None)
        for pl in places:
            d = pl.get("straight_m")
            if d is not None and (nearest_d is None or d < nearest_d):
                (nearest, nearest_d) = (pl["id"], d)
        out = []
        for pl in sorted(places, key=lambda pl: pl["id"])[:limit]:
            lab = self._place_labels.get(pl["id"]) or {}
            name = lab.get("name") or "unnamed"
            cost = pl.get("route_m")
            out.append(
                "  {}{}: {}{} — {:.1f} m away{}".format(
                    pl["id"],
                    " (nearest)" if pl["id"] == nearest else "",
                    name,
                    "" if lab.get("clause") is None else " [clause {}]".format(lab["clause"]),
                    pl.get("straight_m", 0.0),
                    ", {:.0f} steps to return".format(cost / 0.25) if cost else "",
                )
            )
        extra = (
            "" if len(places) <= limit else "\n  ... and {} more, all on the map".format(len(places) - limit)
        )
        return (
            ("visited map nodes; navigate_to_node(id) returns there; terminate_episode stops here:\n"
             if self._AGENTIC else
             "places you have recorded, in the order you reached them — terminate_episode(n) goes back to one and stops there:\n")
            + "\n".join(out)
            + extra
        )

    def recall(self, place: Any) -> list:
        if self._AGENTIC:
            if type(place) is not int or place not in self._tasks.memory.nodes:
                raise ValueError("Use a node ID printed on observe_map")
            refs = self._tasks.memory.nodes[place]["observation_refs"]
            return self._memory_images(refs[-1:]) or ["No image recorded at this node"]
        """The frame a place was recorded at. On demand, never in a listing."""
        self._tool_enter("observe_node")
        try:
            pid = int(place)
        except (TypeError, ValueError):
            return [
                "observe_node takes a place NUMBER, e.g. observe_node(3) — {!r} is not one. The numbers are in the place list on every look.".format(
                    place
                )
            ]
        lab = self._place_labels.get(pid) or {}
        path = lab.get("frame") or self._place_frames.get(pid)
        if not path or not frame_exists(path):
            known = sorted(
                (
                    k
                    for k in set(self._place_labels) | set(self._place_frames)
                    if (self._place_labels.get(k) or {}).get("frame") or self._place_frames.get(k)
                )
            )
            return [
                "place {} has no recorded picture{}".format(
                    pid,
                    "; pictures exist for {}".format(known)
                    if known
                    else " — none of the places have one yet",
                )
            ]
        try:
            png = read_frame(path)
        except OSError as exc:
            return ["could not read the frame for place {}: {}".format(pid, exc)]
        return [
            "place {} — {}: {}".format(pid, lab.get("name") or "unnamed", lab.get("caption") or ""),
            Image(data=png, format="png"),
        ]

    def _instruction_id(self) -> str:
        """Episode id for NavGPT VLA's per-episode bookkeeping. It only labels
        the server's debug GIF, so a stable per-process value is enough."""
        return self.env.get("NAVGPT_EPISODE_ID", "episode")

    def _bump_obs(self, png: bytes, label: str | None = None, kind: str = "view") -> None:
        self._obs_count += 1
        self._live_frame(png, label, kind)

    @staticmethod
    def _compact_motion_result(result):
        # Keep decision-relevant execution facts; raw backend responses remain in the trace.
        # subtask_return already records vla_stop. Do not present the actor's
        # stop signal again under an arrival label: it cannot verify a goal.
        allowed = {"steps_the_vla_took", "step_budget_for_this_call", "distance_travelled_m",
                   "net_displacement_m", "blocked_events", "steps_taken_total",
                   "steps_remaining_approx", "episode_over", "subtask_return", "turn_travelled_deg",
                   "turned_deg", "requested_m", "walked_m", "blocked", "detour_m", "detoured",
                   "nearest_obstacle_before_m", "end_reason", "target_node_id", "termination_reason",
                   "arrived", "step_count", "terminated", "truncated"}
        compact = []
        for item in result:
            if isinstance(item, str) and item.startswith("{"):
                data = json.loads(item)
                compact.append(json.dumps({k: v for k, v in data.items() if k in allowed}))
            else:
                compact.append(item)
        return compact

    def _tool_state(self):
        state = self._tasks.snapshot()
        if self._AGENTIC:
            read = self._memory_map_read
            state["map_read_valid"] = bool(read and read["motion_epoch"] == self._motion_epoch
                                            and read["pending_edge_id"] == self._tasks.memory.pending)
            state["steps_remaining"] = max(0, self.STEP_BUDGET - self._steps_taken)
        return state

    def _register(self, name, description, fn):
        if self._enabled_tools is not None and name not in self._enabled_tools:
            return
        if self._AGENTIC:
            description = _sibling("agentic", "TOOL_DESCRIPTIONS").get(name, description)
        if self.VLA_BACKEND == "lightnav":
            description = description.replace("NavGPT VLA", "LightNav-0")
        if self._SUBTASK:
            import functools

            @functools.wraps(fn)
            def task_call(*args, **kwargs):
                with self._scheduler.lock:
                    before_motion = self._motion_epoch
                    segment = self._AGENTIC and name in ("execute_subtask", "navigate_relative", "navigate_to_node")
                    begun = False
                    try:
                        if segment:
                            if self._episode_over:
                                raise ValueError("Episode is already over")
                            pose = self._last_pose or (self._tasks.frames[-1]["pose"] if self._tasks.frames else {})
                            task_id = kwargs.get("task_id", self._tasks.active_id)
                            task = next((t for t in self._tasks.tasks if t["id"] == task_id), None)
                            self._tasks.memory.begin(pose, self._tasks.frames[-1]["id"] if self._tasks.frames else None,
                                                     name, task, kwargs)
                            begun = True
                            self._memory_map_read = None
                        if not self._AGENTIC and name in ("navigate_relative", "navigate_to_node"):
                            self._tasks.begin()
                        result = fn(*args, **kwargs)
                        if segment:
                            reason = "tool_return"
                            for item in result:
                                if isinstance(item, str) and item.startswith("{"):
                                    status = json.loads(item)
                                    reason = status.get("subtask_return", status.get("end_reason", status.get("termination_reason", reason)))
                                    if status.get("blocked"):
                                        reason = "blocked"
                            pose = self._last_pose
                            if not pose:
                                raise RuntimeError("Execution ended without a confirmed pose")
                            frame = self._tasks.frames[-1] if self._tasks.frames else None
                            frame_id = (frame["id"] if frame and frame["motion_version"] == self._tasks.motion_version
                                        and frame["pose"].get("position") == pose.get("position") else None)
                            self._tasks.memory.finish(pose, frame_id, reason)
                            self._tasks._event("execution_memory", **self._tasks.memory.status())
                            result = self._compact_motion_result(result)
                            result.append(json.dumps({"execution": self._tasks.memory.status()}))
                            result.extend(self._segment_map())
                    except (ValueError, MotionRejected) as exc:
                        if begun:
                            if self._motion_epoch != before_motion:
                                self._tasks.memory.uncertain(str(exc))
                            else:
                                self._tasks.memory.cancel()
                        if self._motion_epoch != before_motion:
                            raise RuntimeError("Tool failed after dispatching movement; retain the lease") from exc
                        if self._AGENTIC:
                            return ToolResult(isError=True, content=[
                                TextContent(text=json.dumps({"error": "precondition_failed", "message": str(exc), "state_changed": False})),
                                TextContent(text=json.dumps({"subtask_state": self._tool_state()})),
                            ])
                        result = [json.dumps({"error": "precondition_failed", "message": str(exc), "state_changed": False})]
                    except Exception as exc:
                        if begun:
                            self._tasks.memory.uncertain(str(exc))
                        raise
                    result.append(json.dumps({"subtask_state": self._tool_state()}))
                    return result

            self.tools.register(name, description, task_call)
            return
        if self._ABLATION and name not in self._CAPABILITIES[self.BUNDLE]:
            return
        if not self._ABLATION:
            self.tools.register(name, description, fn)
            return
        import functools

        if name == "terminate_episode" and self.BUNDLE == "core":

            def stop_here() -> list:
                return self.arrive("here")

            fn = stop_here
            description = "End this episode at the current position. This operation cannot return to a node."

        @functools.wraps(fn)
        def serial(*args, **kwargs):
            with self._scheduler.lock:
                correction = name in ("navigate_relative", "navigate_to_node", "terminate_episode")
                if correction:
                    self._scheduler.correction()
                try:
                    if self._scheduler.failed and name != "terminate_episode":
                        raise RuntimeError("Ablation episode invalid; terminate and exclude it")
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    self._scheduler.fail()
                    return [json.dumps({"ablation_error": str(exc), "review": self._scheduler.snapshot()})]
                if self.INTERFACE == "text_labels":
                    paired = []
                    for item in result:
                        if isinstance(item, Image) and item.data:
                            fields = self._presentation_by_hash.get(hashlib.sha256(item.data).hexdigest())
                            if fields:
                                paired.append(json.dumps(fields))
                        paired.append(item)
                    result = paired
                result.append(json.dumps({"review": self._scheduler.snapshot()}))
                return result

        if name == "navigate_by_instruction":
            description = "Execute a navigation instruction using the VLA. Pass the full original route first. For later corrections, use current observations to give a self-contained remaining route with supported landmarks/directions and the original stopping condition; omit completed clauses. At a scheduled review, repeat the active instruction (or omit it) only when it still fits. Pass a revised instruction explicitly when correction is needed. Visual history is retained. Natural stop/cap returns give the Planner endpoint authority. Review timing is runtime controlled."
        elif name in ("observe_forward", "observe_panorama"):
            description = "Inspect current camera evidence. Labels state the active bearing convention. Observation does not move the robot."
        elif name == "observe_map":
            description = (
                "Inspect the observed route map, stable node IDs and its stated coordinate convention."
            )
        self.tools.register(name, description, serial)

    def update_plan(self, subtasks: list[dict], reason: str, reopen_ids: list[int] | None = None) -> list:
        """Submit 1-8 local goals, each with id, source_quote, instruction and completion_criterion.

        Use exact source quotes, stable IDs and a reason. Explicitly reopen previously
        accepted tasks (and dependent later tasks) when new evidence invalidates them.
        """
        self._tasks.update_plan(subtasks, reason, reopen_ids)
        return []

    def execute_subtask(self) -> list:
        """Execute ONLY the active local goal. Returns fresh views/map; review before another move."""
        self._tasks.begin()
        result = self.vla_navigate(self._tasks.policy_instruction(self._input_profile.include_memory))
        self._tasks.returned("execution_handoff")
        return result

    def review_subtask(self, outcome: str, evidence_refs: list[str], summary: str, reason: str) -> list:
        """Review local completion using current_frame_id AND current_map_id.

        outcome: accepted, continue, or replan. Supply a <=800-character summary
        of completed work and relevant uncertainties, excluding future route clauses.
        """
        self._tasks.review(outcome, evidence_refs, summary, reason)
        return []

    def finish_subtask_episode(self, status: str = "complete") -> list:
        """STOP here. complete requires all tasks accepted; abort records giving up."""
        self._tasks.finish(status)
        return self.arrive("here")

    def update_agentic_plan(self, reason: str, subtasks: list[PlanItem] | None = None,
                            patch: list[PlanPatch] | None = None) -> list:
        """Revise the plan and briefly explain why. Pass exactly one of subtasks or patch.

        subtasks replaces the full plan; preserve IDs of retained tasks. patch only changes
        the status of existing tasks by ID, e.g. marking one completed.
        IDs are distinct positive integers. At most one task is in_progress.
        Reopen completed tasks here before executing them again. Does not move.
        """
        if (subtasks is None) == (patch is None):
            raise ValueError("Pass exactly one of subtasks (full plan) or patch (status changes)")
        if patch is not None:
            self._tasks.patch_plan(patch, reason)
        else:
            self._tasks.update_plan(subtasks, reason)
        return [json.dumps({"plan_updated": True, "revision": self._tasks.revision})]

    def execute_agentic_task(self, task_id: int, instruction: str = "",
                             reset_context: bool = False) -> list:
        """Execute a task by ID. Empty instruction uses the plan; otherwise override this call only.

        reset_context restarts visual history for this task.
        Runs until VLA stop, blockage or runtime limits. Returns process frames, endpoint
        views and movement statistics. The episode remains active unless episode_over is true.
        """
        if reset_context and not self._input_profile.manual_reset:
            raise ValueError('This policy retains episode history; task-local reset is unsupported')
        history_frames = int(self.env.get("NAVGPT_HISTORY_FRAMES", "16"))
        boundary_frames = int(self.env.get("NAVGPT_BOUNDARY_FRAMES", "2"))
        if not 1 <= history_frames <= 16 or not 0 <= boundary_frames <= min(4, history_frames - 1):
            raise ValueError('Invalid actor history budget')
        self._tasks.begin(task_id, instruction, reset_context)
        boundary = self._tasks.completed_boundary
        if (self._actor_history_policy == "completed_subtask" and boundary
                and boundary['task_id'] != task_id):
            start = boundary['frame_index']
            if start > self._actor_history_start:
                # Deliberate contiguous segment reset, never a sparse rolling window.
                # Reset errors propagate: never silently continue an uncertain history.
                self._lazy_vla().reset()
                self._tasks._event(
                    'actor_history_crop', reason='completed_subtask',
                    completed_task_id=boundary['task_id'], next_task_id=task_id,
                    start_frame_id=boundary['frame_id'],
                    discarded_frames=start - self._actor_history_start,
                    retained_frames=len(self._tasks.frames) - start,
                )
                self._actor_history_start = start
            self._tasks.completed_boundary = None
        self._memory_map_read = None
        self._tasks.max_frames, self._tasks.boundary_frames = history_frames, boundary_frames
        self._tasks._event('execution_budget', scope='runtime_and_episode',
                          remaining_actions=max(0, self.STEP_BUDGET - self._steps_taken),
                          runtime_max_actions=self.VLA_MAX_STEPS or None,
                          history_mode=self._input_profile.history,
                          history_policy=self._actor_history_policy,
                          history_frames=history_frames if self._input_profile.history == "bounded" else None,
                          boundary_frames=boundary_frames if self._input_profile.history == "bounded" else 0)
        result = self.vla_navigate(self._tasks.policy_instruction(self._input_profile.include_memory))
        self._tasks.returned('execution_handoff')
        return result

    def execute_agentic_task_without_reset(self, task_id: int, instruction: str = "") -> list:
        """Schema adapter for the reset ablation; execution has one implementation."""
        return self.execute_agentic_task(task_id, instruction)

    def update_memory(self, start_node_id: int, end_node_id: int, caption: str) -> list:
        """Write a 1-800 character caption for the current traversal.

        Use the map returned by the latest movement or observe_map, read in an earlier
        response; read_memory does not satisfy this requirement. start_node_id and
        end_node_id must match the pending segment's IDs printed on that map.
        Describe observed progress, discoveries, failure or return reasons and uncertainty.
        Repeating this call revises only this segment; the next movement closes it.
        """
        self._tool_enter("update_memory")
        read = self._memory_map_read
        if not read or read["motion_epoch"] != self._motion_epoch:
            raise ValueError("Call observe_map before update_memory: no map has been returned since the latest movement")
        if any(n not in read["node_ids"] for n in (start_node_id, end_node_id)):
            raise ValueError("Both node IDs must appear on the last explicitly observed map")
        if read["pending_edge_id"] != self._tasks.memory.pending:
            raise ValueError("Read observe_map for the current execution segment first")
        evidence = {key: value for key, value in read.items() if key != "node_ids"}
        entry = self._tasks.remember(start_node_id, end_node_id, caption, evidence)
        self._live_log({"update_memory": [start_node_id, end_node_id],
                        "memory_revision": self._tasks.memory.revision})
        return [json.dumps({"updated_memory": {"start_node_id": start_node_id, "end_node_id": end_node_id,
                                               "revision": self._tasks.memory.revision}})]

    def read_memory(self, node_id: int, other_node_id: int | None = None,
                    images: Literal["none", "endpoint", "trajectory"] = "none",
                    views: list[Literal["front", "right", "back", "left"]] | None = None,
                    offset: int = 0) -> list:
        """Read up to five experiences, newest first. Does not move.

        node_id selects incoming/outgoing history; other_node_id filters node_id -> other_node_id.
        Pass returned next_offset as offset for older records. images=none returns text;
        endpoint adds the latest node visit or newest matching endpoint; trajectory adds
        up to four process frames. views applies only to endpoint (default front).
        Missing saved views are reported.
        """
        if views is not None and images != "endpoint":
            raise ValueError("views is only valid with images=endpoint")
        self._validate_views(views if views is not None else ["front"])
        self._tool_enter("read_memory")
        page = self._tasks.memory.read(node_id, other_node_id, offset)
        evidence = []
        if images == "trajectory":
            candidates = [(r["sequence"], ref) for r in page["records"] for ref in r["frame_refs"]]
            for sequence, ref in self._tasks.memory.sample(candidates, 4):
                evidence.extend([f"Historical execution {sequence}, {ref}", *self._memory_images([ref])])
        elif images == "endpoint":
            sequence = None
            if other_node_id is None:
                refs = self._tasks.memory.nodes[node_id]["observation_refs"]
                ref = refs[-1] if refs else None
            elif page["records"]:
                sequence = page["records"][0]["sequence"]
                edge = self._tasks.memory.edges[sequence - 1]
                refs = self._tasks.memory.nodes[other_node_id]["observation_refs"]
                ref = next((r for r in reversed(edge["frame_refs"]) if r in refs), None)
            else:
                ref = None
            frame = next((f for f in self._tasks.frames if f["id"] == ref), None)
            selected = views if views is not None else ["front"]
            saved = frame.get("planner_views", {}) if frame else {}
            page["observation"] = {"historical": True, "frame_id": ref, "sequence": sequence,
                                   "orientation": (frame["pose"] if frame else {}).get("rotation_wxyz"),
                                   "missing_views": [v for v in selected if not saved.get(v)]}
            for direction in selected:
                if saved.get(direction):
                    evidence.extend([f"Historical {ref}: {direction}",
                                     Image(data=base64.b64decode(saved[direction]), format="png")])
        return [json.dumps(page, ensure_ascii=False), *evidence]

    def _memory_images(self, refs):
        images = []
        for ref in refs:
            frame = next((f for f in self._tasks.frames if f["id"] == ref), None)
            if frame:
                views = frame["views"]
                value = frame.get("planner_views", {}).get("front") or views.get("front")
                if value:
                    images.append(Image(data=base64.b64decode(value), format="png"))
        return images

    def _return_memory_node(self, node_id):
        self._tool_enter("navigate_to_node")
        if type(node_id) is not int or node_id not in self._tasks.memory.nodes:
            raise ValueError("navigate_to_node requires a node ID printed on observe_map")
        node = self._tasks.memory.nodes[node_id]
        rotation = node.get("rotation_wxyz")
        if (not isinstance(rotation, (list, tuple)) or len(rotation) != 4
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in rotation)
                or sum(v * v for v in rotation) < 1e-12):
            raise ValueError("The target node has no valid recorded orientation")
        result = self._call(self.VERB_PREFIX + "__retrace_path",
                            {"position": node["position"], "rotation_wxyz": list(rotation),
                             "match_3d": True})
        if result.get("error"):
            raise RuntimeError(result["error"])
        self._steps_taken = int(result.get("step_count", self._steps_taken))
        if result.get("truncated") or result.get("terminated"):
            self._episode_over, self._end_reason = True, "step_budget_exhausted"
        pano = self._pano()
        result["target_node_id"] = node_id
        result["termination_reason"] = "returned" if result.get("arrived") else "stopped_short"
        return [json.dumps(result), *self._pano_content(pano)]

    def finish_agentic_episode(self, status: Literal["complete", "abort"], reason: str) -> list:
        """Permanently STOP at the current pose. Declare complete or abort and briefly explain why.
        """
        if not reason.strip():
            raise ValueError("Explain why the episode should end")
        already_over = self._episode_over
        stopped = self.arrive('here')
        if already_over or self._end_reason != "stop_called":
            return stopped  # STOP was not executed, so there is no completion to declare.
        # Declare only after the environment confirmed STOP.
        self._tasks.finish(status)
        self._tasks._event('completion_reason', reason=reason)
        return [*stopped, json.dumps({"declared_status": status, "reason": reason,
                                      "node_id": self._tasks.memory.current_node_id})]
