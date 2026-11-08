"""Owned actor engines, imported inside their dedicated CUDA environment only.

LightNav reuses its pinned inference library and action decoder. Neither its
WebSocket server nor a second SDK scheduler is started by this worker.
"""

from __future__ import annotations

import base64
import io


def habitat_tracking_agent(engine, **options):
    """Official Habitat decode-failure semantics, also for batched inference.

    Only a trajectory decode ValueError becomes a zero-velocity STOP, as in
    TrajVocabVLNCEPolicy.act. Transport, inference and input failures still raise.
    """
    import numpy as np
    from lightnav.tracking import TrackingAgent

    class HabitatTrackingAgent(TrackingAgent):
        def decode_waypoints(self, text):
            self.decode_error = None
            try:
                return super().decode_waypoints(text)
            except ValueError as exc:
                self.decode_error = str(exc)
                return np.zeros((self.H, 3), dtype=np.float32), text

    return HabitatTrackingAgent(engine, **options)


class LightNavEngine:
    def __init__(self, model_path=None, gpu_memory_utilization=0.35, max_num_seqs=1, *, agent=None, predict=None):
        self.predict = predict
        if agent is not None:
            self.agent = agent
            return
        from lightnav.tracking import resolve_action_decoder_from_config, load_centroids
        from lightnav.eval_config import load_eval_config
        from lightnav.inference.config import InferenceConfig
        from lightnav.inference.engine import build_engine

        decoder = resolve_action_decoder_from_config(model_path, "vlnce")
        if decoder is None:
            raise ValueError("Checkpoint must provide its action decoder")
        # Match the official Habitat runner's generation budget. A shorter cap
        # must not truncate grounding prefixes before the action tokens.
        max_tokens = 64
        if decoder["method"] == "rvq":
            from lightnav.traj_vocab import load_rvq_bundle
            rvq = load_rvq_bundle(decoder["bundle_path"], decoder["horizon"], num_frames=0, load_cluster_ids=False)
            max_tokens = max(max_tokens, len(rvq.levels) + 2)
            decoding = {"rvq_bundle": rvq}
        else:
            decoding = {"centroids": load_centroids(decoder["traj_vocab_path"], decoder["K"], decoder["horizon"])}
        config = InferenceConfig(model_path=model_path, backend="vllm_local", max_new_tokens=max_tokens,
                                 gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs)
        tasks = (load_eval_config(model_path) or {}).get("tasks") or {}
        task = "vlnce" if not tasks or "vlnce" in tasks else next(iter(tasks))
        engine, bundle = build_engine(config, task_type=task, max_new_tokens=max_tokens)
        self.agent = habitat_tracking_agent(engine, num_history_frames=int(bundle.num_history_frames), **decoding)

    def __call__(self, action, data):
        if action == "reset":
            self.agent.reset("")
            return {"rc": 0}
        if action not in ("next", "predict"):
            raise ValueError("Unknown LightNav operation")
        import numpy as np
        from PIL import Image
        from lightnav.serving.protocol import decode_prediction_signals

        if action == "next":
            raw = base64.b64decode(data["image"], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                frame = np.asarray(image.convert("RGB")).copy()
            self.agent.observe(frame)
        instruction = data.get("instruction", "")
        if not instruction:
            return {"rc": 0}
        if self.predict is not None:
            self.agent.instruction = instruction
            prediction = self.predict(self.agent)
            return {
                "rc": 0, "actions": {"actions": prediction.waypoints.tolist()},
                "stop": prediction.stop, "raw_text": prediction.raw_text,
                "latency_ms": sum(prediction.timings_ms.get(k, 0) for k in ("vit_ms", "llm_ms")),
                "timings_ms": prediction.timings_ms,
                "decode_error": getattr(self.agent, "decode_error", None),
            }
        waypoints, text, latency = self.agent.predict_waypoints(instruction, task_type="vlnce_traj")
        signals = decode_prediction_signals(
            text,
            vocab_size=getattr(self.agent, "K", None),
            is_rvq=self.agent.rvq is not None,
            waypoints=waypoints,
        )
        return {
            "rc": 0,
            "actions": {"actions": waypoints.tolist()},
            "stop": signals.stop,
            "raw_text": text,
            "latency_ms": latency,
            "decode_error": getattr(self.agent, "decode_error", None),
        }


class NativeEngine:
    """Native action-head model; intentionally not presented as a chat model."""

    def __init__(self, model_path, **options):
        from .agent import NavGPTVLA
        from .api import NavGPTVLAService
        from .context import AgentContext
        from .images import decode_base64_to_image

        self.decode = decode_base64_to_image
        self.agent = NavGPTVLA(model_path=model_path, view_num=4, **options)
        self.service = NavGPTVLAService(self.agent)
        self.context = AgentContext(self.agent, self.decode)
        self.model_path = model_path

    def info(self):
        return {
            "model_name": "navgpt-vla",
            "backend": "navgpt",
            "model_version": self.model_path,
            "context_replace": True,
            "max_context_groups": 16,
            "history_observe": True,
            "seeded_reset": True,
            "execution": "owned_pytorch",
        }

    def reset(self, data):
        self.service.reset()
        self.context.reset()
        self.agent.audit_history = data.get("seed") is not None
        if self.agent.audit_history:
            seed = int(data["seed"])
            self.agent.rng.seed(seed)
        return {"status": "reset complete"}

    def state(self):
        return {"context": self.context.state(), "history_frames": len(self.agent.rgb_lists[0])}

    def replace_context(self, data):
        return self.context.replace(data)

    def observe(self, data):
        self.agent.observe_history([self.decode(data["rgb_" + cam]) for cam in self.context.cameras])
        return {"status": "observed", "frames": len(self.agent.rgb_lists[0])}

    def act(self, data):
        import numpy as np
        import quaternion

        if "frame_id" in data:
            self.context.validate_act(data)
        images = [self.decode(data["rgb_" + cam]) for cam in self.context.cameras]
        action, navigation = self.service.act(
            *images,
            data["text"],
            data["global_gps"],
            quaternion.from_float_array(np.array(data["global_compass"])),
            data["is_stuck"],
        )
        if "frame_id" in data:
            self.context.commit_act(data)
        action["inference_metrics"] = self.agent.last_timings
        return [action, navigation]

    def close(self):
        pass
