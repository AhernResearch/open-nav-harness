"""Owned actor engines, imported inside their dedicated CUDA environment only.

LightNav reuses its pinned inference library and action decoder. Neither its
WebSocket server nor a second SDK scheduler is started by this worker.
"""

from __future__ import annotations

import base64
import io


class LightNavEngine:
    def __init__(self, model_path, gpu_memory_utilization=0.35):
        from lightnav.tracking import build_tracking_agent

        self.agent = build_tracking_agent(
            model_path,
            backend="vllm_local",
            task_key="vlnce",
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def __call__(self, action, data):
        if action == "reset":
            self.agent.reset("")
            return {"rc": 0}
        if action != "next":
            raise ValueError("Unknown LightNav operation")
        import numpy as np
        from PIL import Image
        from lightnav.serving.protocol import decode_prediction_signals

        raw = base64.b64decode(data["image"], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            frame = np.asarray(image.convert("RGB")).copy()
        self.agent.observe(frame)
        instruction = data.get("instruction", "")
        if not instruction:
            return {"rc": 0}
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
            import random
            import numpy as np
            import torch

            seed = int(data["seed"])
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
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
        return [action, navigation]

    def close(self):
        pass
