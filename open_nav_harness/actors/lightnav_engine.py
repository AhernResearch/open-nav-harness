"""The LightNav-0 model behind the actor service. Runs only in the LightNav CUDA environment.

Nothing in the control plane imports this module; it needs lightnav, torch and vLLM.
Usage: python -m open_nav_harness.actors.lightnav_engine --config CONFIG.json
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

import numpy as np
import uvicorn
from lightnav.eval_config import load_eval_config
from lightnav.inference.config import InferenceConfig
from lightnav.inference.engine import build_engine
from lightnav.serving.tracking_service import BatchedTrackingService
from lightnav.tracking import TrackingAgent, load_centroids, resolve_action_decoder_from_config
from lightnav.traj_vocab import load_rvq_bundle
from PIL import Image

from .lightnav import Control, LightNavService, Prediction, create_app

# The official Habitat runner's generation budget; a shorter cap would truncate the
# grounding prefix before the action tokens.
MAX_NEW_TOKENS = 64
POLICY_CAMERA = {"width": 480, "height": 270, "hfov": 120.0, "height_m": 0.88, "pitch_down_rad": 0.0}


class HabitatTrackingAgent(TrackingAgent):
    """Official Habitat decode-failure semantics: only a trajectory decode ValueError is
    tolerated, as zero waypoints with the error recorded. Everything else still raises."""

    decode_error: str | None = None

    def decode_waypoints(self, text):
        self.decode_error = None
        try:
            return super().decode_waypoints(text)
        except ValueError as error:
            self.decode_error = str(error)
            return np.zeros((self.H, 3), dtype=np.float32), text


class SessionAgent(HabitatTrackingAgent):
    """A session's private frame buffer and ViT cache over the shared engine."""


class LightNavBackend:
    def __init__(self, model_path: str, gpu_memory_utilization=0.3, max_num_seqs=4, max_batch_size=4,
                 max_wait_ms=8):
        self.model_path = model_path
        self.options = {"gpu_memory_utilization": gpu_memory_utilization, "max_num_seqs": max_num_seqs,
                        "max_batch_size": max_batch_size, "max_wait_ms": max_wait_ms}
        self.camera = POLICY_CAMERA
        self.identity: dict = {}
        # One thread owns frame conversion; inference runs on the batching scheduler.
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lightnav-frames")

    async def start(self):
        await asyncio.get_running_loop().run_in_executor(self.worker, self._load)
        await self.service.start()

    def _load(self):
        decoder = resolve_action_decoder_from_config(self.model_path, "vlnce")
        if decoder is None:
            raise ValueError("the checkpoint must provide its action decoder")
        max_tokens, decoding = MAX_NEW_TOKENS, {}
        if decoder["method"] == "rvq":
            rvq = load_rvq_bundle(decoder["bundle_path"], decoder["horizon"], num_frames=0,
                                  load_cluster_ids=False)
            max_tokens = max(max_tokens, len(rvq.levels) + 2)
            decoding["rvq_bundle"] = rvq
        else:
            decoding["centroids"] = load_centroids(decoder["traj_vocab_path"], decoder["K"],
                                                   decoder["horizon"])
        config = InferenceConfig(model_path=self.model_path, backend="vllm_local",
                                 max_new_tokens=max_tokens,
                                 gpu_memory_utilization=self.options["gpu_memory_utilization"],
                                 max_num_seqs=self.options["max_num_seqs"])
        tasks = (load_eval_config(self.model_path) or {}).get("tasks") or {}
        task = "vlnce" if not tasks or "vlnce" in tasks else next(iter(tasks))
        engine, bundle = build_engine(config, task_type=task, max_new_tokens=max_tokens)
        history = int(bundle.num_history_frames)
        self.service = BatchedTrackingService(
            engine=engine, bundle=bundle, centroids=decoding.get("centroids"),
            num_history_frames=history, max_new_tokens=engine.max_new_tokens,
            max_batch_size=self.options["max_batch_size"], max_wait_ms=self.options["max_wait_ms"],
            serve_task="vln", rvq_bundle=decoding.get("rvq_bundle"))
        self.decoding, self.history = decoding, history
        self.identity = {
            "name": "lightnav-0", "model_path": self.model_path, "task": task,
            "lightnav": metadata.version("lightnav"), "action_decoder": decoder["method"],
            "num_history_frames": history, "slowfast_tiers": len(bundle.slowfast_tiers or []),
            "video_size": list(bundle.video_size), "video_fps": bundle.video_fps,
            "max_new_tokens": engine.max_new_tokens, "history_buffer": "full_episode", **self.options}

    async def stop(self):
        await self.service.stop()
        self.worker.shutdown(wait=True)

    def open(self) -> SessionAgent:
        return SessionAgent(self.service.engine, centroids=self.service.centroids,
                            rvq_bundle=self.service.rvq_bundle, num_history_frames=self.history)

    async def append(self, handle: SessionAgent, png: bytes):
        def observe():
            with Image.open(io.BytesIO(png)) as image:
                handle.observe(np.asarray(image.convert("RGB")).copy())

        await asyncio.get_running_loop().run_in_executor(self.worker, observe)

    async def predict(self, handle: SessionAgent, instruction: str) -> Prediction:
        handle.instruction = instruction
        result = await self.service.predict(handle)
        return Prediction(result.waypoints.tolist(), bool(result.stop), result.raw_text,
                          handle.decode_error, dict(result.timings_ms))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())
    backend = LightNavBackend(config["model_path"], **config.get("model", {}))
    service = LightNavService(backend, Control(**config.get("control", {})),
                              config.get("max_sessions", 8))
    uvicorn.run(create_app(service), host=config.get("host", "127.0.0.1"), port=config["port"])


if __name__ == "__main__":
    main()
