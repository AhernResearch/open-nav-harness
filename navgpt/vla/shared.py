"""Backend factories: shared weights, private histories, no simulator dependency."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor


class SessionEngine:
    """Use only the policy's private ViT cache; never reset the shared engine."""

    def __init__(self, engine):
        self.engine = engine

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def reset_episode_state(self):
        pass


class BackendFactory:
    def __init__(self, config):
        self.config = config
        self.backend = config["backend"]
        self.executor = None
        self.running = 0
        self.waiting = 0

    async def start(self):
        from .owned import LightNavEngine, NativeEngine, habitat_tracking_agent

        self.loop = asyncio.get_running_loop()
        options = dict(self.config.get("model_options", {}))
        if self.backend == "lightnav":
            options.setdefault("max_num_seqs", self.config.get("max_batch_size", 4))
            from lightnav.serving.tracking_service import BatchedTrackingService

            base = await asyncio.to_thread(LightNavEngine, self.config["model_path"], **options)
            from .lightnav import LightNavBridge

            self.metadata = LightNavBridge(inference=base, **self.config.get("control", {})).info()
            agent = base.agent
            bundle = agent.engine.bundle
            self.metadata['native_input'] = {
                'num_history_frames': agent.num_history_frames,
                'slowfast_tiers': bundle.slowfast_tiers,
                'video_size': list(bundle.video_size),
                'video_fps': bundle.video_fps,
                'max_new_tokens': agent.engine.max_new_tokens,
                'history_buffer': 'full_episode',
            }

            class CancellableTrackingService(BatchedTrackingService):
                def make_session(self, client_id=None):
                    return habitat_tracking_agent(
                        self.engine, centroids=self.centroids, rvq_bundle=self.rvq_bundle,
                        num_history_frames=self.num_history_frames, client_id=client_id,
                    )

                def _infer_batch(self, requests):
                    cancelled = [getattr(r.session, "_onh_cancelled", False) for r in requests]
                    active = [r for r, skip in zip(requests, cancelled) if not skip]
                    results = iter(super()._infer_batch(active) if active else [])
                    return [
                        RuntimeError("Cancelled before GPU execution") if skip else next(results)
                        for skip in cancelled
                    ]

            self.service = CancellableTrackingService(
                engine=agent.engine,
                bundle=agent.engine.bundle,
                centroids=agent.centroids,
                rvq_bundle=agent.rvq,
                num_history_frames=agent.num_history_frames,
                max_new_tokens=agent.engine.max_new_tokens,
                max_batch_size=self.config.get("max_batch_size", 4),
                max_wait_ms=self.config.get("max_wait_ms", 8),
                serve_task="vln",
            )
            await self.service.start()
        elif self.backend == "navgpt":
            # One thread owns all native processor/model work, including CUDA state.
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="native-vla")
            self.base = await self.run(lambda: NativeEngine(self.config["model_path"], **options))
            self.metadata = self.base.info()
        else:
            raise ValueError("Unsupported backend")

    def make(self):
        from .owned import LightNavEngine, NativeEngine

        if self.backend == "navgpt":
            return NativeEngine(
                self.config["model_path"],
                shared_model=(self.base.agent.model, self.base.agent.processor),
                **self.config.get("model_options", {}),
            )
        from .lightnav import LightNavBridge

        agent = self.service.make_session()
        agent.engine = SessionEngine(agent.engine)

        def predict(session):
            return asyncio.run_coroutine_threadsafe(self.service.predict(session), self.loop).result()

        return LightNavBridge(
            inference=LightNavEngine(agent=agent, predict=predict), **self.config.get("control", {})
        )

    def stats(self):
        queued = (
            self.service.scheduler._queue.qsize()
            if self.backend == "lightnav" and hasattr(self, "service")
            else 0
        )
        return {"waiting_requests": self.waiting + queued, "active_operations": self.running}

    async def run(self, fn):
        self.waiting += 1

        def invoke():
            self.waiting -= 1
            self.running += 1
            try:
                return fn()
            finally:
                self.running -= 1

        return await asyncio.get_running_loop().run_in_executor(self.executor, invoke)

    def cancel(self, actor):
        if self.backend == "lightnav":
            actor.inference.agent._onh_cancelled = True

    async def close(self):
        if self.backend == "lightnav":
            await self.service.stop()
        if self.executor is not None:
            self.executor.shutdown(wait=True)
