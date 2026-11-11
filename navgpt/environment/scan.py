"""Habitat-GS scene adapter using the existing environment execution and tools.

The first scan integration is explicitly development-only. There are no goal
coordinates, scoring references or manufactured navigation metrics.
"""

import hashlib

from .r2r import R2RCEEnv
from .scan_assets import asset_path, load_scan_package, sha256


class ScanEnv(R2RCEEnv):
    def __init__(self, scene_manifest, *, config=None, blind=False):
        self.manifest_path, self.manifest, self.navigation, self.scan_episodes = load_scan_package(scene_manifest)
        self.package_root = self.manifest_path.parent
        self.manifest_sha256 = sha256(self.manifest_path)
        self.stage_file = str(asset_path(self.package_root, self.manifest["assets"]["stage"]["path"]))
        self.navmesh_file = str(asset_path(self.package_root / "navigation",
                                           self.navigation["assets"]["navmesh"]["path"]))
        super().__init__(scene_root=str(self.package_root),
                         episodes_file=str(asset_path(self.package_root / "navigation",
                             self.navigation["assets"]["episodes"]["path"])),
                         dataset="scan", split="development", config=config, blind=blind)

    def _load_episode_set(self, **options):
        if options["split"] != "development":
            raise ValueError("Scan integration currently supports only the development split")
        return {"episodes": list(self.scan_episodes), "ground_truth": {}, "dataset": "scan",
                "split": "development", "languages": [], "roles": [],
                "episode_count": len(self.scan_episodes), "gt_covered": 0,
                "episode_sources": [self.episodes_file], "gt_sources": [],
                "dataset_fingerprint": hashlib.sha256((self.manifest_sha256 + sha256(
                    self.package_root / "navigation" / "receipt.json")).encode()).hexdigest()}

    def _scene_file(self, ep):
        return self.stage_file

    def _navmesh_file(self, scene_path):
        return self.navmesh_file

    def _configure_simulator(self, sim_cfg):
        import habitat_sim

        if not hasattr(sim_cfg, "gaussian_time") or not habitat_sim.cuda_enabled:
            raise RuntimeError("Scan rendering requires the pinned CUDA-enabled Habitat-GS build")
        sim_cfg.enable_physics = False
        sim_cfg.navmesh_settings = None

    def _meta(self):
        result = super()._meta()
        result["geodesic_distance"] = None
        return result

    def _metrics(self):
        return {"evaluation_available": False, "purpose": "development",
                "success": None, "spl": None, "ndtw": None, "distance_to_goal": None,
                "steps_taken": self._step_index, "stop_called": self._stop_called,
                "collisions": self._collisions}

    def health(self):
        result = super().health()
        result.update(backend="habitat_gs", purpose="development", evaluation_available=False,
                      geometry_review="required", depth_source="gaussian_unvalidated",
                      scene_manifest_sha256=self.manifest_sha256,
                      habitat_gs_commit=self.manifest["upstream"]["commit"])
        return result
