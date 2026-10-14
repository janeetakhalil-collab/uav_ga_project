"""Discretised hazard grid: priority map, hazard map, no-fly zones, targets.

All maps are indexed [x, y].
"""
import numpy as np
from scipy import ndimage
from .config import EnvConfig, UAVConfig


class Environment:
    def __init__(self, cfg: EnvConfig = None, uav: UAVConfig = None):
        self.cfg = cfg or EnvConfig()
        self.uav = uav or UAVConfig()
        n = self.n = self.cfg.size
        rng = np.random.default_rng(self.cfg.seed)
        X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

        def g(cx, cy, s):
            return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))

        # Disaster-zone priority (0..1): four hot-spots on a low background
        pri = 0.05 + g(12, 38, 6) + 0.9 * g(38, 36, 5) + 0.8 * g(30, 10, 6) + 0.6 * g(10, 10, 4)
        self.priority = pri / pri.max()
        # Hazard pockets (wind / radiation), 0..1 severity
        self.hazard = np.clip(0.9 * g(22, 30, 5) + 0.9 * g(36, 22, 6) + 0.7 * g(10, 22, 4)
                              + 0.8 * g(43, 32, 4) + 0.6 * g(44, 6, 4), 0, 1)
        # Hard no-fly rectangles (x0, x1, y0, y1)
        self.nofly = np.zeros((n, n), bool)
        for x0, x1, y0, y1 in [(20, 27, 16, 22), (3, 8, 28, 33), (40, 46, 12, 17)]:
            self.nofly[x0:x1 + 1, y0:y1 + 1] = True
        # Hidden targets: sampled proportionally to priority^2, never inside no-fly
        p = (self.priority ** 2) * (~self.nofly)
        p = p / p.sum()
        flat = rng.choice(n * n, size=self.cfg.n_targets, replace=False, p=p.ravel())
        self.targets = np.stack(np.unravel_index(flat, (n, n)), 1).astype(float)
        # nearest free cell for every cell (used by the repair operator)
        _, self.nearest_free = ndimage.distance_transform_edt(self.nofly, return_indices=True)
        # integer disk offsets for sensor footprints
        self.disk = {}
        for r in range(int(np.floor(self.uav.r_min)), int(np.ceil(self.uav.r_max)) + 1):
            dx, dy = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
            m = dx ** 2 + dy ** 2 <= r * r
            self.disk[r] = np.stack([dx[m], dy[m]], 1)
        self._pflat = self.priority.ravel() / self.priority.sum()

    def sample_priority_cells(self, rng, k):
        """k random cell coordinates drawn proportionally to disaster priority."""
        idx = rng.choice(self.n * self.n, size=k, p=self._pflat)
        return np.stack(np.unravel_index(idx, (self.n, self.n)), 1).astype(float)