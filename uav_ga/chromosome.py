"""Hybrid real/binary chromosome for a UAV fleet.

For every drone d (D drones) and K segments the genome holds
    deploy[d]        (x, y) real       -> deployment (launch) position
    wp[d, k]         (x, y) real       -> k-th waypoint
    speed[d, k]      real in [vmin,vmax] -> cruise speed on the segment ending at wp[d,k]
    power[d, k]      real in [0, 1]    -> sensor power profile (sets sensor radius & quality)
    active[d, k]     binary            -> sensor ON/OFF schedule on that segment
Total genes = D*(2 + 5K).
"""
import numpy as np


class Chromosome:
    def __init__(self, deploy, wp, speed, power, active):
        self.deploy, self.wp = deploy, wp
        self.speed, self.power, self.active = speed, power, active
        self.metrics = None
        self.fitness = -np.inf

    @property
    def D(self):
        return self.wp.shape[0]

    @property
    def K(self):
        return self.wp.shape[1]

    def path_points(self):
        """(D, K+1, 2): deployment point followed by the K waypoints."""
        return np.concatenate([self.deploy[:, None, :], self.wp], axis=1)

    def copy(self):
        c = Chromosome(self.deploy.copy(), self.wp.copy(), self.speed.copy(),
                       self.power.copy(), self.active.copy())
        c.metrics, c.fitness = self.metrics, self.fitness
        return c

    @staticmethod
    def random(env, rng, D, K, walk_fraction=0.5):
        """Random individual. A share of the routes are random walks (smooth), the
        rest are uniform-random point sets -> structurally diverse initial pool."""
        n, u = env.n, env.uav
        deploy = rng.uniform(0, n - 1, (D, 2))
        wp = np.empty((D, K, 2))
        for d in range(D):
            if rng.random() < walk_fraction:
                steps = rng.normal(0, 9, (K, 2))
                wp[d] = deploy[d] + np.cumsum(steps, axis=0)
            else:
                wp[d] = rng.uniform(0, n - 1, (K, 2))
        speed = rng.uniform(0.7, 1.5, (D, K))
        power = rng.uniform(0.2, 1.0, (D, K))
        active = rng.random((D, K)) < 0.8
        return Chromosome(deploy, wp, speed, power, active)