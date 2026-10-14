"""Multi-objective fitness (weighted-sum scalarisation) with dynamic penalties.

   F = wC*Cov + wD*Det - wE*E_norm - wT*T_norm - wH*H_norm - lambda(g)*V
   lambda(g) = lambda0 * (1 + growth * g/G)      (penalty grows during the run)
"""
import numpy as np


class Evaluator:
    def __init__(self, env, weights, lambda0=2.0, growth=3.0):
        self.env, self.uav, self.w = env, env.uav, weights
        self.lambda0, self.growth = lambda0, growth

    # -------------------------------------------------------------- simulation
    def simulate(self, ch, counts=False):
        env, u = self.env, self.uav
        n, S = env.n, u.samples_per_segment
        D, K = ch.D, ch.K
        P = ch.path_points()
        A, B = P[:, :-1], P[:, 1:]
        seg = B - A
        L = np.linalg.norm(seg, axis=2)                                   # (D,K)
        v, pw, act = ch.speed, ch.power, ch.active
        tsec = L / v
        # propulsion (grows with v^2) + sensor energy (grows with power^2 * time)
        energy = L * (0.5 + 0.5 * v ** 2) + 2.0 * pw ** 2 * act * tsec
        t = np.linspace(0, 1, S)
        pts = A[:, :, None, :] + seg[:, :, None, :] * t[None, None, :, None]   # (D,K,S,2)
        ij = np.clip(np.rint(pts).astype(int), 0, n - 1)
        haz = env.hazard[ij[..., 0], ij[..., 1]]
        hazard = float((haz * (tsec / (S - 1))[:, :, None]).sum())            # time-weighted
        nf = env.nofly[ij[..., 0], ij[..., 1]]
        nofly_len = float((nf * (L / (S - 1))[:, :, None]).sum())

        # ---- sensing: footprint stamping + detection
        rad_f = u.r_min + (u.r_max - u.r_min) * pw
        rad_i = np.rint(rad_f).astype(int)
        shp = ij.shape[:3]
        flat_ij = ij.reshape(-1, 2)
        flat_r = np.broadcast_to(rad_i[:, :, None], shp).ravel()
        flat_a = np.broadcast_to(act[:, :, None], shp).ravel()
        grid = np.zeros((n, n), int if counts else bool)
        for r, off in env.disk.items():
            sel = flat_a & (flat_r == r)
            if not sel.any():
                continue
            cells = flat_ij[sel][:, None, :] + off[None]
            ok = (cells >= 0).all(-1) & (cells < n).all(-1)      # drop out-of-grid cells
            cx, cy = cells[..., 0][ok], cells[..., 1][ok]
            if counts:
                np.add.at(grid, (cx, cy), 1)
            else:
                grid[cx, cy] = True
        covered = grid > 0
        coverage = float(env.priority[covered].sum() / env.priority.sum())

        q = np.minimum(1.0, (0.4 + 0.6 * pw) * 1.2 / v)       # slower + stronger => better detection
        flat_p = pts.reshape(-1, 2)
        flat_rf = np.broadcast_to(rad_f[:, :, None], shp).ravel()
        flat_q = np.broadcast_to(q[:, :, None], shp).ravel()
        d = np.linalg.norm(env.targets[:, None, :] - flat_p[None], axis=2)
        hit = (d <= flat_rf[None]) & flat_a[None]
        detection = float(np.where(hit, flat_q[None], 0.0).max(axis=1).mean())

        Ld, Ed = L.sum(1), energy.sum(1)
        range_v = float(np.maximum(0, Ld - u.max_range).sum() / u.max_range)
        energy_v = float(np.maximum(0, Ed - u.battery).sum() / u.battery)
        violation = range_v + energy_v + 10.0 * nofly_len / u.max_range
        m = dict(coverage=coverage, detection=detection,
                 energy=float(Ed.sum()), energy_norm=float(Ed.sum() / (D * u.battery)),
                 time=float(tsec.sum(1).max()), time_norm=float(tsec.sum(1).max() / u.t_max),
                 hazard=hazard, hazard_norm=hazard / u.h_ref,
                 length=float(Ld.sum()), nofly_len=nofly_len,
                 range_violation=range_v, energy_violation=energy_v,
                 violation=violation, feasible=bool(violation < 1e-9))
        if counts:
            m["cov_counts"] = grid
        return m

    def evaluate(self, ch):
        ch.metrics = self.simulate(ch)
        return ch

    # ------------------------------------------------------------------ scores
    def base(self, m):
        w = self.w
        return (w.coverage * m["coverage"] + w.detection * m["detection"]
                - w.energy * m["energy_norm"] - w.time * m["time_norm"]
                - w.hazard * m["hazard_norm"])

    def score(self, m, gfrac):
        lam = self.lambda0 * (1.0 + self.growth * gfrac)
        return self.base(m) - lam * m["violation"]