"""Hard-constraint handling: repair (bounds, no-fly waypoints, range) -- the
remaining violations (segments crossing no-fly cells, energy overrun, residual
range) are handled by the dynamic penalty in fitness.py."""
import numpy as np


def _push_out_of_nofly(arr, env):
    """Move any point lying in a no-fly cell to the nearest free cell (in place)."""
    ij = np.clip(np.rint(arr).astype(int), 0, env.n - 1)
    bad = env.nofly[ij[..., 0], ij[..., 1]]
    if bad.any():
        bi = ij[bad]
        arr[bad] = env.nearest_free[:, bi[:, 0], bi[:, 1]].T.astype(float)


def repair(ch, env):
    u, n = env.uav, env.n
    ch.deploy[:] = np.clip(ch.deploy, 0, n - 1)
    ch.wp[:] = np.clip(ch.wp, 0, n - 1)
    ch.speed[:] = np.clip(ch.speed, u.v_min, u.v_max)
    ch.power[:] = np.clip(ch.power, 0.0, 1.0)
    _push_out_of_nofly(ch.deploy, env)
    _push_out_of_nofly(ch.wp, env)
    # range repair: contract an over-long route towards its deployment point
    P = ch.path_points()
    length = np.linalg.norm(np.diff(P, axis=1), axis=2).sum(1)
    for d in np.where(length > u.max_range)[0]:
        s = 0.98 * u.max_range / length[d]
        ch.wp[d] = ch.deploy[d] + s * (ch.wp[d] - ch.deploy[d])
    if (length > u.max_range).any():
        _push_out_of_nofly(ch.wp, env)
    return ch