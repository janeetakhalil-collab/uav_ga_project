"""Selection, crossover and mutation operators tailored to spatial genomes."""
import numpy as np


def tournament(fits, k, rng):
    """Tournament selection; k is the selection pressure (k=1 -> random)."""
    idx = rng.integers(0, len(fits), size=k)
    return int(idx[np.argmax(fits[idx])])


def _swap(c1, c2, d, sl):
    for f in ("wp", "speed", "power", "active"):
        a, b = getattr(c1, f), getattr(c2, f)
        tmp = a[d, sl].copy()
        a[d, sl] = b[d, sl]
        b[d, sl] = tmp


def crossover(a, b, rng):
    """Randomly applies one of three spatial crossovers:
       0 route-swap        : exchange whole drone routes between parents
       1 segment-swap      : two-point swap of a waypoint sub-sequence per drone
       2 geometric (blend) : convex combination of coordinates, uniform for bits
    """
    c1, c2 = a.copy(), b.copy()
    D, K = a.D, a.K
    op = rng.integers(0, 3)
    if op == 0:
        for d in range(D):
            if rng.random() < 0.5:
                _swap(c1, c2, d, slice(0, K))
                c1.deploy[d], c2.deploy[d] = b.deploy[d].copy(), a.deploy[d].copy()
    elif op == 1:
        for d in range(D):
            i, j = sorted(rng.integers(0, K + 1, 2))
            if i == j:
                j = min(K, j + 1)
            _swap(c1, c2, d, slice(i, j))
            if rng.random() < 0.5:
                c1.deploy[d], c2.deploy[d] = b.deploy[d].copy(), a.deploy[d].copy()
    else:
        al = rng.random((D, 1, 1))
        c1.wp = al * a.wp + (1 - al) * b.wp
        c2.wp = (1 - al) * a.wp + al * b.wp
        c1.deploy = al[:, 0] * a.deploy + (1 - al[:, 0]) * b.deploy
        c2.deploy = (1 - al[:, 0]) * a.deploy + al[:, 0] * b.deploy
        al2 = rng.random((D, K))
        c1.speed = al2 * a.speed + (1 - al2) * b.speed
        c2.speed = (1 - al2) * a.speed + al2 * b.speed
        c1.power = al2 * a.power + (1 - al2) * b.power
        c2.power = (1 - al2) * a.power + al2 * b.power
        mask = rng.random((D, K)) < 0.5
        c1.active = np.where(mask, a.active, b.active)
        c2.active = np.where(mask, b.active, a.active)
    return c1, c2


def mutate(ch, rng, pm, sigma, env):
    """Spatial mutation suite (in place).
       - Gaussian waypoint / deployment shift (step size sigma, annealed)
       - jump mutation: waypoint re-sampled uniformly OR from the priority map
       - waypoint swap and sub-route reversal (2-opt style)
       - speed / power Gaussian perturbation, sensor-schedule bit-flip
    """
    D, K, n = ch.D, ch.K, env.n
    m = rng.random((D, K)) < pm
    ch.wp[m] += rng.normal(0, sigma, (m.sum(), 2))
    m = rng.random((D, K)) < pm * 0.25
    k = int(m.sum())
    if k:
        half = rng.random(k) < 0.5
        new = np.where(half[:, None], rng.uniform(0, n - 1, (k, 2)), env.sample_priority_cells(rng, k))
        ch.wp[m] = new
    m = rng.random(D) < pm
    ch.deploy[m] += rng.normal(0, sigma, (m.sum(), 2))
    for d in range(D):
        if rng.random() < pm:
            i, j = rng.choice(K, 2, replace=False)
            ch.wp[d, [i, j]] = ch.wp[d, [j, i]]
        if rng.random() < pm * 0.5:
            i, j = sorted(rng.choice(K, 2, replace=False))
            ch.wp[d, i:j + 1] = ch.wp[d, i:j + 1][::-1].copy()
    m = rng.random((D, K)) < pm
    ch.speed[m] += rng.normal(0, 0.25, m.sum())
    m = rng.random((D, K)) < pm
    ch.power[m] += rng.normal(0, 0.15, m.sum())
    m = rng.random((D, K)) < pm * 0.5
    ch.active[m] = ~ch.active[m]
    return ch