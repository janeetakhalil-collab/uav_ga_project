"""Baseline: pure random search with the same evaluation budget."""
import numpy as np
from .chromosome import Chromosome
from .constraints import repair


def random_search(env, evaluator, n_evals, batch, seed):
    rng = np.random.default_rng(seed)
    D, K = env.cfg.n_drones, env.cfg.n_waypoints
    best, best_val, curve, evals = None, -np.inf, [], []
    for i in range(n_evals):
        ch = evaluator.evaluate(repair(Chromosome.random(env, rng, D, K), env))
        if ch.metrics["feasible"]:
            b = evaluator.base(ch.metrics)
            if b > best_val:
                best, best_val = ch, b
        if (i + 1) % batch == 0:
            curve.append(float(best_val) if np.isfinite(best_val) else float("nan"))
            evals.append(i + 1)
    return dict(hof=curve, evals=evals, hof_genome=best)