"""Extra experiment: GA with lower mutation rates (suggested by the sensitivity study)."""
import json
import numpy as np
from .config import GAConfig, Weights
from .experiments import _setup
from .ga import GeneticAlgorithm


def run_tuned(out="results", rates=(0.02, 0.05), seeds=5, gens=150, pop=80):
    env, ev = _setup(Weights())
    res = {}
    for r in rates:
        fin, gens_run, div = [], [], []
        for s in range(seeds):
            h = GeneticAlgorithm(env, ev, GAConfig(generations=gens, pop_size=pop,
                                                   mutation_rate=r, seed=s)).run()
            fin.append(h["hof"][-1]); gens_run.append(h["generations_run"]); div.append(h["diversity"][-1])
            m = h["hof_genome"].metrics
            print(f"mut={r} seed={s} fit={fin[-1]:.4f} gens={gens_run[-1]} cov={m['coverage']:.3f} "
                  f"det={m['detection']:.3f} E={m['energy_norm']:.3f} T={m['time_norm']:.3f} "
                  f"H={m['hazard_norm']:.3f}", flush=True)
        res[str(r)] = dict(final=fin, mean=float(np.mean(fin)), std=float(np.std(fin)),
                           gens=gens_run, div=float(np.mean(div)))
    json.dump(res, open(f"{out}/tuned.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    run_tuned()