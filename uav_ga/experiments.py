"""Experiment drivers: main run, baseline comparison, sensitivity, weight profiles."""
import json, time, itertools
from dataclasses import replace
import numpy as np

from .config import EnvConfig, UAVConfig, Weights, GAConfig
from .environment import Environment
from .fitness import Evaluator
from .ga import GeneticAlgorithm
from .baseline import random_search
from . import visualize as viz


def _setup(w):
    env = Environment(EnvConfig(), UAVConfig())
    return env, Evaluator(env, w)


def jaccard(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


PROFILES = {
    "Balanced": Weights(0.35, 0.25, 0.15, 0.10, 0.15),
    "Coverage-first": Weights(0.50, 0.30, 0.05, 0.05, 0.10),
    "Energy-saver": Weights(0.25, 0.15, 0.35, 0.15, 0.10),
    "Safety-first": Weights(0.25, 0.15, 0.10, 0.10, 0.40),
}


def run_all(out="results", gens=150, pop=80, seeds=5, sens_seeds=3, sens_gens=80, log=print):
    out_res = {}
    env, ev = _setup(Weights())
    viz.plot_environment(env, f"{out}/00_environment.png")
    base_cfg = GAConfig(generations=gens, pop_size=pop)

    # ---------------- 1. GA runs (seeds) + random search baseline, equal budgets
    t0 = time.time(); ga_hs, rs_rs = [], []
    for s in range(seeds):
        h = GeneticAlgorithm(env, ev, replace(base_cfg, seed=s)).run(); ga_hs.append(h)
        budget = h["evals"][-1]
        rs_rs.append(random_search(env, ev, budget, pop, 100 + s))
        log(f"[main] seed {s}: GA {h['hof'][-1]:.4f} ({h['generations_run']} gens, {budget} evals) | RS {rs_rs[-1]['hof'][-1]:.4f}")
    h0 = ga_hs[0]
    viz.plot_convergence(h0, f"{out}/01_convergence.png")
    viz.plot_evolution(h0, env, ev, f"{out}/02_evolution_paths_coverage.png")
    viz.plot_final(h0["hof_genome"], env, ev, f"{out}/03_final_solution.png")
    viz.plot_baseline(ga_hs, rs_rs, f"{out}/04_ga_vs_random.png")
    best_rs = max(rs_rs, key=lambda r: r["hof"][-1])["hof_genome"]
    viz.plot_uniqueness(ga_hs, best_rs, env, f"{out}/05_uniqueness_across_seeds.png")

    keys = ["coverage", "detection", "energy_norm", "time_norm", "hazard_norm", "length"]
    tab = {}
    for name, hs in (("GA", [x["hof_genome"].metrics for x in ga_hs]),
                     ("Random", [r["hof_genome"].metrics for r in rs_rs])):
        tab[name] = {"fitness": [ev.base(m) for m in hs], **{k: [m[k] for m in hs] for k in keys},
                     "feasible": [m["feasible"] for m in hs]}
    covs = [ev.simulate(x["hof_genome"], counts=True)["cov_counts"] > 0 for x in ga_hs]
    rcovs = [ev.simulate(r["hof_genome"], counts=True)["cov_counts"] > 0 for r in rs_rs]
    pj = [jaccard(a, b) for a, b in itertools.combinations(covs, 2)]
    pjr = [jaccard(a, b) for a, b in itertools.combinations(rcovs, 2)]
    out_res["main"] = dict(table=tab, ga_pairwise_jaccard=pj, rs_pairwise_jaccard=pjr,
                           ga_gens_run=[x["generations_run"] for x in ga_hs],
                           ga_evals=[x["evals"][-1] for x in ga_hs],
                           final_diversity=[x["diversity"][-1] for x in ga_hs],
                           init_diversity=[x["diversity"][0] for x in ga_hs],
                           best_metrics={k: v for k, v in h0["hof_genome"].metrics.items() if k != "cov_counts"})
    log(f"[main] done in {time.time()-t0:.0f}s")

    # ---------------- 2. sensitivity analysis
    grids = {"pop_size": [20, 40, 80], "mutation_rate": [0.02, 0.12, 0.30], "tournament_k": [2, 4, 8]}
    sens, stab = {}, {}
    for p, vals in grids.items():
        sens[p], stab[p] = {}, {}
        for v in vals:
            hs = []
            for s in range(sens_seeds):
                cfg = replace(base_cfg, generations=sens_gens, patience=sens_gens, seed=s, **{p: v})
                hs.append(GeneticAlgorithm(env, ev, cfg).run())
            sens[p][v] = hs
            fin = [x["hof"][-1] for x in hs]
            stab[p][str(v)] = dict(
                final_mean=float(np.mean(fin)), final_std=float(np.std(fin)),
                evals=int(np.mean([x["evals"][-1] for x in hs])),
                final_diversity=float(np.mean([x["diversity"][-1] for x in hs])),
                gen_to_95=float(np.mean([next((i for i, q in enumerate(x["hof"]) if q >= 0.95 * x["hof"][-1]),
                                              len(x["hof"])) for x in hs])))
            log(f"[sens] {p}={v}: {stab[p][str(v)]}")
    viz.plot_sensitivity(sens, f"{out}/06_sensitivity.png")
    out_res["sensitivity"] = stab

    # ---------------- 3. weight profiles (trade-off, no single solution)
    profs, ptab = {}, {}
    for nm, w in PROFILES.items():
        e2, ev2 = _setup(w)
        h = GeneticAlgorithm(e2, ev2, replace(base_cfg, seed=0)).run()
        profs[nm] = h["hof_genome"]
        ptab[nm] = dict(weights=w.as_dict(),
                        **{k: v for k, v in h["hof_genome"].metrics.items() if k != "cov_counts"})
        log(f"[profile] {nm}: cov={ptab[nm]['coverage']:.3f} det={ptab[nm]['detection']:.3f} "
            f"E={ptab[nm]['energy_norm']:.3f} T={ptab[nm]['time_norm']:.3f} H={ptab[nm]['hazard_norm']:.3f}")
    viz.plot_profiles(profs, env, f"{out}/07_weight_profiles.png")
    out_res["profiles"] = ptab

    with open(f"{out}/results.json", "w") as f:
        json.dump(out_res, f, indent=1, default=float)
    return out_res