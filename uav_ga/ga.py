"""Object-oriented Genetic Algorithm engine."""
import numpy as np
from .chromosome import Chromosome
from .constraints import repair
from .operators import tournament, crossover, mutate


class GeneticAlgorithm:
    def __init__(self, env, evaluator, cfg):
        self.env, self.ev, self.cfg = env, evaluator, cfg
        self.D, self.K = env.cfg.n_drones, env.cfg.n_waypoints

    def _new(self, rng):
        ch = Chromosome.random(self.env, rng, self.D, self.K)
        return self.ev.evaluate(repair(ch, self.env))

    def _diversity(self, pop):
        wp = np.stack([c.wp for c in pop])
        return float(np.std(wp, axis=0).mean() / self.env.n)

    def run(self):
        cfg, ev, env = self.cfg, self.ev, self.env
        rng = np.random.default_rng(cfg.seed)
        N = cfg.pop_size
        pop = [self._new(rng) for _ in range(N)]
        evals = N
        h = {k: [] for k in ("best", "avg", "avg_base", "feas_frac", "worst",
                             "diversity", "hof", "pm", "evals", "best_genomes")}
        hof, hof_val, last_hof, stagn = None, -np.inf, -np.inf, 0
        for g in range(cfg.generations):
            gfrac = g / max(1, cfg.generations - 1)
            fits = np.array([ev.score(c.metrics, gfrac) for c in pop])
            for c, f in zip(pop, fits):
                c.fitness = f
                if c.metrics["feasible"]:
                    b = ev.base(c.metrics)
                    if b > hof_val:
                        hof_val, hof = b, c.copy()
            stagn = 0 if hof_val > last_hof + 1e-4 else stagn + 1
            last_hof = max(last_hof, hof_val)
            pm = min(0.5, cfg.mutation_rate * (2.0 if stagn >= cfg.stagnation_boost else 1.0))
            ib = int(np.argmax(fits))
            h["best"].append(float(fits[ib]))
            h["avg"].append(float(fits.mean()))
            h["avg_base"].append(float(np.mean([ev.base(c.metrics) for c in pop])))
            h["feas_frac"].append(float(np.mean([c.metrics["feasible"] for c in pop])))
            h["worst"].append(float(fits.min()))
            h["diversity"].append(self._diversity(pop))
            h["hof"].append(float(hof_val) if np.isfinite(hof_val) else float("nan"))
            h["pm"].append(pm)
            h["evals"].append(evals)
            h["best_genomes"].append(pop[ib].copy())
            # ---- termination criteria: max generations OR stagnation (patience)
            if g == cfg.generations - 1 or stagn >= cfg.patience:
                break
            # ---- next generation
            sigma = cfg.sigma_min + (cfg.sigma0 - cfg.sigma_min) * (1 - gfrac)
            order = np.argsort(-fits)
            new = [pop[i].copy() for i in order[:cfg.elite]]
            while len(new) < N:
                a = pop[tournament(fits, cfg.tournament_k, rng)]
                b = pop[tournament(fits, cfg.tournament_k, rng)]
                if rng.random() < cfg.crossover_rate:
                    kids = crossover(a, b, rng)
                else:
                    kids = (a.copy(), b.copy())
                for c in kids:
                    mutate(c, rng, pm, sigma, env)
                    repair(c, env)
                    ev.evaluate(c)
                    evals += 1
                    new.append(c)
            new = new[:N]
            if stagn > 0 and stagn % cfg.stagnation_boost == 0:      # random immigrants
                for i in range(int(cfg.immigrant_frac * N)):
                    new[N - 1 - i] = self._new(rng)
                    evals += 1
            pop = new
        h["final_pop"] = pop
        h["hof_genome"] = hof if hof is not None else pop[int(np.argmax([c.fitness for c in pop]))]
        h["generations_run"] = len(h["best"])
        return h