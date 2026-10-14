"""Entry point.   python main.py            -> full experiment suite (~10-15 min on 1 core)
                 python main.py --quick     -> small smoke test (~10 s)
                 python main.py --single    -> one GA run + plots only (~30 s)"""
import argparse, os
from uav_ga.config import Weights, GAConfig
from uav_ga import experiments as ex, visualize as viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.single:
        from uav_ga.ga import GeneticAlgorithm
        env, ev = ex._setup(Weights())
        h = GeneticAlgorithm(env, ev, GAConfig()).run()
        viz.plot_environment(env, f"{a.out}/00_environment.png")
        viz.plot_convergence(h, f"{a.out}/01_convergence.png")
        viz.plot_evolution(h, env, ev, f"{a.out}/02_evolution_paths_coverage.png")
        viz.plot_final(h["hof_genome"], env, ev, f"{a.out}/03_final_solution.png")
        print({k: v for k, v in h["hof_genome"].metrics.items() if k != "cov_counts"})
    elif a.quick:
        ex.run_all(a.out, gens=20, pop=30, seeds=2, sens_seeds=1, sens_gens=15)
    else:
        ex.run_all(a.out)


if __name__ == "__main__":
    main()