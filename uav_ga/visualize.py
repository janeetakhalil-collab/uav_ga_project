"""All plots (matplotlib, headless)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#1f77b4", "#2ca02c", "#e377c2", "#17becf"]


def _bg(ax, env, title=""):
    n = env.n
    ax.imshow(env.priority.T, origin="lower", cmap="YlOrRd", vmin=0, vmax=1, alpha=0.85)
    ax.contourf(env.hazard.T, levels=[0.35, 0.6, 1.01], colors=["#6baed6", "#08519c"], alpha=0.35)
    ax.imshow(np.ma.masked_where(~env.nofly.T, env.nofly.T.astype(float)), origin="lower",
              cmap="Greys", vmin=0, vmax=1, alpha=0.9)
    ax.scatter(env.targets[:, 0], env.targets[:, 1], marker="x", c="k", s=12, lw=0.8)
    ax.set_xlim(-.5, n - .5); ax.set_ylim(-.5, n - .5); ax.set_aspect("equal")
    ax.set_xticks([0, 25, 49]); ax.set_yticks([0, 25, 49]); ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)


def draw_paths(ax, env, ch, title=""):
    _bg(ax, env, title)
    P = ch.path_points()
    for d in range(ch.D):
        c = COLORS[d % len(COLORS)]
        for k in range(ch.K):
            on = ch.active[d, k]
            ax.plot(P[d, k:k + 2, 0], P[d, k:k + 2, 1], color=c, lw=2.2 if on else 1.0,
                    ls="-" if on else ":", alpha=0.95)
        ax.scatter(P[d, 1:, 0], P[d, 1:, 1], s=9, color=c, zorder=3)
        ax.scatter(*P[d, 0], marker="^", s=55, color=c, edgecolor="k", zorder=4)


def plot_environment(env, path):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    im = ax[0].imshow(env.priority.T, origin="lower", cmap="YlOrRd"); ax[0].set_title("Disaster priority map")
    fig.colorbar(im, ax=ax[0], fraction=.046)
    im = ax[1].imshow(env.hazard.T, origin="lower", cmap="Blues"); ax[1].set_title("Hazard (wind/radiation) map")
    fig.colorbar(im, ax=ax[1], fraction=.046)
    _bg(ax[2], env, "Combined: priority, hazard, no-fly (black), targets (x)")
    plt.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_convergence(h, path, title="GA convergence: best vs average fitness"):
    g = np.arange(len(h["best"]))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(g, h["best"], label="Best fitness (population, penalised)", c="tab:green", lw=2)
    ax[0].plot(g, h["avg_base"], label="Average fitness (population, objective value)", c="tab:blue", lw=2)
    ax[0].plot(g, h["hof"], label="Best feasible so far", c="tab:red", lw=1.2, ls="--")
    ax[0].set_xlabel("Generation"); ax[0].set_ylabel("Fitness"); ax[0].legend(fontsize=8); ax[0].set_title(title, fontsize=10)
    ax[1].plot(g, h["feas_frac"], c="tab:brown", lw=2)
    ax[1].set_ylim(0, 1.02); ax[1].set_xlabel("Generation"); ax[1].set_ylabel("Fraction of feasible individuals")
    ax[1].set_title("Constraint satisfaction in the population", fontsize=10)
    ax[2].plot(g, h["diversity"], c="tab:purple")
    ax[2].set_xlabel("Generation"); ax[2].set_ylabel("Waypoint std / grid size", color="tab:purple")
    a2 = ax[2].twinx(); a2.step(g, h["pm"], c="tab:orange", where="post")
    a2.set_ylabel("Mutation rate", color="tab:orange"); ax[2].set_title("Diversity & adaptive mutation", fontsize=10)
    for a in ax: a.grid(alpha=.3)
    plt.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_evolution(h, env, ev, path, gens=None):
    G = len(h["best_genomes"])
    gens = gens or sorted({0, max(1, G // 12), G // 4, G // 2, G - 1})
    fig, ax = plt.subplots(2, len(gens), figsize=(3.6 * len(gens), 7.2))
    for j, g in enumerate(gens):
        ch = h["best_genomes"][g]
        m = ev.simulate(ch, counts=True)
        draw_paths(ax[0, j], env, ch, f"Gen {g}: paths (cov={m['coverage']:.2f}, det={m['detection']:.2f})")
        ax[1, j].imshow(m["cov_counts"].T, origin="lower", cmap="viridis")
        ax[1, j].contour(env.nofly.T, levels=[0.5], colors="w", linewidths=1)
        ax[1, j].set_title(f"Gen {g}: coverage heat-map", fontsize=9); ax[1, j].tick_params(labelsize=7)
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_final(ch, env, ev, path):
    m = ev.simulate(ch, counts=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    draw_paths(ax[0], env, ch, "Final evolved flight paths  (solid = sensor ON, dotted = OFF)")
    im = ax[1].imshow(m["cov_counts"].T, origin="lower", cmap="viridis")
    ax[1].contour(env.nofly.T, levels=[0.5], colors="w", linewidths=1)
    ax[1].contour(env.priority.T, levels=[0.5, 0.75], colors="r", linewidths=.8)
    ax[1].set_title("Final coverage heat-map (red contours = high priority)", fontsize=9)
    fig.colorbar(im, ax=ax[1], fraction=.046, label="times covered")
    plt.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _pad(curves, L):
    return np.array([c + [c[-1]] * (L - len(c)) for c in curves])


def plot_sensitivity(res, path):
    """res: {param: {value: [hist,...]}}"""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    names = {"pop_size": "Population size", "mutation_rate": "Mutation rate",
             "tournament_k": "Tournament size (selection pressure)"}
    for a, (p, d) in zip(ax, res.items()):
        for v, hs in d.items():
            L = max(len(x["hof"]) for x in hs)
            arr = _pad([x["hof"] for x in hs], L)
            mu, sd = np.nanmean(arr, 0), np.nanstd(arr, 0)
            a.plot(mu, label=f"{v}", lw=2); a.fill_between(range(L), mu - sd, mu + sd, alpha=.15)
        a.set_title(names[p]); a.set_xlabel("Generation"); a.set_ylabel("Best feasible fitness")
        a.grid(alpha=.3); a.legend()
    plt.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_baseline(ga_hs, rs_rs, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hs, lab, c in ((ga_hs, "Genetic Algorithm", "tab:green"), (rs_rs, "Random search", "tab:red")):
        L = min(len(x["hof"]) for x in hs)
        arr = np.array([x["hof"][:L] for x in hs]); ev_ = hs[0]["evals"][:L]
        ax.plot(ev_, np.nanmean(arr, 0), c=c, lw=2, label=lab)
        ax.fill_between(ev_, np.nanmin(arr, 0), np.nanmax(arr, 0), color=c, alpha=.15)
    ax.set_xlabel("Fitness evaluations"); ax.set_ylabel("Best feasible fitness"); ax.grid(alpha=.3); ax.legend()
    ax.set_title("GA vs random search (mean, min-max band over seeds)")
    plt.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_uniqueness(ga_hs, rs_best, env, path):
    n = len(ga_hs) + 1
    fig, ax = plt.subplots(2, 3, figsize=(11, 7.3)); ax = ax.ravel()
    for i, h in enumerate(ga_hs):
        draw_paths(ax[i], env, h["hof_genome"], f"GA seed {i}")
    draw_paths(ax[len(ga_hs)], env, rs_best, "Best random-search solution")
    for a in ax[n:]: a.axis("off")
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_profiles(profs, env, path):
    """profs: {name: genome-with-metrics}"""
    names = list(profs)
    fig, ax = plt.subplots(2, len(names), figsize=(3.7 * len(names), 7.5))
    for j, nm in enumerate(names):
        draw_paths(ax[0, j], env, profs[nm], nm)
    keys = [("coverage", "Coverage"), ("detection", "Detection"), ("energy_norm", "Energy"),
            ("time_norm", "Time"), ("hazard_norm", "Hazard")]
    gs = ax[1, 0].get_gridspec()
    for a in ax[1, :]: a.remove()
    bx = fig.add_subplot(gs[1, :])
    w = 0.8 / len(names)
    for i, nm in enumerate(names):
        bx.bar(np.arange(len(keys)) + i * w, [profs[nm].metrics[k] for k, _ in keys], w, label=nm)
    bx.set_xticks(np.arange(len(keys)) + 0.4 - w / 2); bx.set_xticklabels([l for _, l in keys])
    bx.set_ylabel("Normalised value"); bx.legend(); bx.grid(alpha=.3, axis="y")
    bx.set_title("Trade-offs between objectives for different weight profiles")
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)