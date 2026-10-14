**Strategy Engine for a UAV Fleet using a Genetic Algorithm**

Artificial Intelligence – BSCS 5, Bahria University Lahore Campus – Assignment 2 (CLO4)

**Name:** Janeeta Khalil     **Roll No:** 03-134242-030

---

### 1. Overview

Three UAVs survey a 50×50 grid with disaster hotspots, hazard pockets, no-fly zones, and hidden targets. The engine decides each drone's deployment, waypoints, speed, and sensor schedule. Objectives conflict: full-power sweeps maximise coverage but drain battery and cross hazards; staying safe and cheap leaves hotspots unseen. There is no single optimum, only trade-offs. A Genetic Algorithm fits because the landscape is non-convex, partly discrete, and only evaluable by simulation.

![Figure 1](../results/00_environment.png)

*Figure 1: Priority map, hazard map, and combined view with no-fly zones (black) and hidden targets (×).*

---

### 2. Chromosome Design

Hybrid real/binary multi-segment chromosome. For each drone d and segment k = 1..8:

**Table 1: Chromosome gene blocks**

| Gene block | Type | Meaning |
|---|---|---|
| deploy[d] (x,y) | real | deployment cell |
| wp[d,k] (x,y) | real | waypoint k |
| speed[d,k] | real in [0.5, 2] | cruise speed on segment k |
| power[d,k] | real in [0, 1] | sensor power (radius 2–6 cells) |
| active[d,k] | binary | sensor ON/OFF on segment k |

Total = 3 × (2 + 5·8) = 126 genes.

Rationale: real coordinates let spatial operators act smoothly; sparse waypoints with interpolated segments keep the genome short but describe long routes; speed/power/schedule sit in the same segment as the waypoint so one crossover preserves both the route piece and how it is flown; the binary schedule lets the GA turn the sensor off in transit to save energy.

---

### 3. Fitness

Each segment is sampled at 16 points; sensor discs are stamped on the grid; energy, time, hazard, and violations are accumulated.

- Coverage = priority-weighted fraction of covered cells.
- Detection = mean over targets of q = min(1, (0.4+0.6·power)·1.2/speed) if inside an active footprint, else 0.
- Energy = Σ[length·(0.5 + 0.5v²) + 2·power²·time·ON] / total battery.
- Time = slowest drone's flight time / 250.
- Hazard = Σ hazard(cell)·time-in-cell / 100.

**F = 0.35·Cov + 0.25·Det − 0.15·E − 0.10·T − 0.15·H − λ(g)·V**, λ(g) = 2·(1 + 3g/G).

Weights: benefits sum to 0.60, costs to 0.40 — mission success dominates but costs matter. Coverage outranks detection because it is broader and partly implies detection. Energy and hazard (0.15 each) are the real operational risks; time is lowest (0.10) because the range limit rarely binds. All terms are normalised to roughly [0, 1].

---

### 4. Genetic Operators

- Initialisation: 50 % random-walk routes, 50 % uniform-random point sets.
- Selection: tournament of size k, with 2 elites.
- Crossover (rate 0.85), one of three chosen at random: (i) route swap; (ii) segment swap of a waypoint sub-sequence with speed/power/schedule; (iii) geometric convex blend of coordinates and continuous genes, uniform crossover for bits.
- Mutation: Gaussian waypoint/deployment shift (σ annealed 6 → 0.8 cells); jump mutation re-sampling uniformly or from the priority map; waypoint swap and sub-route reversal; speed/power perturbation; sensor-bit flip.
- Premature-convergence control: after 10 stagnant generations, mutation doubles and 10 % random immigrants replace the worst.
- Termination: 150 generations, or 60 stagnant generations.

---

### 5. Constraints

Hard: path ≤ 150 cells/drone, battery ≤ 200/drone, no-fly zones forbidden.

- Repair: clip bounds; move waypoints out of no-fly cells; contract over-long routes by 0.98·R/length.
- Dynamic penalty: V = Σmax(0, len−150)/150 + Σmax(0, E−200)/200 + 10·(length inside no-fly)/150, multiplied by λ(g), which grows 4× over the run.
- All 5 GA runs and 5 random-search runs returned feasible solutions.

---

### 6. Results

Best fitness rises from 0.05 to 0.38; diversity falls 0.28 → ~0.10 without collapsing. Early routes cross hazards; later generations deploy drones beside hotspots, shorten transit legs, switch the sensor off in transit, and route around hazards. Final: coverage 0.73, detection 1.00, energy 0.48, time 0.32, hazard 0.13, no violations.

![Figure 2](../results/01_convergence.png)

*Figure 2: GA convergence — best vs average fitness, constraint satisfaction, and diversity.*

![Figure 3](../results/02_evolution_paths_coverage.png)

*Figure 3: Best paths and coverage heatmaps at generations 0, 12, 37, 75, 149.*

![Figure 4](../results/03_final_solution.png)

*Figure 4: Final evolved paths (solid = sensor ON, dotted = OFF) and coverage heatmap.*

---

### 7. Hyperparameter Sensitivity

**Table 2: Sensitivity (3 seeds, 80 generations)**

| Parameter | Value | Fitness (mean ± std) | Diversity | Gens to 95% |
|---|---|---|---|---|
| Population | 20 | 0.312 ± 0.026 | 0.083 | 47 |
|  | 40 | 0.331 ± 0.007 | 0.107 | 49 |
|  | 80 | 0.369 ± 0.003 | 0.105 | 50 |
| Mutation | 0.02 | 0.420 ± 0.010 | 0.020 | 50 |
|  | 0.12 | 0.369 ± 0.003 | 0.105 | 50 |
|  | 0.30 | 0.235 ± 0.015 | 0.172 | 32 |
| Tournament k | 2 | 0.273 ± 0.038 | 0.116 | 55 |
|  | 4 | 0.369 ± 0.003 | 0.105 | 50 |
|  | 8 | 0.372 ± 0.011 | 0.105 | 29 |

Findings: larger populations give better and more stable results; mutation rate has the largest effect (0.02 best for fitness, 0.30 noise-dominated) — recommend 0.05 for balance; k = 2 too weak, k = 8 converges fastest but loses diversity.

**Table 3: Weight-profile trade-offs**

| Profile (wC, wD, wE, wT, wH) | Cov | Det | Energy | Time | Hazard |
|---|---|---|---|---|---|
| Balanced (.35, .25, .15, .10, .15) | 0.73 | 1.00 | 0.48 | 0.32 | 0.13 |
| Coverage-first (.50, .30, .05, .05, .10) | 0.83 | 0.99 | 0.79 | 0.36 | 0.41 |
| Energy-saver (.25, .15, .35, .15, .10) | 0.59 | 0.83 | 0.32 | 0.23 | 0.20 |
| Safety-first (.25, .15, .10, .10, .40) | 0.65 | 0.87 | 0.48 | 0.22 | 0.03 |

Coverage-first gains 10 points of coverage but spends 65 % more energy and ~3× the hazard exposure; Safety-first nearly eliminates hazard at the cost of 8 coverage points. The engine returns a solution consistent with the operator's priorities, not a universal optimum.

![Figure 5](../results/06_sensitivity.png)

*Figure 5: Hyperparameter sensitivity (population size, mutation rate, tournament size).*

![Figure 6](../results/07_weight_profiles.png)

*Figure 6: Trade-offs between objectives for four different weight profiles.*

---

### 8. GA vs Random Search

**Table 4: GA vs random search (5 seeds each)**

| Metric (mean ± std) | GA (mut 0.12) | Random search |
|---|---|---|
| Fitness | 0.362 ± 0.021 | 0.258 ± 0.011 |
| Coverage | 0.746 ± 0.027 | 0.677 ± 0.063 |
| Detection | 0.937 ± 0.051 | 0.770 ± 0.041 |
| Energy (norm.) | 0.534 ± 0.059 | 0.650 ± 0.074 |
| Hazard (norm.) | 0.145 ± 0.061 | 0.297 ± 0.160 |
| Time (norm.) | 0.318 ± 0.034 | 0.300 ± 0.032 |

The worst GA run beats the best random run. GA improves fitness by ~40 % (or ~68 % with mutation 0.05), with higher coverage and detection, less energy, and half the hazard exposure. The five GA runs reach similar fitness through different route topologies — coverage maps overlap at Jaccard 0.65 (vs 0.47 for random search).

![Figure 7](../results/04_ga_vs_random.png)

*Figure 7: GA vs random search — best feasible fitness over evaluations.*

![Figure 8](../results/05_uniqueness_across_seeds.png)

*Figure 8: Five GA runs plus one random-search solution — different topologies, similar fitness.*

---

### 9. Limitations and Future Work

Weighted-sum scalarisation gives one trade-off per run; NSGA-II would return the whole Pareto front. Segments are straight, hazards static, no collision avoidance. Extensions: NSGA-II, time-varying hazards, re-planning on drone failure.

---

### 10. Reproducibility

`python main.py` regenerates every figure and `results/results.json` (all seeded).

**GitHub Repository:** https://github.com/janeetakhalil-collab/uav_ga_project
