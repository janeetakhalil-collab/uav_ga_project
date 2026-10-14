# UAV Fleet Strategy Engine – Genetic Algorithm (AI Assignment 2, BSCS-5)

A Genetic Algorithm that jointly evolves deployment positions, waypoint trajectories,
speed profiles and sensor power/ON-OFF schedules for a fleet of 3 UAVs surveying a
50x50 hazard grid, trading off coverage / detection against energy, time and hazard
exposure, under hard constraints (range, battery, no-fly zones).

## Setup & run
    python -m venv venv && source venv/bin/activate      # optional
    pip install -r requirements.txt
    python main.py --single      # one GA run + plots (~30 s)
    python main.py --quick       # smoke test of whole pipeline (~10 s)
    python main.py               # full suite (~10-15 min, 1 core)
    python -m uav_ga.tuned       # extra experiment with mutation 0.02 / 0.05

Figures and results.json are written to results/. Everything is seeded.

## Chromosome (per drone, K=8 segments; 3*(2+5*8) = 126 genes)
deploy[d] real (x,y) | wp[d,k] real (x,y) | speed[d,k] real [0.5,2]
power[d,k] real [0,1] | active[d,k] binary (sensor ON/OFF)

## Fitness
F = 0.35*Coverage + 0.25*Detection - 0.15*Energy - 0.10*Time - 0.15*Hazard - lambda(g)*Violation
lambda(g) = 2*(1 + 3*g/G). Weights are editable in config.py.

## Constraints
Repair: clip to grid, push waypoints out of no-fly cells, contract over-long routes.
Penalty (dynamic): path length > 150, battery > 200, segments crossing no-fly cells.

## Notes
- Default settings (mutation_rate=0.12) reproduce the reported figures.
- A lower mutation rate is better: GAConfig(mutation_rate=0.05) gives 0.434 +/- 0.009
  vs 0.362 +/- 0.021 (5 seeds, 150 generations).
- The random-search baseline uses the same evaluation budget as the GA.