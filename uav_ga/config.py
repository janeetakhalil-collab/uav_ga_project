"""Central configuration objects (all hyper-parameters live here)."""
from dataclasses import dataclass, asdict


@dataclass
class EnvConfig:
    size: int = 50          # grid is size x size cells
    n_drones: int = 3
    n_waypoints: int = 8    # K waypoints per drone (after the deployment point)
    n_targets: int = 30     # hidden targets to detect
    seed: int = 7           # environment is fixed/reproducible


@dataclass
class UAVConfig:
    max_range: float = 150.0   # hard limit: path length per drone (cells)
    battery: float = 200.0     # hard limit: energy per drone (energy units)
    v_min: float = 0.5         # cells / time-unit
    v_max: float = 2.0
    r_min: float = 2.0         # sensor radius at power 0 (cells)
    r_max: float = 6.0         # sensor radius at power 1
    samples_per_segment: int = 16
    t_max: float = 250.0       # reference mission time for normalisation
    h_ref: float = 100.0       # reference hazard exposure for normalisation


@dataclass
class Weights:
    coverage: float = 0.35
    detection: float = 0.25
    energy: float = 0.15
    time: float = 0.10
    hazard: float = 0.15

    def as_dict(self):
        return asdict(self)


@dataclass
class GAConfig:
    pop_size: int = 80
    generations: int = 150
    tournament_k: int = 4         # selection pressure
    crossover_rate: float = 0.85
    mutation_rate: float = 0.12   # per-gene probability
    elite: int = 2
    sigma0: float = 6.0           # initial Gaussian step (cells)
    sigma_min: float = 0.8        # final Gaussian step
    patience: int = 60            # stop if no improvement for this many gens
    stagnation_boost: int = 10    # gens without improvement -> 2x mutation + immigrants
    immigrant_frac: float = 0.10
    seed: int = 0