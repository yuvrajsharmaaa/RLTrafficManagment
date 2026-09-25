"""
Standard (Canonical) Particle Swarm Optimization (PSO) Baseline for Route Planning.

Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization."
Proceedings of ICNN'95 - International Conference on Neural Networks, 1942-1948.
Shi, Y., & Eberhart, R. (1998). "A modified particle swarm optimizer."
1998 IEEE International Conference on Evolutionary Computation Proceedings, 69-73.

Standard Canonical PSO Update Equations:
----------------------------------------
For particle i in {0, ..., M - 1} and dimension d in {0, ..., D - 1} at iteration t:

    v[i][d] = w * v[i][d] + c1 * r1 * (personal_best[i][d] - x[i][d])
                          + c2 * r2 * (global_best[d] - x[i][d])
    x[i][d] = x[i][d] + v[i][d]

Where:
    - w: Inertia weight governing velocity momentum. Default: 0.7 (or linear anneal 0.9 -> 0.4).
    - c1: Cognitive acceleration coefficient (pull toward personal best). Default: 1.5.
    - c2: Social acceleration coefficient (pull toward swarm global best). Default: 1.5.
    - r1, r2: Independent uniform random numbers in [0, 1] drawn per particle, per dimension,
      and per iteration step: r1 ~ U(0, 1), r2 ~ U(0, 1).
    - v[i][d] is clamped to [-v_max, v_max] where v_max = 0.5 * (bounds[1] - bounds[0]) = 0.5.
    - x[i][d] is clamped to [bounds[0], bounds[1]] = [0.0, 1.0].

Exact Problem Representation & Fitness Parity:
----------------------------------------------
- Random-key encoding: identical to qpso.py via `src.planner.qpso_encoding.decode_order()`.
  The continuous position vector x in [0, 1]^D is decoded as `argsort(x)`.
- Route scoring: identical to qpso.py via `src.planner.fitness.score_route()`.
- Swarm budget & evaluations: identical to qpso.py via `default_budget(dim)`.
  num_particles = max(20, 4 * dim)
  max_iterations = max(100, 75 * dim)
  max_restarts = max(5, 5 * dim)
  patience = 15
  tol = 1e-6
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from .fitness import CongestionLookup, score_route
from .qpso_encoding import decode_order

FitnessFn = Callable[[np.ndarray], float]

# Exact Canonical Hyperparameter Defaults
DEFAULT_W = 0.7            # Canonical inertia weight (Shi & Eberhart 1998)
DEFAULT_C1 = 1.5           # Cognitive acceleration coefficient
DEFAULT_C2 = 1.5           # Social acceleration coefficient
DEFAULT_VMAX_RATIO = 0.5   # Velocity clamp ratio: v_max = ratio * (x_max - x_min)


def default_budget(dim: int) -> Tuple[int, int, int]:
    """
    Swarm budget scaled off `dim` (number of stops), matching qpso.py exactly.

    Returns:
        (num_particles, max_iterations, max_restarts)
    """
    return max(20, 4 * dim), max(100, 75 * dim), max(5, 5 * dim)


def _resolve_budget(dim: int, num_particles: Optional[int], max_iterations: Optional[int], max_restarts: Optional[int]):
    """Fill in any budget argument left as None from default_budget(dim)."""
    particles, iterations, restarts = default_budget(dim)
    return (
        particles if num_particles is None else num_particles,
        iterations if max_iterations is None else max_iterations,
        restarts if max_restarts is None else max_restarts,
    )


def standard_pso(
    dim: int,
    fitness_fn: FitnessFn,
    num_particles: Optional[int] = None,
    max_iterations: Optional[int] = None,
    w: float = DEFAULT_W,
    c1: float = DEFAULT_C1,
    c2: float = DEFAULT_C2,
    w_schedule: str = "constant",
    w_max: float = 0.9,
    w_min: float = 0.4,
    bounds: Tuple[float, float] = (0.0, 1.0),
    seed: Optional[int] = None,
    patience: int = 15,
    tol: float = 1e-6,
    max_restarts: Optional[int] = None,
    return_history: bool = False,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """
    Execute standard canonical Particle Swarm Optimization (Kennedy & Eberhart, 1995).

    Args:
        dim: Problem dimensionality (number of stops to sequence).
        fitness_fn: Function mapping continuous position vector x to scalar cost.
        num_particles: Swarm size (defaults to default_budget(dim)[0]).
        max_iterations: Maximum iteration budget (defaults to default_budget(dim)[1]).
        w: Inertia weight (default: 0.7).
        c1: Cognitive acceleration coefficient (default: 1.5).
        c2: Social acceleration coefficient (default: 1.5).
        w_schedule: 'constant' for fixed w, or 'linear' for decreasing schedule w_max -> w_min.
        w_max: Initial inertia weight when w_schedule='linear' (default: 0.9).
        w_min: Final inertia weight when w_schedule='linear' (default: 0.4).
        bounds: Coordinate bounds (lower, upper) for random keys. Default: (0.0, 1.0).
        seed: Random seed for deterministic reproducibility.
        patience: Consecutive iterations without improvement before stagnation restart.
        tol: Minimum score reduction required to count as improvement.
        max_restarts: Maximum consecutive stagnation restarts allowed.

    Returns:
        (best_position, best_score): Best continuous position vector found and its fitness score.
    """
    num_particles, max_iterations, max_restarts = _resolve_budget(
        dim, num_particles, max_iterations, max_restarts
    )

    rng = np.random.default_rng(seed)
    span = bounds[1] - bounds[0]
    v_max = DEFAULT_VMAX_RATIO * span

    def fresh_swarm():
        positions = rng.uniform(bounds[0], bounds[1], (num_particles, dim))
        velocities = rng.uniform(-v_max, v_max, (num_particles, dim))
        return positions, velocities, np.copy(positions), np.full(num_particles, np.inf)

    positions, velocities, personal_best, personal_best_scores = fresh_swarm()
    global_best = np.zeros(dim)
    global_best_score = np.inf

    # Elite record tracked across all restart cycles
    best_position = np.zeros(dim)
    best_score = np.inf

    iterations_since_improvement = 0
    unproductive_restarts = 0
    history: List[float] = []

    for t in range(max_iterations):
        # 1. Evaluate fitness, update personal bests and global best
        for i in range(num_particles):
            score = fitness_fn(positions[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best[i] = np.copy(positions[i])
            if score < global_best_score - tol:
                global_best_score = score
                global_best = np.copy(positions[i])
                iterations_since_improvement = 0
            if score < best_score - tol:
                best_score = score
                best_position = np.copy(positions[i])
                unproductive_restarts = 0

        if return_history:
            history.append(float(best_score))

        # Stagnation restart check (identical restart logic as qpso.py)
        if iterations_since_improvement >= patience:
            if unproductive_restarts >= max_restarts:
                break
            positions, velocities, personal_best, personal_best_scores = fresh_swarm()
            global_best = np.zeros(dim)
            global_best_score = np.inf
            iterations_since_improvement = 0
            unproductive_restarts += 1
            continue

        iterations_since_improvement += 1

        # 2. Determine current inertia weight w(t)
        if w_schedule == "linear":
            current_w = w_max - (w_max - w_min) * (t / max_iterations)
        else:
            current_w = w

        # 3. Canonical Velocity and Position Updates (Kennedy & Eberhart, 1995)
        # r1, r2 ~ Uniform(0, 1) drawn per particle, per dimension
        r1 = rng.uniform(0.0, 1.0, (num_particles, dim))
        r2 = rng.uniform(0.0, 1.0, (num_particles, dim))

        velocities = (
            current_w * velocities
            + c1 * r1 * (personal_best - positions)
            + c2 * r2 * (global_best - positions)
        )

        # Velocity clamping to prevent divergence
        velocities = np.clip(velocities, -v_max, v_max)

        # Position update
        positions = positions + velocities

        # Position clamping to search space bounds
        positions = np.clip(positions, bounds[0], bounds[1])

    if return_history:
        while len(history) < max_iterations:
            history.append(float(best_score))
        return best_position, best_score, np.asarray(history, dtype=float)

    return best_position, best_score


def replan(
    stops: List[str],
    distance_matrix: np.ndarray,
    congestion_lookup: CongestionLookup,
    volatility_index: float = 0.5,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_particles: Optional[int] = None,
    max_iterations: Optional[int] = None,
    w: float = DEFAULT_W,
    c1: float = DEFAULT_C1,
    c2: float = DEFAULT_C2,
    w_schedule: str = "constant",
    seed: Optional[int] = None,
    patience: int = 15,
    tol: float = 1e-6,
    max_restarts: Optional[int] = None,
    return_history: bool = False,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """
    Execute standard PSO to sequence stops on a frozen state snapshot.

    Provides exact signature parity with `src.planner.qpso.replan()` so that
    experiments, benchmarks, and simulation replanners can compare standard PSO
    against QPSO under identical inputs, encoding, and fitness evaluation.

    Args:
        stops: Stop identifiers mapping to distance_matrix rows/cols.
        distance_matrix: (n, n) matrix of live travel times.
        congestion_lookup: Congestion metrics per edge pair.
        volatility_index: Present for interface compatibility (standard PSO is non-adaptive).
        weights: (w1, w2, w3) for travel time, distance, and congestion penalties.
        num_particles: Swarm size (or default_budget(n)[0]).
        max_iterations: Iteration budget (or default_budget(n)[1]).
        w: Inertia weight (default: 0.7).
        c1: Cognitive acceleration coefficient (default: 1.5).
        c2: Social acceleration coefficient (default: 1.5).
        w_schedule: 'constant' or 'linear'.
        seed: Random seed.
        patience: Stagnation iterations before restart.
        tol: Relative tolerance for improvement.
        max_restarts: Max stagnation restarts.

    Returns:
        (best_order, best_score): Decoded stop visitation permutation and total fitness.
    """
    n = len(stops)
    if distance_matrix.shape != (n, n):
        raise ValueError(
            f"distance_matrix shape {distance_matrix.shape} does not match len(stops)={n}."
        )

    def fitness_fn(x: np.ndarray) -> float:
        order = decode_order(x)
        return score_route(order, distance_matrix, congestion_lookup, weights)

    res = standard_pso(
        dim=n,
        fitness_fn=fitness_fn,
        num_particles=num_particles,
        max_iterations=max_iterations,
        w=w,
        c1=c1,
        c2=c2,
        w_schedule=w_schedule,
        seed=seed,
        patience=patience,
        tol=tol,
        max_restarts=max_restarts,
        return_history=return_history,
    )

    if return_history:
        best_position, best_score, history = res
        return decode_order(best_position), best_score, history

    best_position, best_score = res
    return decode_order(best_position), best_score
