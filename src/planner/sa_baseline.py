"""
Standard Simulated Annealing (SA) Baseline for Route Planning.

Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). "Optimization by Simulated Annealing."
Science, 220(4598), 671-680.
Lin, S., & Kernighan, B. W. (1973). "An effective heuristic algorithm for the traveling-salesman problem."
Operations Research, 21(2), 498-516.

Native Permutation Representation:
----------------------------------
Operates directly on discrete visit-order permutations of stops:
an array of stop indices `[0, 1, ..., n - 1]` with no continuous random-key mapping needed.

Standard Simulated Annealing Mechanics:
---------------------------------------
1. Neighbor Generation: 2-Opt Segment-Reversal Move
   - Randomly chooses two distinct cut indices i < j.
   - Reverses the subsegment between i and j inclusive:
     order[i:j+1] = order[i:j+1][::-1]
   - Guaranteed valid permutation by construction (0 duplicates, 0 omitted stops).
2. Cooling Schedule: Geometric Cooling
   - T(t) = max(T_0 * (cooling_rate ** t), T_min)
   - T_0: Initial temperature (default: 100.0)
   - cooling_rate: Geometric decay parameter in (0, 1) (default: 0.99)
3. Acceptance Criterion: Metropolis-Hastings Rule (Minimization)
   - delta = candidate_cost - current_cost
   - If delta <= 0: accept candidate unconditionally (downhill improvement)
   - If delta > 0: accept candidate with probability P = exp(-delta / T)
4. Elite Preservation:
   - Tracks global best solution (best_order, best_score) across all accepted/rejected moves.
5. Stagnation Recovery:
   - If global best fails to improve for `patience = 15` consecutive temperature iterations,
     re-seeds search to escape frozen local basins while preserving the global elite.

Budget & Function Evaluation Parity:
------------------------------------
Uses the exact same `default_budget(dim)` as va_qpso, fixed_beta_qpso, pso_baseline, and ga_baseline:
    steps_per_temp = max(20, 4 * dim)
    max_iterations = max(100, 75 * dim)
    max_restarts = max(5, 5 * dim)
At each temperature step t, `steps_per_temp` candidate neighbor evaluations are performed,
yielding exactly `steps_per_temp * max_iterations` total function evaluations -- strictly matching
the evaluation capacity of QPSO, PSO, and GA.
"""

from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, overload
import numpy as np

from .fitness import CongestionLookup, score_route

FitnessFn = Callable[[np.ndarray], float]

# Canonical Default Hyperparameters
DEFAULT_T0 = 100.0              # Initial temperature
DEFAULT_COOLING_RATE = 0.99     # Geometric cooling factor: T(t) = T0 * (cooling_rate ** t)
DEFAULT_MIN_TEMP = 1e-6         # Temperature floor to prevent division by zero
DEFAULT_PATIENCE = 15           # Stagnation iterations before restart
DEFAULT_TOL = 1e-6              # Relative improvement tolerance


def default_budget(dim: int) -> Tuple[int, int, int]:
    """
    Function evaluation and temperature iteration budget scaled off `dim` (number of stops).
    Matches qpso.py, pso_baseline.py, and ga_baseline.py default_budget() exactly.

    Returns:
        (steps_per_temp, max_iterations, max_restarts)
    """
    return max(20, 4 * dim), max(100, 75 * dim), max(5, 5 * dim)


def _resolve_budget(
    dim: int,
    steps_per_temp: Optional[int],
    max_iterations: Optional[int],
    max_restarts: Optional[int],
) -> Tuple[int, int, int]:
    """Fill in any budget argument left as None from default_budget(dim)."""
    steps, iters, restarts = default_budget(dim)
    return (
        steps if steps_per_temp is None else steps_per_temp,
        iters if max_iterations is None else max_iterations,
        restarts if max_restarts is None else max_restarts,
    )


def two_opt_swap(order: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    2-Opt segment-reversal neighborhood move for permutation tours (Lin & Kernighan 1973).

    Selects two distinct cut points i < j and reverses the subsegment between them.
    Preserves 100% permutation validity (no duplicate or omitted stops).

    Args:
        order: Current stop visitation permutation of shape (n,).
        rng: Random number generator.

    Returns:
        Neighbor permutation copy with reversed subsegment.
    """
    n = len(order)
    if n <= 1:
        return np.copy(order)
    if n == 2:
        return order[::-1].copy()

    idx = rng.choice(n, size=2, replace=False)
    i = int(min(idx))
    j = int(max(idx))

    neighbor = np.copy(order)
    neighbor[i : j + 1] = neighbor[i : j + 1][::-1]
    return neighbor


@overload
def simulated_annealing(
    dim: int,
    fitness_fn: FitnessFn,
    steps_per_temp: Optional[int] = ...,
    max_iterations: Optional[int] = ...,
    t0: float = ...,
    cooling_rate: float = ...,
    min_temp: float = ...,
    seed: Optional[int] = ...,
    patience: int = ...,
    tol: float = ...,
    max_restarts: Optional[int] = ...,
    *,
    return_history: Literal[True],
) -> Tuple[np.ndarray, float, np.ndarray]:
    ...


@overload
def simulated_annealing(
    dim: int,
    fitness_fn: FitnessFn,
    steps_per_temp: Optional[int] = ...,
    max_iterations: Optional[int] = ...,
    t0: float = ...,
    cooling_rate: float = ...,
    min_temp: float = ...,
    seed: Optional[int] = ...,
    patience: int = ...,
    tol: float = ...,
    max_restarts: Optional[int] = ...,
    return_history: Literal[False] = ...,
) -> Tuple[np.ndarray, float]:
    ...


@overload
def simulated_annealing(
    dim: int,
    fitness_fn: FitnessFn,
    steps_per_temp: Optional[int] = ...,
    max_iterations: Optional[int] = ...,
    t0: float = ...,
    cooling_rate: float = ...,
    min_temp: float = ...,
    seed: Optional[int] = ...,
    patience: int = ...,
    tol: float = ...,
    max_restarts: Optional[int] = ...,
    return_history: bool = ...,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    ...


def simulated_annealing(
    dim: int,
    fitness_fn: FitnessFn,
    steps_per_temp: Optional[int] = None,
    max_iterations: Optional[int] = None,
    t0: float = DEFAULT_T0,
    cooling_rate: float = DEFAULT_COOLING_RATE,
    min_temp: float = DEFAULT_MIN_TEMP,
    seed: Optional[int] = None,
    patience: int = DEFAULT_PATIENCE,
    tol: float = DEFAULT_TOL,
    max_restarts: Optional[int] = None,
    return_history: bool = False,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """
    Run Simulated Annealing on discrete stop orderings using 2-opt segment reversal.

    Args:
        dim: Number of stops to sequence (permutation length).
        fitness_fn: Function mapping integer permutation array order -> float cost.
        steps_per_temp: Evaluations per temperature step (defaults to default_budget(dim)[0]).
        max_iterations: Number of temperature levels (defaults to default_budget(dim)[1]).
        t0: Initial temperature T_0 (default: 100.0).
        cooling_rate: Geometric cooling decay factor in (0, 1) (default: 0.99).
        min_temp: Temperature floor to avoid numerical singularity (default: 1e-6).
        seed: Random seed for deterministic reproducibility.
        patience: Stagnation temperature iterations before restart.
        tol: Relative tolerance required to count as improvement.
        max_restarts: Maximum consecutive stagnation restarts allowed.
        return_history: If True, returns history array of best fitness per iteration.

    Returns:
        (best_order, best_score) or (best_order, best_score, history)
    """
    steps_per_temp, max_iterations, max_restarts = _resolve_budget(
        dim, steps_per_temp, max_iterations, max_restarts
    )

    rng = np.random.default_rng(seed)

    def fresh_solution() -> Tuple[np.ndarray, float]:
        order = rng.permutation(dim)
        score = float(fitness_fn(order))
        return order, score

    current_order, current_score = fresh_solution()

    # Elite record tracked across all temperature cycles and restarts
    best_order = np.copy(current_order)
    best_score = current_score

    iterations_since_improvement = 0
    unproductive_restarts = 0
    history: List[float] = []

    for t in range(max_iterations):
        # Geometric cooling: T = T0 * (cooling_rate ** t)
        T = max(t0 * (cooling_rate ** t), min_temp)

        # Explore neighborhood at current temperature
        for _ in range(steps_per_temp):
            candidate = two_opt_swap(current_order, rng)
            candidate_score = float(fitness_fn(candidate))
            delta = candidate_score - current_score

            # Metropolis acceptance criterion
            if delta <= 0.0:
                accept = True
            else:
                exponent = -min(delta / T, 700.0)
                prob = float(np.exp(exponent))
                accept = bool(rng.uniform(0.0, 1.0) < prob)

            if accept:
                current_order = candidate
                current_score = candidate_score

            # Update elite tracker
            if candidate_score < best_score - tol:
                best_score = candidate_score
                best_order = np.copy(candidate)
                iterations_since_improvement = 0
                unproductive_restarts = 0

        if return_history:
            history.append(float(best_score))

        # Check for stagnation across temperature cycles
        if iterations_since_improvement >= patience:
            if unproductive_restarts >= max_restarts:
                break
            # Stagnation restart: re-seed current state to escape local attractor
            current_order, current_score = fresh_solution()
            if current_score < best_score - tol:
                best_score = current_score
                best_order = np.copy(current_order)
                unproductive_restarts = 0
            else:
                unproductive_restarts += 1
            iterations_since_improvement = 0
        else:
            iterations_since_improvement += 1

    if return_history:
        while len(history) < max_iterations:
            history.append(float(best_score))
        return best_order, best_score, np.asarray(history, dtype=float)

    return best_order, best_score


@overload
def replan(
    stops: List[str],
    distance_matrix: np.ndarray,
    congestion_lookup: CongestionLookup,
    volatility_index: float = ...,
    weights: Tuple[float, float, float] = ...,
    steps_per_temp: Optional[int] = ...,
    max_iterations: Optional[int] = ...,
    t0: float = ...,
    cooling_rate: float = ...,
    min_temp: float = ...,
    seed: Optional[int] = ...,
    patience: int = ...,
    tol: float = ...,
    max_restarts: Optional[int] = ...,
    *,
    return_history: Literal[True],
) -> Tuple[np.ndarray, float, np.ndarray]:
    ...


@overload
def replan(
    stops: List[str],
    distance_matrix: np.ndarray,
    congestion_lookup: CongestionLookup,
    volatility_index: float = ...,
    weights: Tuple[float, float, float] = ...,
    steps_per_temp: Optional[int] = ...,
    max_iterations: Optional[int] = ...,
    t0: float = ...,
    cooling_rate: float = ...,
    min_temp: float = ...,
    seed: Optional[int] = ...,
    patience: int = ...,
    tol: float = ...,
    max_restarts: Optional[int] = ...,
    return_history: Literal[False] = ...,
) -> Tuple[np.ndarray, float]:
    ...


@overload
def replan(
    stops: List[str],
    distance_matrix: np.ndarray,
    congestion_lookup: CongestionLookup,
    volatility_index: float = ...,
    weights: Tuple[float, float, float] = ...,
    steps_per_temp: Optional[int] = ...,
    max_iterations: Optional[int] = ...,
    t0: float = ...,
    cooling_rate: float = ...,
    min_temp: float = ...,
    seed: Optional[int] = ...,
    patience: int = ...,
    tol: float = ...,
    max_restarts: Optional[int] = ...,
    return_history: bool = ...,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    ...


def replan(
    stops: List[str],
    distance_matrix: np.ndarray,
    congestion_lookup: CongestionLookup,
    volatility_index: float = 0.5,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    steps_per_temp: Optional[int] = None,
    max_iterations: Optional[int] = None,
    t0: float = DEFAULT_T0,
    cooling_rate: float = DEFAULT_COOLING_RATE,
    min_temp: float = DEFAULT_MIN_TEMP,
    seed: Optional[int] = None,
    patience: int = DEFAULT_PATIENCE,
    tol: float = DEFAULT_TOL,
    max_restarts: Optional[int] = None,
    return_history: bool = False,
    physical_distance_matrix: Optional[np.ndarray] = None,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """
    Re-plan delivery tour order using Simulated Annealing baseline.

    Provides exact signature parity with `src.planner.qpso.replan()`, `pso_baseline.replan()`,
    and `ga_baseline.replan()` so that experiments, benchmarks, and simulation replanners
    can compare all optimizers under identical inputs and fitness.

    Args:
        stops: Stop identifiers mapping to distance_matrix rows/cols.
        distance_matrix: (n, n) matrix of live travel times.
        congestion_lookup: Congestion metrics per edge pair.
        volatility_index: Present for interface compatibility (standard SA is non-adaptive).
        weights: (w1, w2, w3) for travel time, distance, and congestion penalties.
        steps_per_temp: Steps per temperature level (or default_budget(n)[0]).
        max_iterations: Temperature levels (or default_budget(n)[1]).
        t0: Initial temperature (default: 100.0).
        cooling_rate: Geometric cooling decay factor (default: 0.99).
        min_temp: Minimum temperature floor.
        seed: Random seed.
        patience: Stagnation iterations before restart.
        tol: Relative tolerance for improvement.
        max_restarts: Max stagnation restarts.
        return_history: If True, returns history array of best fitness per iteration.
        physical_distance_matrix: Optional (n, n) matrix of physical distances (m).

    Returns:
        (best_order, best_score) or (best_order, best_score, history)
    """
    n = len(stops)
    if distance_matrix.shape != (n, n):
        raise ValueError(
            f"distance_matrix shape {distance_matrix.shape} does not match len(stops)={n}."
        )
    if physical_distance_matrix is not None and physical_distance_matrix.shape != (n, n):
        raise ValueError(
            f"physical_distance_matrix shape {physical_distance_matrix.shape} does not match len(stops)={n}."
        )

    def fitness_fn(order: np.ndarray) -> float:
        return score_route(
            order,
            distance_matrix,
            physical_distance_matrix if physical_distance_matrix is not None else distance_matrix,
            congestion_lookup,
            weights,
        )

    if return_history:
        return simulated_annealing(
            dim=n,
            fitness_fn=fitness_fn,
            steps_per_temp=steps_per_temp,
            max_iterations=max_iterations,
            t0=t0,
            cooling_rate=cooling_rate,
            min_temp=min_temp,
            seed=seed,
            patience=patience,
            tol=tol,
            max_restarts=max_restarts,
            return_history=True,
        )

    return simulated_annealing(
        dim=n,
        fitness_fn=fitness_fn,
        steps_per_temp=steps_per_temp,
        max_iterations=max_iterations,
        t0=t0,
        cooling_rate=cooling_rate,
        min_temp=min_temp,
        seed=seed,
        patience=patience,
        tol=tol,
        max_restarts=max_restarts,
        return_history=False,
    )
