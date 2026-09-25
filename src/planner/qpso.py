"""
QPSO per Sun, Feng & Xu (2004)'s delta-potential-well formulation.

Sun, J., Feng, B., & Xu, W. (2004). "Particle swarm optimization with
particles having quantum behavior." Proceedings of the 2004 Congress on
Evolutionary Computation, 325-331.

Exact update equations used below (particle i, dimension d, iteration t):

    mbest_d = (1/M) * sum_i personal_best[i][d]
    phi ~ U(0, 1)
    p_id = phi * personal_best[i][d] + (1 - phi) * global_best[d]
    u ~ U(0, 1);  k ~ U(0, 1)
    x[i][d] = p_id + beta * |mbest_d - x[i][d]| * ln(1/u)   if k >= 0.5
            = p_id - beta * |mbest_d - x[i][d]| * ln(1/u)   otherwise

This is the standard form cited throughout the QPSO literature -- not a
simplified variant. Both variants below (fixed_beta_qpso, va_qpso) share this
exact core loop; they differ ONLY in how `beta` is computed each iteration.

Restart-on-stagnation
---------------------
The update equations above are used exactly as published. What is layered on
top is a restart strategy, applied identically to BOTH beta variants so the
fixed-beta vs volatility-adaptive comparison stays fair.

Motivation (measured, see validate_brute_force.py): a single uninterrupted
run contracts toward mbest/gbest until the random keys concentrate so tightly
that argsort keeps decoding the same family of orderings. On a 6-stop case
the swarm reached only 26-30% of the 720 possible orderings across 6000
evaluations -- where uniform random sampling covers 100% on the same budget
-- and stayed trapped at a local optimum even when run for 1000 iterations
with stopping disabled. It was trapped, not truncated, so a bigger iteration
budget does not help.

On stagnation (global_best not improved by more than `tol` for `patience`
consecutive iterations) the swarm is re-initialized: fresh uniform random
positions, cleared personal bests, and a cleared global_best. Each such
restart cycle is therefore a full, untouched QPSO run in its own right; the
best solution found across ALL cycles is tracked separately and returned.
Clearing global_best matters: on a held-out sweep of seeds 30-129 on the
6-stop case, retaining it as a surviving attractor found the true optimum in
95/100 runs versus 100/100 when cleared (both with restarts unlimited within
a 200-iteration budget).

Termination: the run ends after `max_restarts` consecutive restart cycles
fail to improve the across-cycle best, or when `max_iterations` total
iterations are consumed, whichever comes first. Note `t` in beta(t) is the
GLOBAL iteration index, so fixed_beta_qpso's linear anneal still spans the
whole budget rather than resetting each cycle.

Budget: num_particles, max_iterations and max_restarts all default to None
and are then scaled off `dim` by default_budget() -- a flat budget silently
degrades as the problem grows. Pass explicit values to override.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .fitness import CongestionLookup, score_route
from .qpso_encoding import decode_order

FitnessFn = Callable[[np.ndarray], float]
BetaFn = Callable[[int], float]  # iteration t -> beta(t)


def default_budget(dim: int) -> Tuple[int, int, int]:
    """
    Swarm budget scaled off `dim` (the number of stops).

    A flat budget does not hold as the problem grows: permutation space is
    dim!, so a setting tuned at one size silently degrades at the next. On
    held-out seeds against brute-force optima, a flat (30 particles, 200
    iterations, 5 restarts) found the true optimum in 50/50, 48/50, 43/50
    and 38/50 runs at dim 6, 7, 8 and 9 respectively -- steadily worse with
    size. The scaling below held at 50/50 across all four sizes.

    Cost at dim=9 is ~176ms per run, which is comfortably inside a re-plan
    tick; the cheaper (50*dim, 3*dim) variant runs ~115ms but gave up a run
    at dim=9, and this is the budget the reported optima depend on.

    Returns:
        (num_particles, max_iterations, max_restarts)
    """
    return max(20, 4 * dim), max(100, 75 * dim), max(5, 5 * dim)


def _resolve_budget(dim, num_particles, max_iterations, max_restarts):
    """Fill in any budget argument left as None from default_budget(dim)."""
    particles, iterations, restarts = default_budget(dim)
    return (
        particles if num_particles is None else num_particles,
        iterations if max_iterations is None else max_iterations,
        restarts if max_restarts is None else max_restarts,
    )


def _run_qpso(
    dim: int,
    fitness_fn: FitnessFn,
    beta_fn: BetaFn,
    num_particles: int,
    max_iterations: int,
    bounds: Tuple[float, float],
    seed: Optional[int],
    patience: int,
    tol: float,
    max_restarts: int,
    return_history: bool = False,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """
    Shared QPSO core loop (Sun, Feng & Xu, 2004) with restart-on-stagnation
    -- see module docstring for the exact update equations and the restart
    rationale. Both fixed_beta_qpso and va_qpso call this with different
    `beta_fn` implementations and nothing else differs.

    Returns:
        (best position across all restart cycles, its fitness).
    """
    rng = np.random.default_rng(seed)

    def fresh_swarm():
        positions = rng.uniform(bounds[0], bounds[1], (num_particles, dim))
        return positions, np.copy(positions), np.full(num_particles, np.inf)

    positions, personal_best, personal_best_scores = fresh_swarm()
    global_best = np.zeros(dim)
    global_best_score = np.inf

    # Elite record across all restart cycles -- this is what gets returned.
    best_position = np.zeros(dim)
    best_score = np.inf

    iterations_since_improvement = 0
    unproductive_restarts = 0
    history: List[float] = []

    for t in range(max_iterations):
        # 1. Evaluate fitness, update personal_best / global_best.
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

        # Restart-on-stagnation (see module docstring). The swarm is cleared
        # entirely, global_best included, so the next cycle is an untouched
        # QPSO run rather than one still anchored to the stale incumbent.
        if iterations_since_improvement >= patience:
            if unproductive_restarts >= max_restarts:
                break
            positions, personal_best, personal_best_scores = fresh_swarm()
            global_best = np.zeros(dim)
            global_best_score = np.inf
            iterations_since_improvement = 0
            unproductive_restarts += 1
            continue
        iterations_since_improvement += 1

        # 2. mbest_d = mean personal best across the swarm, per dimension.
        mbest = np.mean(personal_best, axis=0)

        # 3. beta(t) -- the only thing that differs between variants.
        beta = beta_fn(t)

        # 4. Quantum position update, per particle per dimension.
        phi = rng.uniform(0.0, 1.0, (num_particles, dim))
        p = phi * personal_best + (1 - phi) * global_best

        u = rng.uniform(1e-12, 1.0, (num_particles, dim))
        k = rng.uniform(0.0, 1.0, (num_particles, dim))
        sign = np.where(k >= 0.5, 1.0, -1.0)

        positions = p + sign * beta * np.abs(mbest - positions) * np.log(1.0 / u)
        positions = np.clip(positions, bounds[0], bounds[1])

    if return_history:
        while len(history) < max_iterations:
            history.append(float(best_score))
        return best_position, best_score, np.asarray(history, dtype=float)

    return best_position, best_score


def fixed_beta_qpso(
    dim: int,
    fitness_fn: FitnessFn,
    num_particles: Optional[int] = None,
    max_iterations: Optional[int] = None,
    beta_max: float = 1.0,
    beta_min: float = 0.5,
    bounds: Tuple[float, float] = (0.0, 1.0),
    seed: Optional[int] = None,
    patience: int = 15,
    tol: float = 1e-6,
    max_restarts: Optional[int] = None,
    return_history: bool = False,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """
    Standard linear-anneal beta baseline, used throughout the QPSO literature
    to compare novel variants against (this is that baseline, not a
    strawman):

        beta(t) = beta_max - (beta_max - beta_min) * (t / max_iterations)

    beta depends only on iteration progress -- a property of the swarm's own
    schedule, not of anything external. See va_qpso below for the contrast.
    """
    num_particles, max_iterations, max_restarts = _resolve_budget(
        dim, num_particles, max_iterations, max_restarts)

    def beta_fn(t: int) -> float:
        return beta_max - (beta_max - beta_min) * (t / max_iterations)

    return _run_qpso(dim, fitness_fn, beta_fn, num_particles, max_iterations, bounds, seed, patience, tol, max_restarts, return_history=return_history)


# --- va_qpso: volatility-adaptive beta (this project's contribution) ------
#
# Prior adaptive-beta strategies in the QPSO literature -- iteration-count
# schedules (fixed_beta_qpso above), fitness-stagnation triggers, population-
# diversity measures -- all derive beta from the swarm's OWN internal search
# state: how far along the run is, whether gbest has stopped improving, how
# spread out the particles currently are. Every one of those quantities is
# computable from the optimizer alone, with no reference to the problem it
# is solving.
#
# va_qpso instead derives beta from a measurement of the EXTERNAL
# environment being optimized over: live traffic volatility on the road
# network (see volatility.NetworkVolatilityIndex), taken at the moment
# replan() is invoked. It has nothing to do with this swarm run's iteration
# count, fitness history, or particle spread -- a converged, static swarm
# and a freshly-initialized one get the same beta if network conditions are
# the same. That is the paper's actual contribution, so it must stay
# unambiguous in code and comments alike: no iteration/fitness/diversity
# term is allowed to leak into how beta is computed here.
def va_qpso(
    dim: int,
    fitness_fn: FitnessFn,
    volatility_index: float,
    num_particles: Optional[int] = None,
    max_iterations: Optional[int] = None,
    beta_max: float = 1.0,
    beta_min: float = 0.5,
    bounds: Tuple[float, float] = (0.0, 1.0),
    seed: Optional[int] = None,
    patience: int = 15,
    tol: float = 1e-6,
    max_restarts: Optional[int] = None,
    return_history: bool = False,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """
    beta = beta_min + (beta_max - beta_min) * volatility_index

    volatility_index must already be normalized to [0, 1] (e.g. from
    volatility.NetworkVolatilityIndex.update()) and is held fixed for the
    whole run: it is a snapshot of network conditions at replan time, not a
    per-iteration swarm-state signal.
    """
    if not 0.0 <= volatility_index <= 1.0:
        raise ValueError(f"volatility_index must be in [0, 1], got {volatility_index}")

    num_particles, max_iterations, max_restarts = _resolve_budget(
        dim, num_particles, max_iterations, max_restarts)
    beta = beta_min + (beta_max - beta_min) * volatility_index

    def beta_fn(t: int) -> float:
        return beta

    return _run_qpso(dim, fitness_fn, beta_fn, num_particles, max_iterations, bounds, seed, patience, tol, max_restarts, return_history=return_history)


def replan(
    stops: List[str],
    distance_matrix: np.ndarray,
    congestion_lookup: CongestionLookup,
    volatility_index: float = 0.5,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_particles: Optional[int] = None,
    max_iterations: Optional[int] = None,
    beta_max: float = 1.0,
    beta_min: float = 0.5,
    seed: Optional[int] = None,
    patience: int = 15,
    tol: float = 1e-6,
    max_restarts: Optional[int] = None,
    algorithm: str = "va_qpso",
    return_history: bool = False,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """
    Run QPSO to convergence (see module docstring for the stopping
    criterion) on a frozen re-plan snapshot, and return the best stop order
    found plus its fitness score.

    Args:
        stops: Stop identifiers, in the order they map to distance_matrix's
            rows/columns.
        distance_matrix: (n, n) live-weighted travel-time matrix for this
            snapshot, e.g. from qpso_encoding.compute_distance_matrix().
        congestion_lookup: Per-leg edge occupancy/capacity data (see
            fitness.CongestionLookup).
        volatility_index: Current network volatility in [0, 1], e.g. from
            volatility.NetworkVolatilityIndex.update(), used when algorithm="va_qpso".
        weights: (w1, w2, w3) passed through to score_route for (T, D, C).
        algorithm: "va_qpso" (volatility-adaptive) or "fixed_beta_qpso" (linear anneal).

    Returns:
        (best_order, best_score): best_order is the decoded visit-order
        permutation (indices into distance_matrix / stops), best_score is
        its score_route() fitness.
    """
    n = len(stops)
    if distance_matrix.shape != (n, n):
        raise ValueError(
            f"distance_matrix shape {distance_matrix.shape} does not match len(stops)={n}."
        )

    def fitness_fn(x: np.ndarray) -> float:
        order = decode_order(x)
        return score_route(order, distance_matrix, congestion_lookup, weights)

    if algorithm == "va_qpso":
        res = va_qpso(
            dim=n,
            fitness_fn=fitness_fn,
            volatility_index=volatility_index,
            num_particles=num_particles,
            max_iterations=max_iterations,
            beta_max=beta_max,
            beta_min=beta_min,
            seed=seed,
            patience=patience,
            tol=tol,
            max_restarts=max_restarts,
            return_history=return_history,
        )
    elif algorithm == "fixed_beta_qpso":
        res = fixed_beta_qpso(
            dim=n,
            fitness_fn=fitness_fn,
            num_particles=num_particles,
            max_iterations=max_iterations,
            beta_max=beta_max,
            beta_min=beta_min,
            seed=seed,
            patience=patience,
            tol=tol,
            max_restarts=max_restarts,
            return_history=return_history,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Must be 'va_qpso' or 'fixed_beta_qpso'.")

    if return_history:
        best_pos, best_score, history = res
        return decode_order(best_pos), best_score, history

    best_pos, best_score = res
    return decode_order(best_pos), best_score
