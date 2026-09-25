"""
Unit and integration tests for convergence logging and metrics across all planners.

Verifies:
1. History logging: va_qpso, fixed_beta_qpso, standard_pso, and ga return
   history arrays of length max_iterations with non-increasing values.
2. Final value consistency: history[-1] == best_score across all algorithms.
3. Dijkstra baseline trajectory: returns flat array of length max_iterations.
4. Replan interface consistency: replan(..., return_history=True) returns
   (order, score, history) across all five methods.
5. Convergence metric calculation:
   - Iterations to 95% of run's own best fitness.
   - Hit rate (% of runs within tolerance of reference best).
"""

import numpy as np
import pytest

from src.planner.qpso import fixed_beta_qpso, va_qpso, replan as qpso_replan
from src.planner.pso_baseline import standard_pso, replan as pso_replan
from src.planner.ga_baseline import genetic_algorithm, replan as ga_replan
from src.planner.dijkstra_baseline import dijkstra_nearest_neighbor, replan as dijkstra_replan
from src.planner.qpso_encoding import decode_order


def sphere_fitness(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def perm_fitness(order: np.ndarray) -> float:
    target = np.arange(len(order))
    return float(np.sum((order - target)**2))


def compute_iterations_to_95(history: np.ndarray) -> int:
    """
    Iterations to reach 95% of that run's own best-found improvement.

    f_0 = history[0]
    f_star = history[-1]
    Target = f_0 - 0.95 * (f_0 - f_star) = f_star + 0.05 * (f_0 - f_star)
    """
    if len(history) == 0:
        return 0
    f_0 = history[0]
    f_star = history[-1]
    delta = f_0 - f_star
    if delta <= 1e-12:
        return 0

    target = f_star + 0.05 * delta
    indices = np.where(history <= target + 1e-12)[0]
    return int(indices[0]) if len(indices) > 0 else len(history) - 1


def compute_hit_rate(scores: np.ndarray, ref_best: float, tol: float = 1e-4) -> float:
    """Percentage of trials that reach within `tol` of reference best fitness."""
    hits = np.sum(scores <= ref_best + tol)
    return float((hits / len(scores)) * 100.0) if len(scores) > 0 else 0.0


def test_qpso_history_logging():
    """Verify VA-QPSO and Fixed-Beta QPSO log valid per-iteration history."""
    dim = 4
    max_iter = 120

    # 1. va_qpso
    pos_va, score_va, hist_va = va_qpso(
        dim=dim,
        fitness_fn=sphere_fitness,
        volatility_index=0.6,
        max_iterations=max_iter,
        seed=101,
        return_history=True,
    )
    assert len(hist_va) == max_iter
    assert np.isclose(hist_va[-1], score_va)
    assert np.all(np.diff(hist_va) <= 1e-9), "VA-QPSO history must be non-increasing"

    # 2. fixed_beta_qpso
    pos_fb, score_fb, hist_fb = fixed_beta_qpso(
        dim=dim,
        fitness_fn=sphere_fitness,
        max_iterations=max_iter,
        seed=102,
        return_history=True,
    )
    assert len(hist_fb) == max_iter
    assert np.isclose(hist_fb[-1], score_fb)
    assert np.all(np.diff(hist_fb) <= 1e-9), "Fixed-Beta QPSO history must be non-increasing"


def test_pso_baseline_history_logging():
    """Verify Standard PSO logs valid per-iteration history."""
    dim = 4
    max_iter = 150

    pos, score, hist = standard_pso(
        dim=dim,
        fitness_fn=sphere_fitness,
        max_iterations=max_iter,
        seed=201,
        return_history=True,
    )
    assert len(hist) == max_iter
    assert np.isclose(hist[-1], score)
    assert np.all(np.diff(hist) <= 1e-9), "Standard PSO history must be non-increasing"


def test_ga_baseline_history_logging():
    """Verify Permutation GA logs valid per-generation history."""
    dim = 4
    max_gen = 100

    order, score, hist = genetic_algorithm(
        dim=dim,
        fitness_fn=perm_fitness,
        max_generations=max_gen,
        seed=301,
        return_history=True,
    )
    assert len(hist) == max_gen
    assert np.isclose(hist[-1], score)
    assert np.all(np.diff(hist) <= 1e-9), "GA history must be non-increasing"


def test_dijkstra_baseline_history_logging():
    """Verify Dijkstra Nearest-Neighbor returns flat trajectory of requested iteration length."""
    stops = ["A", "B", "C", "D"]
    dist_mat = np.array([
        [0.0, 10.0, 20.0, 30.0],
        [10.0, 0.0, 15.0, 25.0],
        [20.0, 15.0, 0.0, 10.0],
        [30.0, 25.0, 10.0, 0.0],
    ])
    max_iter = 250

    order, score, hist = dijkstra_nearest_neighbor(
        stops=stops,
        distance_matrix=dist_mat,
        return_history=True,
        max_iterations=max_iter,
    )
    assert len(hist) == max_iter
    assert np.all(hist == score), "Dijkstra history should be constant across iterations"


def test_replan_history_parity():
    """Verify replan(..., return_history=True) returns 3-tuple across all algorithms."""
    stops = ["A", "B", "C", "D"]
    dist_mat = np.array([
        [0.0, 10.0, 20.0, 30.0],
        [10.0, 0.0, 15.0, 25.0],
        [20.0, 15.0, 0.0, 10.0],
        [30.0, 25.0, 10.0, 0.0],
    ])
    congestion_lookup = {}
    max_iter = 80

    # 1. VA-QPSO
    o1, s1, h1 = qpso_replan(
        stops, dist_mat, congestion_lookup, volatility_index=0.5,
        max_iterations=max_iter, return_history=True, algorithm="va_qpso"
    )
    assert len(h1) == max_iter
    assert np.isclose(h1[-1], s1)

    # 2. Fixed-Beta QPSO
    o2, s2, h2 = qpso_replan(
        stops, dist_mat, congestion_lookup, volatility_index=0.5,
        max_iterations=max_iter, return_history=True, algorithm="fixed_beta_qpso"
    )
    assert len(h2) == max_iter
    assert np.isclose(h2[-1], s2)

    # 3. Standard PSO
    o3, s3, h3 = pso_replan(
        stops, dist_mat, congestion_lookup,
        max_iterations=max_iter, return_history=True
    )
    assert len(h3) == max_iter
    assert np.isclose(h3[-1], s3)

    # 4. GA
    o4, s4, h4 = ga_replan(
        stops, dist_mat, congestion_lookup,
        max_generations=max_iter, return_history=True
    )
    assert len(h4) == max_iter
    assert np.isclose(h4[-1], s4)

    # 5. Dijkstra NN
    o5, s5, h5 = dijkstra_replan(
        stops, dist_mat, congestion_lookup,
        max_iterations=max_iter, return_history=True
    )
    assert len(h5) == max_iter
    assert np.all(h5 == s5)


def test_convergence_speed_metric():
    """Verify compute_iterations_to_95 correctly identifies 95% improvement index."""
    # Synthetic trajectory starting at 100 and reaching 0: target = 0 + 0.05 * 100 = 5.0
    hist = np.array([100.0, 80.0, 50.0, 20.0, 5.0, 2.0, 0.0])
    idx = compute_iterations_to_95(hist)
    assert idx == 4  # hist[4] == 5.0 <= target

    # Flat history
    flat_hist = np.array([50.0, 50.0, 50.0])
    assert compute_iterations_to_95(flat_hist) == 0


def test_hit_rate_metric():
    """Verify compute_hit_rate calculates percentage within tolerance."""
    scores = np.array([10.0, 10.00005, 10.0001, 10.5, 12.0])
    ref = 10.0

    hit_rate = compute_hit_rate(scores, ref, tol=1e-4)
    # scores[0], scores[1], scores[2] <= 10.0001 -> 3 out of 5 = 60.0%
    assert np.isclose(hit_rate, 60.0)
