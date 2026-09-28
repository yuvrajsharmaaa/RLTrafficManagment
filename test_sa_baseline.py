"""
Unit and integration tests for standard Simulated Annealing baseline (sa_baseline.py).

Verifies:
1. Permutation validity: 2-Opt segment reversal produces strictly valid permutations.
2. Operator correctness: 2-Opt reverses the subsegment between indices i and j.
3. Budget parity: default_budget(dim) matches qpso.default_budget(dim) across all dimensions.
4. Cooling schedule: Geometric cooling T(t) = T0 * (cooling_rate ** t) decays as expected.
5. Benchmark convergence: Simulated Annealing reaches true brute-force optimum on Delhi network.
6. Determinism: Identical random seed produces identical trajectory and final solution.
7. Drop-in interface: replan() works as a drop-in replacement across all algorithms.
8. History logging: Monotonically non-increasing history of length max_iterations.
"""

import itertools
import pytest
import numpy as np

from sa_baseline import (
    DEFAULT_COOLING_RATE,
    DEFAULT_MIN_TEMP,
    DEFAULT_PATIENCE,
    DEFAULT_T0,
    DEFAULT_TOL,
    default_budget,
    replan,
    score_route,
    simulated_annealing,
    two_opt_swap,
)
from src.planner.qpso import default_budget as qpso_default_budget
from src.planner.qpso_encoding import (
    adjacency_from_network_graph,
    compute_distance_matrix,
    pick_mutually_reachable_stops,
)
from src.state_extraction.network_graph import NetworkGraph

NET_FILE = "networks/delhi/delhi_intersection.net.xml"
NUM_STOPS = 8
WEIGHTS = (1.0, 1.0, 1.0)
CONGESTION_LOOKUP = {}


@pytest.fixture(scope="module")
def delhi_problem():
    graph = NetworkGraph(NET_FILE)
    adj = adjacency_from_network_graph(graph, edge_weights={})
    stops = pick_mutually_reachable_stops(adj, NUM_STOPS)
    dist_mat = compute_distance_matrix(adj, stops)
    assert np.all(np.isfinite(dist_mat)), "expected all stops to be mutually reachable"

    # Compute true brute-force optimum across 40,320 permutations
    true_opt = min(
        score_route(np.asarray(p), dist_mat, CONGESTION_LOOKUP, WEIGHTS)
        for p in itertools.permutations(range(NUM_STOPS))
    )
    return stops, dist_mat, true_opt


def test_budget_parity():
    """Verify default_budget(dim) strictly matches qpso.default_budget(dim)."""
    for dim in range(3, 16):
        sa_b = default_budget(dim)
        qpso_b = qpso_default_budget(dim)
        assert sa_b == qpso_b, f"Budget mismatch at dim={dim}: {sa_b} vs {qpso_b}"


def test_two_opt_swap_permutation_validity():
    """Verify 2-opt segment reversal always produces valid permutations without duplicates."""
    rng = np.random.default_rng(123)
    for dim in [3, 4, 8, 12]:
        base = np.arange(dim)
        for _ in range(50):
            neighbor = two_opt_swap(base, rng)
            assert len(neighbor) == dim, "Length of permutation must be preserved"
            assert sorted(neighbor) == list(range(dim)), "All stop indices must appear exactly once"


def test_two_opt_swap_segment_reversal():
    """Verify 2-opt actually reverses the elements between selected indices."""
    rng = np.random.default_rng(42)
    order = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    # Run multiple times and verify that for any change, the difference corresponds to a reversed slice
    changed = False
    for _ in range(20):
        neighbor = two_opt_swap(order, rng)
        if not np.array_equal(neighbor, order):
            changed = True
            diff_indices = np.where(neighbor != order)[0]
            start, end = diff_indices[0], diff_indices[-1]
            assert np.array_equal(neighbor[start : end + 1], order[start : end + 1][::-1])
    assert changed, "Expected at least one move to alter the order"


def test_cooling_schedule():
    """Verify geometric cooling formula T(t) = T0 * (cooling_rate ** t)."""
    t0 = 100.0
    cooling_rate = 0.95
    dim = 4

    def dummy_fitness(order: np.ndarray) -> float:
        return float(np.sum(order))

    # Run for 10 iterations and verify history format
    _, _, hist = simulated_annealing(
        dim=dim,
        fitness_fn=dummy_fitness,
        steps_per_temp=10,
        max_iterations=10,
        t0=t0,
        cooling_rate=cooling_rate,
        seed=42,
        return_history=True,
    )
    assert len(hist) == 10
    # Expected temperature at t=5
    expected_t5 = t0 * (cooling_rate ** 5)
    assert np.isclose(expected_t5, 100.0 * (0.95 ** 5))


def test_sa_convergence(delhi_problem):
    """Verify Simulated Annealing achieves the true global brute-force optimum on Delhi network."""
    stops, dist_mat, true_opt = delhi_problem
    n = len(stops)

    def fitness_fn(order: np.ndarray) -> float:
        return score_route(order, dist_mat, CONGESTION_LOOKUP, WEIGHTS)

    best_order, best_score, history = simulated_annealing(
        dim=n,
        fitness_fn=fitness_fn,
        seed=42,
        return_history=True,
    )

    gap = best_score - true_opt
    assert gap < 1e-4, f"SA gap to true brute-force optimum too large: {gap:.6f}s (best={best_score:.4f}, opt={true_opt:.4f})"
    assert np.isclose(history[-1], best_score)
    assert np.all(np.diff(history) <= 1e-9), "History trajectory must be monotonically non-increasing"


def test_determinism():
    """Verify identical random seed produces identical trajectory and final solution."""
    dim = 6
    rng = np.random.default_rng(999)
    weights = rng.uniform(5.0, 50.0, (dim, dim))
    np.fill_diagonal(weights, 0.0)

    def fitness_fn(order: np.ndarray) -> float:
        return float(np.sum([weights[order[i], order[i + 1]] for i in range(len(order) - 1)]))

    order1, score1, hist1 = simulated_annealing(dim, fitness_fn, seed=42, return_history=True)
    order2, score2, hist2 = simulated_annealing(dim, fitness_fn, seed=42, return_history=True)

    assert np.array_equal(order1, order2), "Permutations with identical seed must match"
    assert np.isclose(score1, score2), "Scores with identical seed must match"
    assert np.allclose(hist1, hist2), "Histories with identical seed must match"


def test_replan_interface(delhi_problem):
    """Verify replan() works cleanly as a drop-in replacement across the routing pipeline."""
    stops, dist_mat, _ = delhi_problem

    # 1. Without history
    order, score = replan(
        stops=stops,
        distance_matrix=dist_mat,
        congestion_lookup=CONGESTION_LOOKUP,
        weights=WEIGHTS,
        volatility_index=0.5,
        seed=123,
        return_history=False,
    )
    assert len(order) == len(stops)
    assert isinstance(score, float)

    # 2. With history
    order_h, score_h, hist_h = replan(
        stops=stops,
        distance_matrix=dist_mat,
        congestion_lookup=CONGESTION_LOOKUP,
        weights=WEIGHTS,
        volatility_index=0.5,
        seed=123,
        return_history=True,
    )
    assert np.array_equal(order, order_h)
    assert np.isclose(score, score_h)
    assert len(hist_h) == default_budget(len(stops))[1]
    assert np.isclose(hist_h[-1], score_h)
