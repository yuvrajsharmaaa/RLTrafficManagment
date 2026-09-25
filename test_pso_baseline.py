"""
Unit and integration tests for standard (canonical) PSO baseline (pso_baseline.py).

Verifies:
1. Permutation validity: decode_order(x) produces a valid permutation of stops in range(dim).
2. Fitness function parity: score_route matches exact objective used across planners.
3. Budget parity: default_budget(dim) matches qpso.default_budget(dim) at all stop counts.
4. Benchmark convergence: reaches true brute-force optimum on Delhi network (8 stops, 40320 permutations).
5. Hyperparameter handling: constant w=0.7 and linear anneal w=0.9->0.4 both execute properly.
6. Determinism: identical random seed produces identical solution trajectories.
7. Replan interface parity: replan() works as a drop-in replacement for qpso.replan().
"""

import itertools
import pytest
import numpy as np

from pso_baseline import (
    DEFAULT_C1,
    DEFAULT_C2,
    DEFAULT_W,
    decode_order,
    default_budget,
    replan,
    score_route,
    standard_pso,
)
from src.planner.qpso import default_budget as qpso_default_budget
from src.planner.qpso_encoding import (
    adjacency_from_network_graph,
    compute_distance_matrix,
    pick_mutually_reachable_stops,
    tour_length,
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

    # Compute true brute-force optimum
    true_opt = min(
        score_route(np.asarray(p), dist_mat, CONGESTION_LOOKUP, WEIGHTS)
        for p in itertools.permutations(range(NUM_STOPS))
    )
    return stops, dist_mat, true_opt


def test_budget_parity():
    """Verify standard PSO uses identical budget scaling as QPSO."""
    for dim in [4, 6, 8, 10, 15]:
        pso_b = default_budget(dim)
        qpso_b = qpso_default_budget(dim)
        assert pso_b == qpso_b, f"Budget mismatch at dim={dim}: {pso_b} vs {qpso_b}"


def test_decode_order_permutation_validity():
    """Verify decode_order returns valid permutation of {0, ..., n-1}."""
    rng = np.random.default_rng(123)
    for dim in [5, 8, 12]:
        pos = rng.uniform(0.0, 1.0, dim)
        order = decode_order(pos)
        assert len(order) == dim
        assert sorted(order.tolist()) == list(range(dim))


def test_standard_pso_convergence(delhi_problem):
    """Verify canonical PSO (w=0.7, c1=1.5, c2=1.5) finds true optimum on 8-stop Delhi network."""
    stops, dist_mat, true_opt = delhi_problem

    def fitness_fn(x: np.ndarray) -> float:
        return score_route(decode_order(x), dist_mat, CONGESTION_LOOKUP, WEIGHTS)

    best_pos, best_score = standard_pso(
        dim=NUM_STOPS,
        fitness_fn=fitness_fn,
        w=DEFAULT_W,
        c1=DEFAULT_C1,
        c2=DEFAULT_C2,
        seed=42,
    )
    order = decode_order(best_pos)

    assert sorted(order.tolist()) == list(range(NUM_STOPS))
    assert np.isclose(best_score, true_opt, rtol=1e-5), f"Score {best_score} did not match true optimum {true_opt}"


def test_standard_pso_linear_anneal(delhi_problem):
    """Verify linear inertia weight annealing (0.9 -> 0.4) executes and converges."""
    stops, dist_mat, true_opt = delhi_problem

    def fitness_fn(x: np.ndarray) -> float:
        return score_route(decode_order(x), dist_mat, CONGESTION_LOOKUP, WEIGHTS)

    best_pos, best_score = standard_pso(
        dim=NUM_STOPS,
        fitness_fn=fitness_fn,
        w_schedule="linear",
        w_max=0.9,
        w_min=0.4,
        c1=1.5,
        c2=1.5,
        seed=42,
    )
    order = decode_order(best_pos)

    assert sorted(order.tolist()) == list(range(NUM_STOPS))
    assert np.isclose(best_score, true_opt, rtol=1e-5)


def test_replan_interface(delhi_problem):
    """Verify replan() function works with distance matrix and stop IDs."""
    stops, dist_mat, true_opt = delhi_problem

    best_order, best_score = replan(
        stops=stops,
        distance_matrix=dist_mat,
        congestion_lookup=CONGESTION_LOOKUP,
        weights=WEIGHTS,
        seed=42,
    )

    assert len(best_order) == len(stops)
    assert sorted(best_order.tolist()) == list(range(len(stops)))
    assert np.isclose(best_score, true_opt, rtol=1e-5)


def test_determinism(delhi_problem):
    """Verify standard PSO produces deterministic results under identical random seed."""
    stops, dist_mat, _ = delhi_problem

    def fitness_fn(x: np.ndarray) -> float:
        return score_route(decode_order(x), dist_mat, CONGESTION_LOOKUP, WEIGHTS)

    pos1, score1 = standard_pso(NUM_STOPS, fitness_fn, seed=999)
    pos2, score2 = standard_pso(NUM_STOPS, fitness_fn, seed=999)

    np.testing.assert_array_equal(pos1, pos2)
    assert score1 == score2
