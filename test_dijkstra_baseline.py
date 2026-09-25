"""
Unit and integration tests for Dijkstra Nearest-Neighbor baseline (dijkstra_baseline.py).

Verifies:
1. Permutation validity: returns a strictly valid permutation of stop indices {0, ..., n-1}.
2. Labeling integrity: ALGORITHM_LABEL is strictly "Dijkstra (nearest-neighbor heuristic)".
3. Start index adherence: chosen start_idx is always visited first (order[0] == start_idx).
4. Multi-start heuristic: all_starts finds a route with score <= single-start route.
5. Graph vs distance matrix equivalence: direct repeated Dijkstra on graph matches matrix lookup.
6. Naive order baseline: dijkstra_naive visits stops in input order [0, 1, ..., n-1].
7. Determinism: identical inputs produce strictly identical solution trajectories.
8. Replan parity: replan() works as a drop-in replacement with exact signature compatibility.
9. Edge case safety: handles 0 stops, 1 stop, and validates dimensional bounds.
"""

import itertools
import pytest
import numpy as np

from dijkstra_baseline import (
    ALGORITHM_LABEL,
    ALL_STARTS_LABEL,
    NAIVE_LABEL,
    dijkstra_all_starts_nearest_neighbor,
    dijkstra_from_graph,
    dijkstra_naive,
    dijkstra_nearest_neighbor,
    replan,
    score_route,
)
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
    return graph, stops, dist_mat, true_opt


def test_labeling_integrity():
    """Verify standard baseline labeling requirements for academic publication."""
    assert ALGORITHM_LABEL == "Dijkstra (nearest-neighbor heuristic)"
    assert NAIVE_LABEL == "Dijkstra (naive input order)"
    assert ALL_STARTS_LABEL == "Dijkstra (multi-start nearest-neighbor heuristic)"


def test_permutation_validity(delhi_problem):
    """Verify Dijkstra nearest-neighbor returns a valid permutation of {0, ..., n-1}."""
    _, stops, dist_mat, _ = delhi_problem
    n = len(stops)

    order, score = dijkstra_nearest_neighbor(stops, dist_mat, CONGESTION_LOOKUP, WEIGHTS, start_idx=0)

    assert len(order) == n
    assert sorted(order.tolist()) == list(range(n)), f"Invalid permutation: {order}"
    assert np.isfinite(score)
    assert score > 0.0


def test_start_index_adherence(delhi_problem):
    """Verify that start_idx is always the first stop in the output order."""
    _, stops, dist_mat, _ = delhi_problem
    n = len(stops)

    for s in range(n):
        order, score = dijkstra_nearest_neighbor(stops, dist_mat, CONGESTION_LOOKUP, WEIGHTS, start_idx=s)
        assert order[0] == s, f"Expected route to start at {s}, but started at {order[0]}"
        assert sorted(order.tolist()) == list(range(n))


def test_multi_start_nearest_neighbor(delhi_problem):
    """Verify that all-starts nearest-neighbor score <= single-start nearest-neighbor score."""
    _, stops, dist_mat, _ = delhi_problem

    order_0, score_0 = dijkstra_nearest_neighbor(stops, dist_mat, CONGESTION_LOOKUP, WEIGHTS, start_idx=0)
    best_order, best_score = dijkstra_all_starts_nearest_neighbor(stops, dist_mat, CONGESTION_LOOKUP, WEIGHTS)

    assert len(best_order) == len(stops)
    assert sorted(best_order.tolist()) == list(range(len(stops)))
    assert best_score <= score_0 + 1e-9, f"All-starts score {best_score} should be <= single start {score_0}"


def test_naive_input_order(delhi_problem):
    """Verify naive order visits stops in the exact sequence given [0, 1, ..., n-1]."""
    _, stops, dist_mat, _ = delhi_problem
    n = len(stops)

    order, score = dijkstra_naive(stops, dist_mat, CONGESTION_LOOKUP, WEIGHTS)

    assert np.array_equal(order, np.arange(n))
    expected_score = score_route(np.arange(n), dist_mat, CONGESTION_LOOKUP, WEIGHTS)
    assert np.isclose(score, expected_score)


def test_graph_vs_distance_matrix_equivalence(delhi_problem):
    """
    Verify that executing repeated Dijkstra directly on the NetworkGraph
    produces the identical route and score as precomputed distance matrix lookup.
    """
    graph, stops, dist_mat, _ = delhi_problem

    order_mat, score_mat = dijkstra_nearest_neighbor(stops, dist_mat, CONGESTION_LOOKUP, WEIGHTS, start_idx=0)
    order_graph, score_graph = dijkstra_from_graph(
        network_graph=graph,
        stops=stops,
        edge_weights={},
        congestion_lookup=CONGESTION_LOOKUP,
        weights=WEIGHTS,
        start_idx=0,
    )

    np.testing.assert_array_equal(order_mat, order_graph)
    assert np.isclose(score_mat, score_graph, rtol=1e-6)


def test_determinism(delhi_problem):
    """Verify Dijkstra nearest neighbor is purely deterministic."""
    _, stops, dist_mat, _ = delhi_problem

    o1, s1 = dijkstra_nearest_neighbor(stops, dist_mat, CONGESTION_LOOKUP, WEIGHTS, start_idx=0)
    o2, s2 = dijkstra_nearest_neighbor(stops, dist_mat, CONGESTION_LOOKUP, WEIGHTS, start_idx=0)

    np.testing.assert_array_equal(o1, o2)
    assert s1 == s2


def test_replan_interface_parity(delhi_problem):
    """Verify replan() functions as a polymorphic drop-in replacement with QPSO/PSO/GA."""
    _, stops, dist_mat, _ = delhi_problem

    # Default call (start_idx=0)
    order, score = replan(
        stops=stops,
        distance_matrix=dist_mat,
        congestion_lookup=CONGESTION_LOOKUP,
        volatility_index=0.8,
        weights=WEIGHTS,
    )
    assert len(order) == len(stops)
    assert sorted(order.tolist()) == list(range(len(stops)))

    # Swallowing unused metaheuristic arguments
    order_kw, score_kw = replan(
        stops=stops,
        distance_matrix=dist_mat,
        congestion_lookup=CONGESTION_LOOKUP,
        volatility_index=0.5,
        weights=WEIGHTS,
        seed=42,
        population_size=50,
        max_iterations=1000,
        c1=1.5,
        crossover_rate=0.9,
    )
    np.testing.assert_array_equal(order, order_kw)
    assert score == score_kw

    # All starts mode
    order_all, score_all = replan(
        stops=stops,
        distance_matrix=dist_mat,
        congestion_lookup=CONGESTION_LOOKUP,
        weights=WEIGHTS,
        all_starts=True,
    )
    assert score_all <= score + 1e-9


def test_edge_cases():
    """Verify edge cases: 0 stops, 1 stop, and shape validation."""
    # 0 stops
    empty_stops: list = []
    empty_mat = np.zeros((0, 0))
    order_0, score_0 = dijkstra_nearest_neighbor(empty_stops, empty_mat)
    assert len(order_0) == 0
    assert score_0 == 0.0

    # 1 stop
    single_stops = ["node_A"]
    single_mat = np.zeros((1, 1))
    order_1, score_1 = dijkstra_nearest_neighbor(single_stops, single_mat)
    assert order_1.tolist() == [0]
    assert score_1 == 0.0

    # Shape mismatch
    with pytest.raises(ValueError):
        dijkstra_nearest_neighbor(["a", "b"], np.zeros((3, 3)))

    # Out of bounds start index
    with pytest.raises(IndexError):
        dijkstra_nearest_neighbor(["a", "b"], np.zeros((2, 2)), start_idx=5)
