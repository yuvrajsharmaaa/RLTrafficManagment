"""
Unit tests for src/planner/fitness.py, with a hand-checkable 4-stop case
and confirmation of decoupled travel time (T) and physical distance (D)
on the chaotic traffic tier.

Route: order = [0, 1, 2, 3] (open path, 3 legs: 0->1, 1->2, 2->3).

Travel-time matrix (T) leg values (seconds):
    leg 0->1: 10.0 s
    leg 1->2: 20.0 s
    leg 2->3: 15.0 s
  => T = 10 + 20 + 15 = 45.0 s

Physical-distance matrix (D) leg values (meters, along the time-optimal path):
    leg 0->1: 100.0 m
    leg 1->2: 250.0 m
    leg 2->3: 150.0 m
  => D = 100 + 250 + 150 = 500.0 m

congestion_lookup (occupancy/capacity per edge, squared, summed):
    leg 0->1: one edge,  occ=5,  cap=10 -> (0.5)^2 = 0.25
    leg 1->2: one edge,  occ=8,  cap=10 -> (0.8)^2 = 0.64
    leg 2->3: two edges, occ=2,  cap=10 -> (0.2)^2 = 0.04
                         occ=9,  cap=10 -> (0.9)^2 = 0.81
  => C = 0.25 + 0.64 + 0.04 + 0.81 = 1.74

weights = (0.5, 0.3, 0.2) (already sums to 1)
  => Decoupled score = 0.5*45.0 + 0.3*500.0 + 0.2*1.74 = 22.5 + 150.0 + 0.348 = 172.848
  => Legacy score (when D defaults to T) = 0.5*45.0 + 0.3*45.0 + 0.2*1.74 = 36.348
"""

from pathlib import Path
import numpy as np
import pytest

from experiment import build_scenario_distance_and_congestion
from src.planner.fitness import route_components, score_route
from src.planner.qpso_encoding import (
    adjacency_from_network_graph,
    compute_travel_and_distance_matrices,
    pick_mutually_reachable_stops,
)
from src.state_extraction.network_graph import NetworkGraph

ORDER = np.array([0, 1, 2, 3])

DECOY = 999.0
TIME_MATRIX = np.array([
    [0.0,   10.0,  DECOY, DECOY],
    [10.0,  0.0,   20.0,  DECOY],
    [DECOY, 20.0,  0.0,   15.0],
    [DECOY, DECOY, 15.0,  0.0],
])

PHYSICAL_DISTANCE_MATRIX = np.array([
    [0.0,    100.0,  DECOY,  DECOY],
    [100.0,  0.0,    250.0,  DECOY],
    [DECOY,  250.0,  0.0,    150.0],
    [DECOY,  DECOY,  150.0,  0.0],
])

CONGESTION_LOOKUP = {
    (0, 1): [{"occupancy": 5.0, "capacity": 10.0}],
    (1, 2): [{"occupancy": 8.0, "capacity": 10.0}],
    (2, 3): [
        {"occupancy": 2.0, "capacity": 10.0},
        {"occupancy": 9.0, "capacity": 10.0},
    ],
}

EXPECTED_T = 45.0
EXPECTED_D_LEGACY = 45.0
EXPECTED_D_PHYSICAL = 500.0
EXPECTED_C = 1.74


def test_route_components_hand_checked():
    # 1. Legacy call (single matrix): T == D
    T_leg, D_leg, C_leg = route_components(ORDER, TIME_MATRIX, CONGESTION_LOOKUP)
    assert np.isclose(T_leg, EXPECTED_T)
    assert np.isclose(D_leg, EXPECTED_D_LEGACY)
    assert np.isclose(C_leg, EXPECTED_C)

    # 2. Decoupled call (positional both matrices): T != D
    T_dec, D_dec, C_dec = route_components(
        ORDER, TIME_MATRIX, PHYSICAL_DISTANCE_MATRIX, CONGESTION_LOOKUP
    )
    assert np.isclose(T_dec, EXPECTED_T)
    assert np.isclose(D_dec, EXPECTED_D_PHYSICAL)
    assert np.isclose(C_dec, EXPECTED_C)
    assert not np.isclose(T_dec, D_dec)

    # 3. Decoupled call (keyword distance_matrix)
    T_kw, D_kw, C_kw = route_components(
        ORDER, TIME_MATRIX, CONGESTION_LOOKUP, distance_matrix=PHYSICAL_DISTANCE_MATRIX
    )
    assert np.isclose(T_kw, EXPECTED_T)
    assert np.isclose(D_kw, EXPECTED_D_PHYSICAL)
    assert np.isclose(C_kw, EXPECTED_C)


def test_score_route_hand_checked():
    weights = (0.5, 0.3, 0.2)  # already sums to 1

    # 1. Legacy score (1 matrix): D defaults to T
    legacy_expected = 0.5 * EXPECTED_T + 0.3 * EXPECTED_D_LEGACY + 0.2 * EXPECTED_C
    assert np.isclose(legacy_expected, 36.348)
    score_leg = score_route(ORDER, TIME_MATRIX, CONGESTION_LOOKUP, weights)
    assert np.isclose(score_leg, legacy_expected)

    # 2. Decoupled score (positional both matrices): D is physical distance (500.0)
    decoupled_expected = 0.5 * EXPECTED_T + 0.3 * EXPECTED_D_PHYSICAL + 0.2 * EXPECTED_C
    assert np.isclose(decoupled_expected, 172.848)
    score_dec = score_route(
        ORDER, TIME_MATRIX, PHYSICAL_DISTANCE_MATRIX, CONGESTION_LOOKUP, weights
    )
    assert np.isclose(score_dec, decoupled_expected)

    # 3. Decoupled score (keyword distance_matrix)
    score_kw = score_route(
        ORDER, TIME_MATRIX, CONGESTION_LOOKUP, weights, distance_matrix=PHYSICAL_DISTANCE_MATRIX
    )
    assert np.isclose(score_kw, decoupled_expected)


def test_score_route_normalizes_unnormalized_weights():
    raw_weights = (5.0, 3.0, 2.0)
    normalized_weights = (0.5, 0.3, 0.2)

    # Legacy 1-matrix
    norm_score = score_route(ORDER, TIME_MATRIX, CONGESTION_LOOKUP, normalized_weights)
    raw_score = score_route(ORDER, TIME_MATRIX, CONGESTION_LOOKUP, raw_weights)
    assert np.isclose(raw_score, norm_score)

    # Decoupled both matrices
    norm_dec = score_route(
        ORDER, TIME_MATRIX, PHYSICAL_DISTANCE_MATRIX, CONGESTION_LOOKUP, normalized_weights
    )
    raw_dec = score_route(
        ORDER, TIME_MATRIX, PHYSICAL_DISTANCE_MATRIX, CONGESTION_LOOKUP, raw_weights
    )
    assert np.isclose(raw_dec, norm_dec)


def test_missing_leg_in_congestion_lookup_contributes_zero():
    lookup_without_last_leg = {
        (0, 1): CONGESTION_LOOKUP[(0, 1)],
        (1, 2): CONGESTION_LOOKUP[(1, 2)],
        # (2, 3) intentionally omitted
    }
    T, D, C = route_components(
        ORDER, TIME_MATRIX, PHYSICAL_DISTANCE_MATRIX, lookup_without_last_leg
    )
    assert np.isclose(T, EXPECTED_T)
    assert np.isclose(D, EXPECTED_D_PHYSICAL)
    assert np.isclose(C, 0.25 + 0.64)


def test_zero_sum_weights_raise():
    with pytest.raises(ValueError, match="weights must sum to a positive value"):
        score_route(
            ORDER, TIME_MATRIX, PHYSICAL_DISTANCE_MATRIX, CONGESTION_LOOKUP, (0.0, 0.0, 0.0)
        )


def test_t_ne_d_on_chaotic_tier():
    """
    Confirm that on the chaotic tier (delhi_intersection.net.xml under extreme volatility):
    1. Travel-time matrix T (in seconds) and physical-distance matrix D (in meters)
       along the time-optimal path are distinct: T != D.
    2. Route components evaluated on both matrices yield decoupled T and D values.
    3. score_route correctly weights both decoupled components.
    """
    net_file = "networks/delhi/delhi_intersection.net.xml"
    assert Path(net_file).exists(), f"Network file missing: {net_file}"

    network_graph = NetworkGraph(net_file)
    free_flow_adj = adjacency_from_network_graph(network_graph, edge_weights={})
    stops = pick_mutually_reachable_stops(free_flow_adj, 6)

    # Build chaotic tier matrices (volatility_index = 0.95)
    time_matrix, dist_matrix, cong_lookup, vol_idx = build_scenario_distance_and_congestion(
        network_graph=network_graph,
        stops=stops,
        tier="chaotic",
        seed=42,
        return_physical_distance=True,
    )

    # Assert matrix-level decoupling
    assert not np.allclose(time_matrix, dist_matrix), "Matrices T and D must not be identical!"

    # Evaluate route components for tour order
    order = np.arange(len(stops))
    T, D, C = route_components(order, time_matrix, dist_matrix, cong_lookup)

    assert T > 0.0, "Total travel time must be positive"
    assert D > 0.0, "Total physical distance must be positive"
    assert not np.isclose(T, D), f"T ({T:.2f}s) and D ({D:.2f}m) must not be equal on chaotic tier!"

    # Decoupled route fitness scoring
    weights = (0.4, 0.4, 0.2)
    score_pos = score_route(order, time_matrix, dist_matrix, cong_lookup, weights)
    score_kw = score_route(
        order, time_matrix, cong_lookup, weights, distance_matrix=dist_matrix
    )
    expected_score = 0.4 * T + 0.4 * D + 0.2 * C

    assert np.isclose(score_pos, expected_score)
    assert np.isclose(score_kw, expected_score)

    print(f"\n[Chaotic Tier Verified] Route T = {T:.2f}s, D = {D:.2f}m, C = {C:.4f}")
    print(f"[Chaotic Tier Verified] Decoupled fitness score = {score_pos:.4f}")


if __name__ == "__main__":
    test_route_components_hand_checked()
    test_score_route_hand_checked()
    test_score_route_normalizes_unnormalized_weights()
    test_missing_leg_in_congestion_lookup_contributes_zero()
    test_zero_sum_weights_raise()
    test_t_ne_d_on_chaotic_tier()
    print("\nOK: all test_fitness.py unit tests passed.")
