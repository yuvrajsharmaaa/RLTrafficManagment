"""
Test script comparing fixed_beta_qpso and va_qpso (src/planner/qpso.py) on
the real Delhi network.

Both variants share the exact same core swarm loop (Sun, Feng & Xu, 2004)
and differ only in how beta is computed (see qpso.py's module docstring).
This exercises both with NO explicit num_particles/max_iterations/
max_restarts, so it is exactly what a caller gets by default -- the budget
is scaled off dim by qpso.default_budget(). Checks:
  - both reach the TRUE optimum (brute-forced over all 8! = 40320 orderings,
    feasible at this size), not just "better than the naive identity order"
  - va_qpso's beta actually responds to volatility_index -- a volatility of
    0.0 vs 1.0 (anneal floor 0.5 vs 0.75) changes the swarm's trajectory
    under an otherwise identical seed.
    Asserted on the gbest position, not the score: with the scaled default
    budget both settings reliably reach the same (optimal) score, so score
    equality is expected and is not evidence beta was ignored -- only
    trajectory divergence is.
  - replan() is equivalent to calling va_qpso directly with a
    score_route-based fitness function
"""

import itertools

import numpy as np

from src.planner.fitness import score_route
from src.planner.qpso import default_budget, fixed_beta_qpso, replan, va_qpso
from src.planner.qpso_encoding import (
    adjacency_from_network_graph,
    compute_distance_matrix,
    decode_order,
    pick_mutually_reachable_stops,
    tour_length,
)
from src.state_extraction.network_graph import NetworkGraph

NET_FILE = "networks/delhi/delhi_intersection.net.xml"
NUM_STOPS = 8
CONGESTION_LOOKUP = {}
WEIGHTS = (1.0, 1.0, 1.0)

_network_graph = NetworkGraph(NET_FILE)
_adjacency = adjacency_from_network_graph(_network_graph, edge_weights={})
STOPS = pick_mutually_reachable_stops(_adjacency, NUM_STOPS)
DISTANCE_MATRIX = compute_distance_matrix(_adjacency, STOPS)
assert np.all(np.isfinite(DISTANCE_MATRIX)), "expected all stops to be mutually reachable"

IDENTITY_DISTANCE = tour_length(np.arange(NUM_STOPS), DISTANCE_MATRIX)


def _objective(order) -> float:
    return score_route(np.asarray(order), DISTANCE_MATRIX, CONGESTION_LOOKUP, WEIGHTS)


def _fitness_fn(x: np.ndarray) -> float:
    return _objective(decode_order(x))


# Brute-forced once (8! = 40320, sub-second) so tests check against the true
# optimum rather than just "beat the naive identity order".
_TRUE_OPTIMUM_SCORE = min(_objective(order) for order in itertools.permutations(range(NUM_STOPS)))


def test_fixed_beta_qpso_beats_identity():
    best_position, best_score = fixed_beta_qpso(
        dim=NUM_STOPS,
        fitness_fn=_fitness_fn,
        seed=42,
    )
    best_order = decode_order(best_position)

    print(f"fixed_beta_qpso (default_budget={default_budget(NUM_STOPS)}): "
          f"order={best_order.tolist()} score={best_score:.4f} "
          f"(true optimum={_TRUE_OPTIMUM_SCORE:.4f}, identity={IDENTITY_DISTANCE:.2f})")

    assert sorted(best_order.tolist()) == list(range(NUM_STOPS))
    assert np.isfinite(best_score)
    assert best_score <= tour_length(np.arange(NUM_STOPS), DISTANCE_MATRIX)
    assert np.isclose(best_score, _TRUE_OPTIMUM_SCORE, rtol=1e-6)


def test_va_qpso_beats_identity():
    best_position, best_score = va_qpso(
        dim=NUM_STOPS,
        fitness_fn=_fitness_fn,
        volatility_index=0.5,
        seed=42,
    )
    best_order = decode_order(best_position)

    print(f"va_qpso (volatility=0.5, default_budget={default_budget(NUM_STOPS)}): "
          f"order={best_order.tolist()} score={best_score:.4f} "
          f"(true optimum={_TRUE_OPTIMUM_SCORE:.4f})")

    assert sorted(best_order.tolist()) == list(range(NUM_STOPS))
    assert np.isfinite(best_score)
    assert np.isclose(best_score, _TRUE_OPTIMUM_SCORE, rtol=1e-6)


def test_va_qpso_beta_responds_to_volatility_index():
    """
    va_qpso anneals beta from beta_max toward beta_min + 0.25 * volatility_index.
    With the same seed and fitness landscape, volatility_index=0.0 (floor 0.5)
    and volatility_index=1.0 (floor 0.75) scale every position update after
    t = 0 differently, so the swarm must traverse a different trajectory.

    Asserted on the returned gbest POSITION, not the score: with the scaled
    default budget both settings reliably converge to the SAME (optimal)
    score (see test_va_qpso_beats_identity), so equal scores here are
    expected, not a sign beta was ignored. Only the position -- the actual
    search trajectory -- is evidence one way or the other.
    """
    position_low, score_low = va_qpso(
        dim=NUM_STOPS, fitness_fn=_fitness_fn, volatility_index=0.0, seed=123,
    )
    position_high, score_high = va_qpso(
        dim=NUM_STOPS, fitness_fn=_fitness_fn, volatility_index=1.0, seed=123,
    )

    print(f"va_qpso volatility=0.0 -> score={score_low:.4f}; "
          f"volatility=1.0 -> score={score_high:.4f} "
          f"(same score: {np.isclose(score_low, score_high)}, "
          f"same trajectory: {np.allclose(position_low, position_high)})")

    assert not np.allclose(position_low, position_high), (
        "expected different beta (from different volatility_index) to change "
        "the swarm's trajectory under an identical seed"
    )


def test_va_qpso_rejects_out_of_range_volatility_index():
    try:
        va_qpso(dim=NUM_STOPS, fitness_fn=_fitness_fn, volatility_index=1.5, seed=1)
        assert False, "expected ValueError for out-of-range volatility_index"
    except ValueError:
        pass


def test_replan_matches_va_qpso_via_score_route():
    """
    replan() should be equivalent to calling va_qpso directly with a
    score_route-based fitness function (empty congestion_lookup, default
    equal weights) -- same seed, same result, both under the same default
    (dim-scaled) budget.
    """
    order_a, score_a = replan(
        STOPS, DISTANCE_MATRIX, CONGESTION_LOOKUP, volatility_index=0.5, seed=99,
    )
    best_position, score_b = va_qpso(
        dim=NUM_STOPS, fitness_fn=_fitness_fn, volatility_index=0.5, seed=99,
    )
    order_b = decode_order(best_position)

    print(f"replan() score={score_a:.4f} vs direct va_qpso+score_route score={score_b:.4f}")

    assert order_a.tolist() == order_b.tolist()
    assert np.isclose(score_a, score_b)


if __name__ == "__main__":
    test_fixed_beta_qpso_beats_identity()
    test_va_qpso_beats_identity()
    test_va_qpso_beta_responds_to_volatility_index()
    test_va_qpso_rejects_out_of_range_volatility_index()
    test_replan_matches_va_qpso_via_score_route()
    print("OK: all qpso.py tests passed.")
