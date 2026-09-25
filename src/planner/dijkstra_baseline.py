"""
Dijkstra Nearest-Neighbor Heuristic Baseline for Multi-Stop Route Planning.

Methodological Context & Academic Justification:
-----------------------------------------------
Dijkstra's algorithm (Dijkstra, 1959) solves the single-source shortest path (SSSP)
problem on a weighted graph in O(E + V log V). It finds the minimum-cost trajectory
between a single origin and one or all destination nodes.

Crucially, Dijkstra's algorithm alone does NOT solve combinatorial ordering or the
Traveling Salesperson / Vehicle Routing Problem (TSP / VRP). Plain Dijkstra computes
distances between nodes; it does not determine *which* stop to visit next when given
an unordered or multi-target set of delivery locations. Claiming that "plain Dijkstra"
optimizes or plans a multi-stop delivery tour is methodologically inaccurate in the
literature.

For an honest and fair baseline against heuristic and metaheuristic optimizers
(such as QPSO, Standard PSO, and Genetic Algorithms), two standard Dijkstra-based
baselines exist in the VRP / TSP literature:

1. Naive Input Order Baseline ("Dijkstra (naive input order)"):
   Visits stops in the caller-supplied sequence [0, 1, ..., n - 1], with each leg
   routed via Dijkstra shortest paths on the live-weighted network graph.
   Represents a driver following an un-optimized itinerary.

2. Nearest-Neighbor Heuristic Baseline ("Dijkstra (nearest-neighbor heuristic)"):
   Greedily builds a tour by starting at an initial stop (typically stop 0, the depot),
   and at each step selecting the nearest unvisited stop based on Dijkstra shortest-path
   distances. Repeated until all stops are visited (Rosenkrantz, Stearns & Lewis, 1977;
   Johnson & McGeoch, 1997).

This module implements (2) as the primary baseline, explicitly labeled as:
    "Dijkstra (nearest-neighbor heuristic)"
in all benchmarks, tables, and replan interfaces to maintain scientific accuracy in
academic papers and comparative reports.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.planner.fitness import CongestionLookup, score_route
from src.planner.qpso_encoding import (
    _dijkstra_single_source,
    adjacency_from_network_graph,
    compute_distance_matrix,
)
from src.state_extraction.network_graph import NetworkGraph

ALGORITHM_LABEL = "Dijkstra (nearest-neighbor heuristic)"
NAIVE_LABEL = "Dijkstra (naive input order)"
ALL_STARTS_LABEL = "Dijkstra (multi-start nearest-neighbor heuristic)"

__all__ = [
    "ALGORITHM_LABEL",
    "NAIVE_LABEL",
    "ALL_STARTS_LABEL",
    "dijkstra_nearest_neighbor",
    "dijkstra_all_starts_nearest_neighbor",
    "dijkstra_naive",
    "dijkstra_from_graph",
    "replan",
]


def dijkstra_nearest_neighbor(
    stops: List[str],
    distance_matrix: np.ndarray,
    congestion_lookup: Optional[CongestionLookup] = None,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    start_idx: int = 0,
    return_history: bool = False,
    max_iterations: Optional[int] = None,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """
    Construct a delivery route using the greedy Nearest-Neighbor heuristic
    over Dijkstra-derived pairwise shortest path distances.

    Starting at `start_idx` (default 0), greedily visits the unvisited stop
    with the smallest Dijkstra travel time/distance from the current stop.

    Args:
        stops: List of stop identifiers (length n).
        distance_matrix: (n, n) matrix of shortest-path distances between stops,
            precomputed via Dijkstra's algorithm on the road graph.
        congestion_lookup: Optional edge congestion dictionary for fitness evaluation.
        weights: (w1, w2, w3) weighting for travel time, distance, and congestion.
        start_idx: Index in `stops` to begin the tour from (default: 0).

    Returns:
        (order, score):
            - order: 1D numpy array of stop indices in visitation order.
            - score: Route fitness score evaluated via score_route().
    """
    n = len(stops)
    if distance_matrix.shape != (n, n):
        raise ValueError(
            f"distance_matrix shape {distance_matrix.shape} does not match len(stops)={n}."
        )
    if n == 0:
        return np.array([], dtype=int), 0.0
    if n == 1:
        order = np.array([0], dtype=int)
        score = score_route(order, distance_matrix, congestion_lookup or {}, weights)
        return order, score

    if not (0 <= start_idx < n):
        raise IndexError(f"start_idx={start_idx} is out of bounds for {n} stops.")

    visited = [start_idx]
    unvisited = set(range(n)) - {start_idx}

    current = start_idx
    while unvisited:
        # Greedily pick the unvisited stop with minimum Dijkstra shortest-path cost
        # Stable deterministic tie-breaking by candidate stop index
        next_stop = min(
            unvisited,
            key=lambda candidate: (distance_matrix[current, candidate], candidate),
        )
        visited.append(next_stop)
        unvisited.remove(next_stop)
        current = next_stop

    order = np.asarray(visited, dtype=int)
    score = score_route(order, distance_matrix, congestion_lookup or {}, weights)
    if return_history:
        iter_count = max_iterations if max_iterations is not None else max(100, 75 * n)
        history = np.full(iter_count, score, dtype=float)
        return order, score, history
    return order, score


def dijkstra_all_starts_nearest_neighbor(
    stops: List[str],
    distance_matrix: np.ndarray,
    congestion_lookup: Optional[CongestionLookup] = None,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, float]:
    """
    Multi-start Nearest-Neighbor heuristic across all possible origin stops.

    Evaluates the greedy Nearest-Neighbor heuristic starting from each of the
    n candidate stops and selects the route yielding the lowest total score.

    Args:
        stops: List of stop identifiers.
        distance_matrix: (n, n) Dijkstra shortest-path distance matrix.
        congestion_lookup: Optional edge congestion lookup.
        weights: (w1, w2, w3) fitness weights.

    Returns:
        (best_order, best_score): Best visitation permutation and its fitness.
    """
    n = len(stops)
    if n == 0:
        return np.array([], dtype=int), 0.0

    best_order: Optional[np.ndarray] = None
    best_score = float("inf")

    for start in range(n):
        order, score = dijkstra_nearest_neighbor(
            stops=stops,
            distance_matrix=distance_matrix,
            congestion_lookup=congestion_lookup,
            weights=weights,
            start_idx=start,
        )
        if score < best_score:
            best_score = score
            best_order = order

    assert best_order is not None
    return best_order, best_score


def dijkstra_naive(
    stops: List[str],
    distance_matrix: np.ndarray,
    congestion_lookup: Optional[CongestionLookup] = None,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, float]:
    """
    Naive baseline: visits stops in the exact order given [0, 1, ..., n - 1],
    scored using Dijkstra shortest paths between consecutive stops.

    Args:
        stops: List of stop identifiers.
        distance_matrix: (n, n) Dijkstra shortest-path matrix.
        congestion_lookup: Optional edge congestion lookup.
        weights: (w1, w2, w3) fitness weights.

    Returns:
        (order, score): Natural order array [0, 1, ..., n - 1] and fitness score.
    """
    n = len(stops)
    order = np.arange(n, dtype=int)
    score = score_route(order, distance_matrix, congestion_lookup or {}, weights)
    return order, score


def dijkstra_from_graph(
    network_graph: NetworkGraph,
    stops: List[str],
    edge_weights: Optional[Dict[str, float]] = None,
    congestion_lookup: Optional[CongestionLookup] = None,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    start_idx: int = 0,
) -> Tuple[np.ndarray, float]:
    """
    Construct a nearest-neighbor route via repeated on-demand Dijkstra calls
    directly on the underlying road network graph.

    At each step k, executes single-source Dijkstra from the current stop on the
    network graph, finds the nearest unvisited target stop in `stops`, advances
    to that stop, and repeats until all stops are sequenced.

    Args:
        network_graph: Parsed NetworkGraph instance.
        stops: List of node IDs representing stops to visit.
        edge_weights: Optional live edge weights (falls back to free-flow).
        congestion_lookup: Optional congestion dictionary.
        weights: (w1, w2, w3) fitness weights.
        start_idx: Index of starting stop in `stops`.

    Returns:
        (order, score): Visitation permutation and fitness score.
    """
    n = len(stops)
    if n <= 1:
        order = np.arange(n, dtype=int)
        dist_mat = compute_distance_matrix(
            adjacency_from_network_graph(network_graph, edge_weights or {}), stops
        )
        score = score_route(order, dist_mat, congestion_lookup or {}, weights)
        return order, score

    adjacency = adjacency_from_network_graph(network_graph, edge_weights or {})
    visited = [start_idx]
    unvisited = set(range(n)) - {start_idx}

    current_idx = start_idx
    while unvisited:
        source_node = stops[current_idx]
        distances = _dijkstra_single_source(adjacency, source_node)

        # Find closest unvisited target stop
        best_candidate: Optional[int] = None
        best_dist = float("inf")

        for candidate in sorted(unvisited):
            cand_node = stops[candidate]
            cand_dist = distances.get(cand_node, float("inf"))
            if cand_dist < best_dist:
                best_dist = cand_dist
                best_candidate = candidate

        if best_candidate is None or not np.isfinite(best_dist):
            # Fallback if graph is disconnected: pick arbitrary remaining stop
            best_candidate = min(unvisited)

        visited.append(best_candidate)
        unvisited.remove(best_candidate)
        current_idx = best_candidate

    order = np.asarray(visited, dtype=int)
    dist_mat = compute_distance_matrix(adjacency, stops)
    score = score_route(order, dist_mat, congestion_lookup or {}, weights)
    return order, score


def replan(
    stops: List[str],
    distance_matrix: np.ndarray,
    congestion_lookup: CongestionLookup,
    volatility_index: float = 0.5,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    start_idx: int = 0,
    all_starts: bool = False,
    return_history: bool = False,
    max_iterations: Optional[int] = None,
    **kwargs,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """
    Execute Dijkstra Nearest-Neighbor routing on a frozen state snapshot.

    Provides exact signature parity with `qpso.replan()`, `pso_baseline.replan()`,
    and `ga_baseline.replan()`. Allows simulation replanners and benchmarking
    frameworks to invoke the Dijkstra Nearest-Neighbor baseline as a drop-in
    replacement under identical evaluation contracts.

    Args:
        stops: Stop identifiers mapping to distance_matrix rows/cols.
        distance_matrix: (n, n) matrix of live travel times precomputed via Dijkstra.
        congestion_lookup: Congestion metrics per edge pair.
        volatility_index: Interface compatibility parameter (Dijkstra NN uses live matrix).
        weights: (w1, w2, w3) for travel time, distance, and congestion penalties.
        start_idx: Origin stop index when all_starts is False (default: 0).
        all_starts: If True, evaluates all n starting points and returns the best.
        **kwargs: Swallows unused hyperparameter arguments (seed, population_size,
                  max_iterations, etc.) for seamless polymorphic interface compatibility.

    Returns:
        (best_order, best_score): Stop visitation order permutation and total fitness.
    """
    if all_starts:
        order, score = dijkstra_all_starts_nearest_neighbor(
            stops=stops,
            distance_matrix=distance_matrix,
            congestion_lookup=congestion_lookup,
            weights=weights,
        )
        if return_history:
            iter_count = max_iterations if max_iterations is not None else max(100, 75 * len(stops))
            history = np.full(iter_count, score, dtype=float)
            return order, score, history
        return order, score

    return dijkstra_nearest_neighbor(
        stops=stops,
        distance_matrix=distance_matrix,
        congestion_lookup=congestion_lookup,
        weights=weights,
        start_idx=start_idx,
        return_history=return_history,
        max_iterations=max_iterations,
    )
