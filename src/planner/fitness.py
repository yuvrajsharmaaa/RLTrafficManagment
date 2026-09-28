"""
Route fitness scoring: travel time, physical distance, and congestion cost.

score_route() combines three weighted terms into a single scalar (lower is
better) — the intended QPSO fitness function once a candidate particle
position has been decoded to a stop order (see qpso_encoding.decode_order /
qpso.replan):

    score = w1*T + w2*D + w3*C

- T: total travel time (seconds) between consecutive stops in `order`, read from
  `time_matrix` (or `distance_matrix`). That matrix is built from live/current
  edge speeds (e.g. a state.py subscription snapshot fed through
  qpso_encoding.compute_distance_matrix), not static free-flow times —
  otherwise T does not reflect current congestion at all.
- D: total physical distance (meters) between consecutive stops in `order`, read
  from `distance_matrix` (sum of edge lengths from the .net.xml along the
  TIME-optimal path actually driven). This decouples spatial distance from travel
  time without evaluating an unrealistic separate shortest-distance path that
  no vehicle takes.
  If a distinct physical-distance matrix is not supplied, D defaults to T for
  backwards compatibility.
- C: congestion cost. For every road-network edge touched by the route —
  every edge on every leg (order[k] -> order[k+1]) — add
  (occupancy / capacity) ** 2. Squaring means a near-saturated edge
  (ratio close to 1) contributes disproportionately more than a lightly
  used one, rather than penalizing congestion linearly. A route that
  traverses the same physical edge on two different legs is charged twice,
  since it really does add load to that edge twice.

`weights` is normalized so w1 + w2 + w3 == 1 before use, regardless of what
scale the caller passes in.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Maps a route leg -- the pair of stop indices (order[k], order[k+1]) exactly
# as they appear consecutively in `order` -- to every road-network edge that
# leg passes through, each as {"occupancy": float, "capacity": float}.
CongestionLookup = Dict[Tuple[int, int], List[Dict[str, float]]]


def _resolve_route_args(
    time_matrix: np.ndarray,
    args: tuple,
    distance_matrix: Optional[np.ndarray],
    congestion_lookup: Optional[CongestionLookup],
    weights: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, CongestionLookup, Tuple[float, float, float]]:
    """
    Resolve positional/keyword arguments across both legacy (single matrix)
    and modern (both travel-time and physical-distance matrices) call patterns.
    """
    actual_time = time_matrix
    actual_dist = distance_matrix
    actual_cong = congestion_lookup
    actual_weights = weights

    if actual_dist is not None:
        # distance_matrix explicitly passed as keyword
        if len(args) >= 1 and actual_cong is None:
            actual_cong = args[0]
        if len(args) >= 2 and actual_weights is None:
            actual_weights = args[1]
    else:
        # distance_matrix was not passed as keyword
        if len(args) >= 1 and isinstance(args[0], np.ndarray):
            # Positional pattern: (order, time_matrix, distance_matrix, [congestion_lookup], [weights])
            actual_dist = args[0]
            if len(args) >= 2 and actual_cong is None:
                actual_cong = args[1]
            if len(args) >= 3 and actual_weights is None:
                actual_weights = args[2]
        else:
            # Positional pattern: (order, distance_matrix, [congestion_lookup], [weights])
            actual_dist = actual_time
            if len(args) >= 1 and actual_cong is None:
                actual_cong = args[0]
            if len(args) >= 2 and actual_weights is None:
                actual_weights = args[1]

    if actual_dist is None:
        actual_dist = actual_time
    if actual_cong is None:
        actual_cong = {}
    if actual_weights is None:
        actual_weights = (1.0, 1.0, 1.0)

    return actual_time, actual_dist, actual_cong, actual_weights


def route_components(
    order: np.ndarray,
    time_matrix: np.ndarray,
    *args,
    distance_matrix: Optional[np.ndarray] = None,
    congestion_lookup: Optional[CongestionLookup] = None,
) -> Tuple[float, float, float]:
    """
    Compute the raw (T, D, C) components for a route, before weighting.

    Supports both signatures:
    - Decoupled: route_components(order, time_matrix, distance_matrix, congestion_lookup)
    - Legacy:    route_components(order, distance_matrix, congestion_lookup)
    - Keyword:   route_components(order, time_matrix, congestion_lookup, distance_matrix=...)

    Args:
        order: Visit-order permutation (indices into distance/time matrix).
        time_matrix: (n, n) live travel-time matrix in seconds.
        *args: Either (distance_matrix, congestion_lookup) or (congestion_lookup,).
        distance_matrix: (n, n) physical-distance matrix in meters (sum of edge
            lengths along the time-optimal path). If None, defaults to time_matrix.
        congestion_lookup: See CongestionLookup above.

    Returns:
        (T, D, C) -- travel time (s), physical distance (m), congestion cost.
    """
    time_mat, dist_mat, cong_look, _ = _resolve_route_args(
        time_matrix, args, distance_matrix, congestion_lookup
    )

    legs = list(zip(order[:-1], order[1:]))

    T = 0.0
    D = 0.0
    for i, j in legs:
        T += float(time_mat[i, j])
        D += float(dist_mat[i, j])

    C = 0.0
    for i, j in legs:
        for edge in cong_look.get((int(i), int(j)), ()):
            ratio = edge["occupancy"] / edge["capacity"]
            C += ratio ** 2

    return T, D, C


def score_route(
    order: np.ndarray,
    time_matrix: np.ndarray,
    *args,
    weights: Optional[Tuple[float, float, float]] = None,
    distance_matrix: Optional[np.ndarray] = None,
    congestion_lookup: Optional[CongestionLookup] = None,
) -> float:
    """
    Weighted route fitness: w1*T + w2*D + w3*C, with weights normalized to
    sum to 1 before use. Lower is better.

    Supports both signatures:
    - Decoupled: score_route(order, time_matrix, distance_matrix, congestion_lookup, weights)
    - Legacy:    score_route(order, distance_matrix, congestion_lookup, weights)
    - Keyword:   score_route(order, time_matrix, congestion_lookup, weights, distance_matrix=...)

    Args:
        order: Visit-order permutation (indices into distance/time matrix).
        time_matrix: (n, n) live-weighted travel-time matrix (s).
        *args: Either (distance_matrix, congestion_lookup, weights) or
            (congestion_lookup, weights).
        weights: (w1, w2, w3) for (T, D, C). Need not already sum to 1 --
            they are rescaled by their sum before use.
        distance_matrix: (n, n) physical-distance matrix (m) along the time-optimal path.
        congestion_lookup: See CongestionLookup above.

    Returns:
        Scalar fitness score.
    """
    time_mat, dist_mat, cong_look, actual_weights = _resolve_route_args(
        time_matrix, args, distance_matrix, congestion_lookup, weights
    )

    w1, w2, w3 = actual_weights
    total_weight = w1 + w2 + w3
    if total_weight <= 0:
        raise ValueError(f"weights must sum to a positive value, got {actual_weights}")
    w1, w2, w3 = w1 / total_weight, w2 / total_weight, w3 / total_weight

    T, D, C = route_components(order, time_mat, dist_mat, cong_look)
    return w1 * T + w2 * D + w3 * C
