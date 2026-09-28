"""
Random-Key Encoding and Distance-Matrix Precomputation for QPSO Route Planning.

QPSO (see qpso.py) searches a continuous real-valued space, but stop
sequencing is a discrete permutation problem. Bean's random-key encoding
bridges the two: each particle position is a real-valued vector x in R^n, and
the visit order is decoded as decode_order(x) = argsort(x). Because argsort is
stable under small perturbations of x (nudging one coordinate a little only
swaps it past neighbors whose keys are close in value), small moves in
continuous QPSO-space correspond to small changes in visit order. That
locality is what makes a continuous swarm search work on a combinatorial
sequencing problem at all — without it, arbitrary encodings would make the
fitness landscape discontinuous and the swarm's velocity/position updates
meaningless.

Reference:
    Bean, J.C. (1994). "Genetic algorithms and random keys for sequencing
    and optimization." ORSA Journal on Computing, 6(2), 154-160.
"""

import heapq
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from src.state_extraction.network_graph import NetworkGraph


def decode_order(x: np.ndarray) -> np.ndarray:
    """
    Decode a random-key vector into a visit-order permutation (Bean, 1994).

    Args:
        x: Real-valued particle position, shape (n,). Values need not be
           sorted, bounded to a specific range, or unique — only their
           relative order matters.

    Returns:
        Integer array of shape (n,): the indices of x in ascending order,
        i.e. the order in which stops should be visited.
    """
    return np.argsort(x, kind="stable")


def _dijkstra_single_source(
    adjacency: Dict[str, List[Any]],
    source: str,
    *,
    return_physical_lengths: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, float]]]:
    """
    Single-source shortest path distances via Dijkstra's algorithm.

    Priority is strictly travel-time (time-optimal path). When
    `return_physical_lengths=True`, also returns the sum of physical edge
    lengths along the time-optimal path actually driven.

    O(E + V log V) with a binary heap.
    """
    times: Dict[str, float] = {source: 0.0}
    lengths: Dict[str, float] = {source: 0.0}
    visited = set()
    pq: List[Tuple[float, float, str]] = [(0.0, 0.0, source)]  # (time, physical_length, node)

    while pq:
        time_cost, phys_cost, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        for item in adjacency.get(node, []):
            neighbor = item[0]
            edge_time = item[1]
            edge_len = item[2] if len(item) > 2 else edge_time

            if neighbor in visited:
                continue

            new_time = time_cost + edge_time
            new_len = phys_cost + edge_len

            cur_best_time = times.get(neighbor, float("inf"))
            if new_time < cur_best_time - 1e-12 or (
                abs(new_time - cur_best_time) <= 1e-12
                and new_len < lengths.get(neighbor, float("inf"))
            ):
                times[neighbor] = new_time
                lengths[neighbor] = new_len
                heapq.heappush(pq, (new_time, new_len, neighbor))

    if return_physical_lengths:
        return times, lengths
    return times


def compute_distance_matrix(
    adjacency: Dict[str, List[Any]],
    stops: List[str],
    *,
    return_physical_distance: bool = False,
    return_both: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Precompute the full stop-to-stop shortest-path travel-time matrix and
    optionally the physical-distance matrix along those time-optimal paths.

    Runs one Dijkstra per stop (O(n) runs, each O(E + V log V) on the live-
    weighted graph) and caches every pairwise result in an n x n matrix.

    Args:
        adjacency: Directed graph as node -> [(neighbor, weight, length), ...]
            or [(neighbor, weight), ...].
        stops: Node ids to compute pairwise distances between.
        return_physical_distance: If True, returns (time_matrix, distance_matrix).
        return_both: Alias for return_physical_distance.

    Returns:
        (n, n) time_matrix, or (time_matrix, distance_matrix) if return_both is True.
    """
    n = len(stops)
    time_matrix = np.full((n, n), np.inf, dtype=float)
    dist_matrix = np.full((n, n), np.inf, dtype=float)

    for i, source in enumerate(stops):
        times, phys_lengths = _dijkstra_single_source(
            adjacency, source, return_physical_lengths=True
        )
        for j, target in enumerate(stops):
            if target in times:
                time_matrix[i, j] = times[target]
                dist_matrix[i, j] = phys_lengths[target]

    if return_physical_distance or return_both:
        return time_matrix, dist_matrix
    return time_matrix


def compute_travel_and_distance_matrices(
    adjacency: Dict[str, List[Any]],
    stops: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute both the live-weighted travel-time matrix T (in seconds) and the
    physical-distance matrix D (in meters) along the TIME-optimal path actually driven.

    Returns:
        (time_matrix, distance_matrix)
    """
    return compute_distance_matrix(adjacency, stops, return_both=True)


def adjacency_from_network_graph(
    network_graph: NetworkGraph,
    edge_weights: Dict[str, float],
) -> Dict[str, List[Tuple[str, float, float]]]:
    """
    Build a node -> [(neighbor, weight, length), ...] adjacency dict from a parsed
    NetworkGraph, using a caller-supplied edge-weight snapshot for live travel times
    and the static edge length from .net.xml for physical distance.
    """
    adjacency: Dict[str, List[Tuple[str, float, float]]] = {}

    for edge_id, edge in network_graph.edges.items():
        from_node, to_node = edge["from"], edge["to"]
        length = float(edge.get("length", 0.0))
        weight = edge_weights.get(edge_id)
        if weight is None:
            speed = float(edge.get("speed", 13.89))
            weight = length / speed if speed > 0 else float("inf")
        adjacency.setdefault(from_node, []).append((to_node, weight, length))

    return adjacency


def _reachable_from(
    adjacency: Dict[str, List[Any]],
    source: str,
) -> set:
    """Nodes reachable from `source` by following adjacency edges forward."""
    seen = {source}
    stack = [source]
    while stack:
        node = stack.pop()
        for item in adjacency.get(node, []):
            neighbor = item[0]
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _reverse_adjacency(
    adjacency: Dict[str, List[Any]],
) -> Dict[str, List[Tuple]]:
    reverse: Dict[str, List[Tuple]] = {}
    for node, edges in adjacency.items():
        for item in edges:
            neighbor = item[0]
            weight = item[1]
            if len(item) > 2:
                reverse.setdefault(neighbor, []).append((node, weight, item[2]))
            else:
                reverse.setdefault(neighbor, []).append((node, weight))
    return reverse


def pick_mutually_reachable_stops(
    adjacency: Dict[str, List[Tuple[str, float]]],
    num_stops: int,
) -> List[str]:
    """
    Pick `num_stops` nodes that are all mutually reachable, so the distance
    matrix built from them has no np.inf entries.

    Real road networks are directed (one-ways), so an arbitrary set of nodes
    is usually NOT mutually reachable and would yield an unusable matrix. A
    node x is mutually reachable with `source` iff x is forward-reachable
    from source AND source is forward-reachable from x (i.e. x is reachable
    from source on the reversed graph); the intersection of those two sets is
    exactly the strongly connected component containing `source`.

    Raises:
        RuntimeError: if no strongly connected component is large enough.
    """
    reverse = _reverse_adjacency(adjacency)
    for source in adjacency:
        if not adjacency.get(source):
            continue
        scc = _reachable_from(adjacency, source) & _reachable_from(reverse, source)
        if len(scc) >= num_stops:
            return sorted(scc)[:num_stops]
    raise RuntimeError(
        f"Could not find {num_stops} mutually-reachable nodes in this network."
    )


def tour_length(order: np.ndarray, distance_matrix: np.ndarray) -> float:
    """
    Total distance of visiting stops in `order` (as produced by
    decode_order), using O(n) lookups into a precomputed distance_matrix
    rather than any graph search.
    """
    total = 0.0
    for a, b in zip(order[:-1], order[1:]):
        total += distance_matrix[a, b]
    return total
