"""
Routing Baselines for comparison against QPSO and metaheuristics.

Includes:
- Dijkstra Single-Source Shortest Path
- Dijkstra Nearest-Neighbor Heuristic (Rosenkrantz et al., 1977)
- Standard Particle Swarm Optimization (PSO)
"""

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.planner.dijkstra_baseline import (
    ALGORITHM_LABEL,
    ALL_STARTS_LABEL,
    NAIVE_LABEL,
    dijkstra_all_starts_nearest_neighbor,
    dijkstra_from_graph,
    dijkstra_naive,
    dijkstra_nearest_neighbor,
    replan as dijkstra_replan,
)
from src.planner.fitness import CongestionLookup


class DijkstraBaseline:
    """Standard Dijkstra single-source shortest path algorithm on road graph."""

    def __init__(self, adjacency: Dict[str, List[Tuple[str, float]]]):
        self.adj = adjacency

    def find_shortest_path(self, start: str, end: str) -> Tuple[List[str], float]:
        """Compute shortest path between start and end nodes."""
        pq = [(0.0, start, [start])]
        visited = set()

        while pq:
            cost, current, path = heapq.heappop(pq)
            if current == end:
                return path, cost
            if current in visited:
                continue
            visited.add(current)

            for neighbor, edge_cost in self.adj.get(current, []):
                if neighbor not in visited:
                    heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))

        return [], float("inf")


class DijkstraNearestNeighborBaseline:
    """
    Dijkstra Nearest-Neighbor Heuristic Baseline.

    Label: "Dijkstra (nearest-neighbor heuristic)"
    Dijkstra alone computes point-to-point shortest paths; it does not solve
    combinatorial stop sequencing (TSP/VRP). This baseline greedily advances to
    the nearest unvisited stop at each step using Dijkstra shortest paths.
    """

    LABEL = ALGORITHM_LABEL
    NAIVE_LABEL = NAIVE_LABEL

    @staticmethod
    def solve(
        stops: List[str],
        distance_matrix: np.ndarray,
        congestion_lookup: Optional[CongestionLookup] = None,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        start_idx: int = 0,
    ) -> Tuple[np.ndarray, float]:
        return dijkstra_nearest_neighbor(
            stops=stops,
            distance_matrix=distance_matrix,
            congestion_lookup=congestion_lookup,
            weights=weights,
            start_idx=start_idx,
        )

    @staticmethod
    def replan(
        stops: List[str],
        distance_matrix: np.ndarray,
        congestion_lookup: CongestionLookup,
        **kwargs,
    ) -> Tuple[np.ndarray, float]:
        return dijkstra_replan(
            stops=stops,
            distance_matrix=distance_matrix,
            congestion_lookup=congestion_lookup,
            **kwargs,
        )


class StandardPSOBaseline:
    """Standard velocity-displacement Particle Swarm Optimization."""
    pass
