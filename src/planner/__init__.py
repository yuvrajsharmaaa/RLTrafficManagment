"""
Route Planning & Optimization algorithms (QPSO, PSO, GA, and Dijkstra Baselines).
"""

from .objective import RoutingObjective
from .baselines import DijkstraBaseline, StandardPSOBaseline
from .qpso_encoding import (
    decode_order,
    compute_distance_matrix,
    adjacency_from_network_graph,
    pick_mutually_reachable_stops,
    tour_length,
)
from .fitness import score_route, route_components
from .qpso import fixed_beta_qpso, va_qpso, replan, default_budget
from .pso_baseline import standard_pso, replan as pso_replan
from .ga_baseline import genetic_algorithm, replan as ga_replan
from .dijkstra_baseline import (
    ALGORITHM_LABEL as DIJKSTRA_NN_LABEL,
    NAIVE_LABEL as DIJKSTRA_NAIVE_LABEL,
    ALL_STARTS_LABEL as DIJKSTRA_ALL_STARTS_LABEL,
    dijkstra_nearest_neighbor,
    dijkstra_all_starts_nearest_neighbor,
    dijkstra_naive,
    dijkstra_from_graph,
    replan as dijkstra_replan,
)

__all__ = [
    "RoutingObjective",
    "DijkstraBaseline",
    "StandardPSOBaseline",
    "decode_order",
    "compute_distance_matrix",
    "adjacency_from_network_graph",
    "pick_mutually_reachable_stops",
    "tour_length",
    "score_route",
    "route_components",
    "fixed_beta_qpso",
    "va_qpso",
    "replan",
    "default_budget",
    "standard_pso",
    "pso_replan",
    "genetic_algorithm",
    "ga_replan",
    "DIJKSTRA_NN_LABEL",
    "DIJKSTRA_NAIVE_LABEL",
    "DIJKSTRA_ALL_STARTS_LABEL",
    "dijkstra_nearest_neighbor",
    "dijkstra_all_starts_nearest_neighbor",
    "dijkstra_naive",
    "dijkstra_from_graph",
    "dijkstra_replan",
]
