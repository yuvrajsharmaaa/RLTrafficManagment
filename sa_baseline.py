"""
Standard Simulated Annealing (SA) Baseline for Route Planning.

Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). "Optimization by Simulated Annealing."
Science, 220(4598), 671-680.
Lin, S., & Kernighan, B. W. (1973). "An effective heuristic algorithm for the traveling-salesman problem."
Operations Research, 21(2), 498-516.

Native Permutation Representation:
----------------------------------
Operates directly on discrete visit-order permutations of stops:
an array of stop indices `[0, 1, ..., n - 1]` with no continuous random-key mapping needed.

Standard Simulated Annealing Mechanics:
---------------------------------------
1. Neighbor Generation: 2-Opt Segment-Reversal Move
   - Randomly chooses two distinct cut indices i < j.
   - Reverses the subsegment between i and j inclusive:
     order[i:j+1] = order[i:j+1][::-1]
   - Guaranteed valid permutation by construction (0 duplicates, 0 omitted stops).
2. Cooling Schedule: Geometric Cooling
   - T(t) = max(T_0 * (cooling_rate ** t), T_min)
   - T_0: Initial temperature (default: 100.0)
   - cooling_rate: Geometric decay parameter in (0, 1) (default: 0.99)
3. Acceptance Criterion: Metropolis-Hastings Rule (Minimization)
   - delta = candidate_cost - current_cost
   - If delta <= 0: accept candidate unconditionally (downhill improvement)
   - If delta > 0: accept candidate with probability P = exp(-delta / T)
4. Elite Preservation:
   - Tracks global best solution (best_order, best_score) across all accepted/rejected moves.
5. Stagnation Recovery:
   - If global best fails to improve for `patience = 15` consecutive temperature iterations,
     re-seeds search to escape frozen local basins while preserving the global elite.

Budget & Function Evaluation Parity:
------------------------------------
Uses the exact same `default_budget(dim)` as va_qpso, fixed_beta_qpso, pso_baseline, and ga_baseline:
    steps_per_temp = max(20, 4 * dim)
    max_iterations = max(100, 75 * dim)
    max_restarts = max(5, 5 * dim)
At each temperature step t, `steps_per_temp` candidate neighbor evaluations are performed,
yielding exactly `steps_per_temp * max_iterations` total function evaluations -- strictly matching
the evaluation capacity of QPSO, PSO, and GA.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.planner.fitness import CongestionLookup, score_route
from src.planner.sa_baseline import (
    DEFAULT_COOLING_RATE,
    DEFAULT_MIN_TEMP,
    DEFAULT_PATIENCE,
    DEFAULT_T0,
    DEFAULT_TOL,
    default_budget,
    replan,
    simulated_annealing,
    two_opt_swap,
)

FitnessFn = Callable[[np.ndarray], float]

__all__ = [
    "DEFAULT_T0",
    "DEFAULT_COOLING_RATE",
    "DEFAULT_MIN_TEMP",
    "DEFAULT_PATIENCE",
    "DEFAULT_TOL",
    "default_budget",
    "two_opt_swap",
    "simulated_annealing",
    "replan",
    "score_route",
]


def run_benchmark():
    """Run benchmark against true brute-force optimum on Delhi network."""
    import itertools
    import time
    from src.planner.qpso import va_qpso
    from src.planner.qpso_encoding import (
        adjacency_from_network_graph,
        compute_distance_matrix,
        decode_order,
        pick_mutually_reachable_stops,
    )
    from src.state_extraction.network_graph import NetworkGraph

    net_file = PROJECT_ROOT / "networks" / "delhi" / "delhi_intersection.net.xml"
    if not net_file.exists():
        print(f"Network file not found at {net_file}, skipping live network benchmark.")
        return

    num_stops = 8
    graph = NetworkGraph(str(net_file))
    adj = adjacency_from_network_graph(graph, edge_weights={})
    stops = pick_mutually_reachable_stops(adj, num_stops)
    dist_mat = compute_distance_matrix(adj, stops)

    def perm_fitness(order: np.ndarray) -> float:
        return score_route(order, dist_mat, {})

    def cont_fitness(x: np.ndarray) -> float:
        return score_route(decode_order(x), dist_mat, {})

    print("\n" + "=" * 70)
    print(f"Brute-forcing true global optimum across {num_stops}! = 40,320 permutations...")
    t0 = time.perf_counter()
    all_perms = itertools.permutations(range(num_stops))
    true_opt = min(perm_fitness(np.asarray(p)) for p in all_perms)
    print(f"True global optimum fitness: {true_opt:.6f} s (computed in {time.perf_counter() - t0:.2f}s)")
    print("=" * 70)

    # Evaluate Simulated Annealing
    t0 = time.perf_counter()
    sa_order, sa_score = simulated_annealing(
        dim=num_stops,
        fitness_fn=perm_fitness,
        seed=42,
    )
    t_sa = (time.perf_counter() - t0) * 1000.0
    print(f"Simulated Annealing: score = {sa_score:.6f} s, runtime = {t_sa:.2f} ms, gap = {sa_score - true_opt:.6f} s")

    # Evaluate VA-QPSO
    t0 = time.perf_counter()
    qpso_pos, qpso_score = va_qpso(
        dim=num_stops,
        fitness_fn=cont_fitness,
        volatility_index=0.5,
        seed=42,
    )
    t_qpso = (time.perf_counter() - t0) * 1000.0
    print(f"VA-QPSO            : score = {qpso_score:.6f} s, runtime = {t_qpso:.2f} ms, gap = {qpso_score - true_opt:.6f} s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_benchmark()
