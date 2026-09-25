"""
Canonical Standard (Non-Quantum) Particle Swarm Optimization (PSO) Baseline.

Implements standard continuous PSO (Kennedy & Eberhart 1995; Shi & Eberhart 1998)
on the exact same random-key encoding (Bean 1994) and fitness function (score_route)
used by qpso.py.

Standard Canonical PSO Update Equations:
----------------------------------------
For particle i in {0, ..., M - 1} and dimension d in {0, ..., D - 1} at iteration t:

    v[i][d] = w * v[i][d] + c1 * r1 * (personal_best[i][d] - x[i][d])
                          + c2 * r2 * (global_best[d] - x[i][d])
    x[i][d] = x[i][d] + v[i][d]

Documented Hyperparameters:
---------------------------
- Inertia weight w: 0.7 (canonical default; optional linearly decreasing 0.9 -> 0.4)
- Cognitive acceleration c1: 1.5 (Kennedy & Eberhart default)
- Social acceleration c2: 1.5 (Kennedy & Eberhart default)
- Random factors r1, r2: uniform(0, 1) drawn independently per particle/dimension/step
- Velocity clamping: v_max = 0.5 * (bounds[1] - bounds[0]) = 0.5
- Position clamping: x in [0.0, 1.0]

Budget Parity:
--------------
Uses the exact same `default_budget(dim)` as va_qpso and fixed_beta_qpso:
    num_particles = max(20, 4 * dim)
    max_iterations = max(100, 75 * dim)
    max_restarts = max(5, 5 * dim)
    patience = 15
    tol = 1e-6
Equal iteration budget and swarm size ensure a strictly fair comparison with equal
function evaluations.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.planner.fitness import CongestionLookup, score_route
from src.planner.qpso_encoding import decode_order
from src.planner.pso_baseline import (
    DEFAULT_C1,
    DEFAULT_C2,
    DEFAULT_VMAX_RATIO,
    DEFAULT_W,
    default_budget,
    replan,
    standard_pso,
)

FitnessFn = Callable[[np.ndarray], float]

__all__ = [
    "DEFAULT_W",
    "DEFAULT_C1",
    "DEFAULT_C2",
    "DEFAULT_VMAX_RATIO",
    "default_budget",
    "standard_pso",
    "replan",
    "decode_order",
    "score_route",
]


def run_benchmark():
    """Run comparative benchmark against QPSO on Delhi network."""
    import itertools
    import time
    from src.planner.qpso import fixed_beta_qpso, va_qpso
    from src.planner.qpso_encoding import (
        adjacency_from_network_graph,
        compute_distance_matrix,
        pick_mutually_reachable_stops,
        tour_length,
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

    def fitness_fn(x: np.ndarray) -> float:
        order = decode_order(x)
        return score_route(order, dist_mat, {}, (1.0, 1.0, 1.0))

    # True brute-force optimum
    t0 = time.perf_counter()
    true_opt = min(
        score_route(np.asarray(p), dist_mat, {}, (1.0, 1.0, 1.0))
        for p in itertools.permutations(range(num_stops))
    )
    t_bf = time.perf_counter() - t0

    import math
    budget = default_budget(num_stops)
    print(f"Delhi Network Benchmark ({num_stops} Stops, {math.factorial(num_stops)} permutations)")
    print(f"Equal Budget: {budget[0]} particles, {budget[1]} max iterations, {budget[2]} max restarts")
    print(f"Brute-Force True Optimum: {true_opt:.4f} s (computed in {t_bf*1000:.1f}ms)\n")

    # 1. Standard PSO (w=0.7, c1=1.5, c2=1.5)
    t0 = time.perf_counter()
    pso_pos, pso_score = standard_pso(num_stops, fitness_fn, seed=42)
    t_pso = time.perf_counter() - t0
    pso_order = decode_order(pso_pos)

    # 2. Standard PSO with linear anneal w (0.9 -> 0.4)
    t0 = time.perf_counter()
    pso_lin_pos, pso_lin_score = standard_pso(num_stops, fitness_fn, w_schedule="linear", seed=42)
    t_pso_lin = time.perf_counter() - t0
    pso_lin_order = decode_order(pso_lin_pos)

    # 3. Fixed-Beta QPSO (linear anneal beta)
    t0 = time.perf_counter()
    fqpso_pos, fqpso_score = fixed_beta_qpso(num_stops, fitness_fn, seed=42)
    t_fqpso = time.perf_counter() - t0
    fqpso_order = decode_order(fqpso_pos)

    # 4. Volatility-Adaptive QPSO (VA-QPSO, v=0.5)
    t0 = time.perf_counter()
    vqpso_pos, vqpso_score = va_qpso(num_stops, fitness_fn, volatility_index=0.5, seed=42)
    t_vqpso = time.perf_counter() - t0
    vqpso_order = decode_order(vqpso_pos)

    results = [
        ("Standard PSO (w=0.7, c1=1.5, c2=1.5)", pso_order, pso_score, t_pso),
        ("Standard PSO (Linear w: 0.9->0.4)", pso_lin_order, pso_lin_score, t_pso_lin),
        ("Fixed-Beta QPSO (Linear beta)", fqpso_order, fqpso_score, t_fqpso),
        ("VA-QPSO (Volatility-Adaptive beta)", vqpso_order, vqpso_score, t_vqpso),
    ]

    print(f"{'Algorithm':<38} | {'Best Fitness (s)':<16} | {'Gap vs Optimum':<14} | {'Runtime (ms)':<12}")
    print("-" * 88)
    for name, order, score, runtime in results:
        gap = score - true_opt
        print(f"{name:<38} | {score:<16.4f} | {gap:<14.4f} | {runtime*1000:<12.1f}")


if __name__ == "__main__":
    run_benchmark()
