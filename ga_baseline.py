"""
Standard Permutation Genetic Algorithm (GA) Baseline for Route Planning.

Goldberg, D. E. (1989). "Genetic Algorithms in Search, Optimization, and Machine Learning."
Davis, L. (1985). "Applying adaptive planned algorithms to the traveling salesman problem."
Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 973-976.

Native Permutation Representation:
----------------------------------
Operates directly on discrete visit-order permutations of stops:
an array of stop indices `[0, 1, ..., n - 1]` with no continuous random-key mapping needed.

Standard Genetic Operators:
---------------------------
1. Selection: Tournament Selection (size 3-5, default k=3).
   Picks k candidates uniformly at random; the candidate with the lowest cost (best fitness)
   wins and becomes a parent.
2. Crossover: Order Crossover (OX / OX1, Davis 1985).
   Preserves relative ordering from both parents:
   - Selects two random cut points c1 < c2.
   - Copies slice parent1[c1:c2] into child.
   - Fills remaining positions in circular order starting at c2 using elements from parent2
     that are not already present in the copied slice.
   - Crossover rate: default 0.9.
3. Mutation: Swap Mutation.
   Swaps two randomly chosen stop positions with probability p_m per individual (default: 0.08).
4. Elitism:
   Carries over the top E individuals unchanged each generation (default: E=2).

Documented Hyperparameters:
---------------------------
- Tournament size k: 3 (configurable in [3, 5])
- Crossover probability p_c: 0.9 (Order Crossover OX)
- Mutation probability p_m: 0.08 (Swap mutation, range [0.05, 0.10])
- Elitism count E: 2 (preserves top 2 performers)
- Stagnation patience: 15 generations
- Relative improvement tolerance: 1e-6

Budget & Function Evaluation Parity:
------------------------------------
Uses the exact same `default_budget(dim)` as va_qpso, fixed_beta_qpso, and pso_baseline:
    population_size = max(20, 4 * dim)
    max_generations = max(100, 75 * dim)
    max_restarts = max(5, 5 * dim)
Equal population size and generation budget ensure identical function evaluation capacity
across all four algorithms.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.planner.fitness import CongestionLookup, score_route
from src.planner.ga_baseline import (
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_ELITISM,
    DEFAULT_MUTATION_RATE,
    DEFAULT_PATIENCE,
    DEFAULT_TOL,
    DEFAULT_TOURNAMENT_SIZE,
    default_budget,
    genetic_algorithm,
    order_crossover,
    replan,
    swap_mutation,
    tournament_selection,
)

FitnessFn = Callable[[np.ndarray], float]

__all__ = [
    "DEFAULT_TOURNAMENT_SIZE",
    "DEFAULT_CROSSOVER_RATE",
    "DEFAULT_MUTATION_RATE",
    "DEFAULT_ELITISM",
    "DEFAULT_PATIENCE",
    "DEFAULT_TOL",
    "default_budget",
    "order_crossover",
    "swap_mutation",
    "tournament_selection",
    "genetic_algorithm",
    "replan",
    "score_route",
]


def run_benchmark():
    """Run comprehensive 4-way benchmark against true brute-force optimum on Delhi network."""
    import itertools
    import math
    import time
    from pso_baseline import standard_pso
    from src.planner.qpso import fixed_beta_qpso, va_qpso
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

    # Permutation fitness function (for GA)
    def ga_fitness_fn(order: np.ndarray) -> float:
        return score_route(order, dist_mat, {}, (1.0, 1.0, 1.0))

    # Continuous fitness function (for QPSO and Standard PSO random-key encoding)
    def continuous_fitness_fn(x: np.ndarray) -> float:
        order = decode_order(x)
        return score_route(order, dist_mat, {}, (1.0, 1.0, 1.0))

    # True brute-force optimum
    t0 = time.perf_counter()
    true_opt = min(
        score_route(np.asarray(p), dist_mat, {}, (1.0, 1.0, 1.0))
        for p in itertools.permutations(range(num_stops))
    )
    t_bf = time.perf_counter() - t0

    budget = default_budget(num_stops)
    print(f"==========================================================================================")
    print(f" Delhi Road Network Routing Benchmark (8 Stops, {math.factorial(num_stops):,} Permutations)")
    print(f" Equal Budget Parity across All 4 Optimizers:")
    print(f"   - Swarm / Population Size: {budget[0]}")
    print(f"   - Iteration / Generation Limit: {budget[1]}")
    print(f"   - Max Stagnation Restarts: {budget[2]}")
    print(f" True Brute-Force Global Optimum: {true_opt:.4f} s (computed in {t_bf*1000:.1f} ms)")
    print(f"==========================================================================================\n")

    # 1. Permutation GA (Tournament k=3, Order Crossover OX, Swap Mutation p_m=0.08, Elitism E=2)
    t0 = time.perf_counter()
    ga_order, ga_score = genetic_algorithm(
        dim=num_stops,
        fitness_fn=ga_fitness_fn,
        tournament_size=3,
        crossover_rate=0.90,
        mutation_rate=0.08,
        elitism=2,
        seed=42,
    )
    t_ga = time.perf_counter() - t0

    # 2. Standard PSO (w=0.7, c1=1.5, c2=1.5)
    t0 = time.perf_counter()
    pso_pos, pso_score = standard_pso(num_stops, continuous_fitness_fn, seed=42)
    t_pso = time.perf_counter() - t0
    pso_order = decode_order(pso_pos)

    # 3. Fixed-Beta QPSO (linear anneal beta 1.0 -> 0.5)
    t0 = time.perf_counter()
    fqpso_pos, fqpso_score = fixed_beta_qpso(num_stops, continuous_fitness_fn, seed=42)
    t_fqpso = time.perf_counter() - t0
    fqpso_order = decode_order(fqpso_pos)

    # 4. Volatility-Adaptive QPSO (VA-QPSO, v=0.5)
    t0 = time.perf_counter()
    vqpso_pos, vqpso_score = va_qpso(num_stops, continuous_fitness_fn, volatility_index=0.5, seed=42)
    t_vqpso = time.perf_counter() - t0
    vqpso_order = decode_order(vqpso_pos)

    results = [
        ("Permutation GA (OX, Swap, Elitist)", ga_order, ga_score, t_ga),
        ("Standard PSO (w=0.7, c1=1.5, c2=1.5)", pso_order, pso_score, t_pso),
        ("Fixed-Beta QPSO (Linear beta 1.0->0.5)", fqpso_order, fqpso_score, t_fqpso),
        ("VA-QPSO (Volatility-Adaptive beta)", vqpso_order, vqpso_score, t_vqpso),
    ]

    header = f"{'Algorithm':<40} | {'Best Score (s)':<15} | {'Gap vs Optimum':<15} | {'Runtime (ms)':<12}"
    print(header)
    print("-" * len(header))
    for name, order, score, runtime in results:
        gap = score - true_opt
        print(f"{name:<40} | {score:<15.4f} | {gap:<15.4f} | {runtime*1000:<12.1f}")
    print()


if __name__ == "__main__":
    run_benchmark()
