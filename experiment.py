"""
Paired Comparative Experiment Runner & Convergence Benchmarking.

Evaluates route optimization algorithms:
  1. va_qpso: Volatility-Adaptive QPSO (Sun et al. 2004 with adaptive beta)
  2. fixed_beta_qpso: Linear-anneal beta QPSO baseline
  3. standard_pso: Canonical continuous PSO (Kennedy & Eberhart 1995)
  4. ga: Standard Permutation Genetic Algorithm (Davis 1985; Goldberg 1989)
  5. dijkstra_nn: Dijkstra (nearest-neighbor heuristic)

Modes:
  - convergence: Evaluates all 5 algorithms across identical seeded trials (default: 30 seeds),
    logging per-iteration fitness history, computing iterations to reach 95% of run's best
    fitness (convergence speed), hit rate against reference best known solution, and generates
    the standard optimization convergence plot (fitness vs iteration with shaded std-dev bands).
  - simulation: Executes TraCI-driven SUMO simulation across volatility tiers with reactive rerouting.
  - both: Executes convergence benchmark followed by simulation runs.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.planner.fitness import CongestionLookup, route_components, score_route
from src.planner.qpso import default_budget, replan as qpso_replan
from src.planner.pso_baseline import replan as pso_replan
from src.planner.ga_baseline import replan as ga_replan
from src.planner.dijkstra_baseline import (
    ALGORITHM_LABEL as DIJKSTRA_NN_LABEL,
    replan as dijkstra_replan,
)
from src.planner.qpso_encoding import (
    adjacency_from_network_graph,
    compute_distance_matrix,
    pick_mutually_reachable_stops,
)
from src.reactive.arbiter import ReplanArbiter
from src.reactive.reactive import evaluate_vehicle_reroute
from src.state_extraction.network_graph import NetworkGraph
from src.state_extraction.state import SubscriptionStateExtractor
from src.volatility import NetworkVolatilityIndex
from traci.exceptions import FatalTraCIError, TraCIException

# Constants
NET_FILE = "networks/delhi/delhi_intersection.net.xml"
NUM_STOPS = 8
N_MAX = 120.0
N_MIN = 20.0
N_FIXED = 70.0

OCCUPANCY_THRESHOLD = 0.8
MIN_OCCUPANCY_IMPROVEMENT = 0.15
MAX_EXTRA_DISTANCE_RATIO = 0.3
ARBITER_WINDOW_SECONDS = 60.0
ARBITER_REROUTE_THRESHOLD = 5
VOLATILITY_WINDOW = 15
REFERENCE_VARIANCE = 0.002

ALGORITHM_DISPLAY_NAMES = {
    "va_qpso": "VA-QPSO (Volatility-Adaptive)",
    "fixed_beta_qpso": "Fixed-Beta QPSO (Linear Anneal)",
    "standard_pso": "Standard PSO (Kennedy & Eberhart)",
    "ga": "Permutation GA (OX / Swap / Elitist)",
    "dijkstra_nn": DIJKSTRA_NN_LABEL,
}

ALGORITHM_COLORS = {
    "va_qpso": "#1f77b4",          # Deep Blue
    "fixed_beta_qpso": "#9467bd",   # Purple
    "standard_pso": "#ff7f0e",     # Orange
    "ga": "#2ca02c",               # Green
    "dijkstra_nn": "#d62728",      # Crimson Red
}


def replan_interval(volatility_index: float) -> float:
    """Volatility-adaptive replan interval in [N_MIN, N_MAX]."""
    return N_MAX - (N_MAX - N_MIN) * volatility_index


def compute_iterations_to_95(history: np.ndarray) -> int:
    """
    Iterations to reach 95% of that run's own best-found improvement.

    f_0 = history[0] (initial fitness at iteration 0)
    f_star = history[-1] (best-found fitness at termination)
    delta = f_0 - f_star
    Target fitness: f_target = f_star + 0.05 * delta
    Returns the first iteration index t where history[t] <= f_target.
    """
    if len(history) == 0:
        return 0
    f_0 = history[0]
    f_star = history[-1]
    delta = f_0 - f_star
    if delta <= 1e-12:
        return 0
    target = f_star + 0.05 * delta
    indices = np.where(history <= target + 1e-12)[0]
    return int(indices[0]) if len(indices) > 0 else len(history) - 1


def compute_hit_rate(scores: np.ndarray, ref_bests: np.ndarray, tol: float = 1e-4) -> float:
    """
    Percentage of trials that reach within `tol` of the reference best solution
    for that scenario instance.
    """
    if len(scores) == 0:
        return 0.0
    hits = np.sum(scores <= ref_bests + tol)
    return float((hits / len(scores)) * 100.0)


def build_scenario_distance_and_congestion(
    network_graph: NetworkGraph,
    stops: List[str],
    tier: str = "medium",
    seed: int = 42,
) -> Tuple[np.ndarray, CongestionLookup, float]:
    """
    Construct distance matrix and realistic congestion lookup for a given scenario tier.
    """
    tier_volatilities = {"low": 0.20, "medium": 0.50, "high": 0.85}
    volatility_index = tier_volatilities.get(tier.lower(), 0.50)

    rng = np.random.default_rng(seed)
    edge_weights: Dict[str, float] = {}

    # Base travel times from network geometry with tier-scaled congestion variance
    for edge_id, edge in network_graph.edges.items():
        base_time = edge["length"] / edge["speed"] if edge["speed"] > 0 else 1.0
        # Congestion delay multiplier scaled by volatility
        congestion_factor = 1.0 + rng.exponential(scale=0.15 + 0.35 * volatility_index)
        edge_weights[edge_id] = base_time * congestion_factor

    adjacency = adjacency_from_network_graph(network_graph, edge_weights)
    distance_matrix = compute_distance_matrix(adjacency, stops)

    congestion_lookup: CongestionLookup = {}
    n = len(stops)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            source_node = stops[i]
            target_node = stops[j]
            edge_records = []
            for edge_id, e_data in network_graph.edges.items():
                if e_data["from"] == source_node or e_data["to"] == target_node:
                    occ = float(np.clip(rng.beta(2.0, 5.0 - 3.0 * volatility_index), 0.05, 0.98))
                    edge_records.append({"occupancy": occ, "capacity": 1.0})
            if edge_records:
                congestion_lookup[(i, j)] = edge_records

    return distance_matrix, congestion_lookup, volatility_index


def plot_convergence(
    history_dict: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Route Optimization Convergence on Delhi Road Network",
    subtitle: str = "Mean best-found fitness +/- 1 std-dev across 30 seeded trials",
):
    """
    Generate publication-standard convergence plot: Fitness vs Iteration
    with one line per algorithm and shaded std-dev bands.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=300)

    # Determine max iterations
    first_algo = next(iter(history_dict.keys()))
    max_iters = history_dict[first_algo].shape[1]
    iterations = np.arange(max_iters)

    for algo_key, histories in history_dict.items():
        label = ALGORITHM_DISPLAY_NAMES.get(algo_key, algo_key)
        color = ALGORITHM_COLORS.get(algo_key, "#333333")

        mean_curve = np.mean(histories, axis=0)
        std_curve = np.std(histories, axis=0)

        is_dijkstra = (algo_key == "dijkstra_nn")
        linestyle = "--" if is_dijkstra else "-"
        linewidth = 2.4 if algo_key == "va_qpso" else (1.8 if not is_dijkstra else 1.6)
        alpha = 0.95 if algo_key == "va_qpso" else 0.85

        line = ax.plot(
            iterations,
            mean_curve,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )

        if not is_dijkstra:
            ax.fill_between(
                iterations,
                np.maximum(0.0, mean_curve - std_curve),
                mean_curve + std_curve,
                color=color,
                alpha=0.18,
            )

    ax.set_title(f"{title}\n({subtitle})", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Optimization Iteration / Generation", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Best-Found Route Fitness Score (s)", fontsize=11, fontweight="semibold")
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.set_xlim(0, max_iters - 1)

    legend = ax.legend(loc="upper right", framealpha=0.92, fontsize=9.5, facecolor="white", edgecolor="#cccccc")
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence plot saved successfully to: {output_path.resolve()}")


def run_convergence_experiment(
    stops: List[str],
    network_graph: NetworkGraph,
    num_seeds: int = 30,
    start_seed: int = 42,
    tier: str = "medium",
    tolerance: float = 1e-4,
    output_plot: str = "results/convergence_comparison.png",
    output_json: str = "results/convergence_metrics.json",
    output_csv: str = "results/convergence_metrics.csv",
) -> Dict[str, Any]:
    """
    Run convergence benchmarking across all 5 algorithms over identical seeded trials.
    Computes convergence speed (iterations to 95%) and hit rate vs best-known solutions.
    """
    n = len(stops)
    budget = default_budget(n)
    max_iters = budget[1]

    algorithms = ["va_qpso", "fixed_beta_qpso", "standard_pso", "ga", "dijkstra_nn"]
    seeds = [start_seed + i for i in range(num_seeds)]

    print("\n" + "=" * 92)
    print(f" RUNNING ROUTE OPTIMIZATION CONVERGENCE BENCHMARK ({num_seeds} Seeded Trials)")
    print(f" Problem Size: {n} Stops | Scenario Tier: {tier.upper()} | Max Iterations: {max_iters}")
    print(f" Algorithms: VA-QPSO, Fixed-Beta QPSO, Standard PSO, Permutation GA, {DIJKSTRA_NN_LABEL}")
    print("=" * 92 + "\n")

    # Storage for histories and scores: algo -> array of shape (num_seeds, max_iters)
    histories: Dict[str, np.ndarray] = {
        algo: np.zeros((num_seeds, max_iters), dtype=float) for algo in algorithms
    }
    final_scores: Dict[str, np.ndarray] = {
        algo: np.zeros(num_seeds, dtype=float) for algo in algorithms
    }
    elapsed_times: Dict[str, List[float]] = {algo: [] for algo in algorithms}

    # Execute seeded trials
    for s_idx, seed in enumerate(seeds):
        dist_mat, cong_lookup, vol_idx = build_scenario_distance_and_congestion(
            network_graph, stops, tier=tier, seed=seed
        )

        for algo in algorithms:
            t0 = time.perf_counter()
            if algo in ("va_qpso", "fixed_beta_qpso"):
                order, score, hist = qpso_replan(
                    stops=stops,
                    distance_matrix=dist_mat,
                    congestion_lookup=cong_lookup,
                    volatility_index=vol_idx,
                    algorithm=algo,
                    seed=seed,
                    return_history=True,
                )
            elif algo == "standard_pso":
                order, score, hist = pso_replan(
                    stops=stops,
                    distance_matrix=dist_mat,
                    congestion_lookup=cong_lookup,
                    volatility_index=vol_idx,
                    seed=seed,
                    return_history=True,
                )
            elif algo == "ga":
                order, score, hist = ga_replan(
                    stops=stops,
                    distance_matrix=dist_mat,
                    congestion_lookup=cong_lookup,
                    volatility_index=vol_idx,
                    seed=seed,
                    return_history=True,
                )
            elif algo == "dijkstra_nn":
                order, score, hist = dijkstra_replan(
                    stops=stops,
                    distance_matrix=dist_mat,
                    congestion_lookup=cong_lookup,
                    volatility_index=vol_idx,
                    start_idx=0,
                    return_history=True,
                    max_iterations=max_iters,
                )
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

            elapsed = time.perf_counter() - t0
            histories[algo][s_idx, :] = hist
            final_scores[algo][s_idx] = score
            elapsed_times[algo].append(elapsed)

        if (s_idx + 1) % 5 == 0 or (s_idx + 1) == num_seeds:
            print(f"  Completed trials [{s_idx + 1:02d}/{num_seeds:02d}]...")

    # Determine reference best solution per trial across ALL algorithms
    # ref_bests[s_idx] = min fitness achieved by ANY algorithm for that seed instance
    ref_bests = np.min(np.column_stack([final_scores[algo] for algo in algorithms]), axis=1)
    global_best_known = float(np.min(ref_bests))

    summary_rows = []
    print("\n" + "=" * 105)
    print(f" CONVERGENCE & HIT-RATE SUMMARY REPORT ({num_seeds} Seeded Instances, Delhi Network)")
    print(f" Reference Benchmark: Best fitness found by ANY algorithm on each scenario instance")
    print(f" Global Best Solution across all runs: {global_best_known:.4f} s")
    print("=" * 105)
    header = (
        f"{'Algorithm':<38} | {'Mean Score (s)':<14} | {'Std Dev':<9} | "
        f"{'Iter to 95%':<12} | {'Hit Rate (%)':<12} | {'Avg Runtime':<11}"
    )
    print(header)
    print("-" * len(header))

    metrics_export = []

    for algo in algorithms:
        name = ALGORITHM_DISPLAY_NAMES.get(algo, algo)
        scores = final_scores[algo]
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        # 1. Iterations to reach 95% of run's own best fitness
        iter_95_list = [compute_iterations_to_95(histories[algo][i]) for i in range(num_seeds)]
        mean_iter_95 = float(np.mean(iter_95_list))
        std_iter_95 = float(np.std(iter_95_list))

        # 2. Hit rate: % of trials reaching within tolerance of reference best
        hit_rate = compute_hit_rate(scores, ref_bests, tol=tolerance)
        hit_rate_1pct = compute_hit_rate(scores, ref_bests * 1.01, tol=0.0)

        # Average runtime
        avg_time_ms = float(np.mean(elapsed_times[algo]) * 1000.0)

        print(
            f"{name:<38} | {mean_score:<14.4f} | {std_score:<9.4f} | "
            f"{mean_iter_95:5.1f} +/- {std_iter_95:<4.1f} | {hit_rate:5.1f}%      | {avg_time_ms:6.2f} ms"
        )

        metrics_export.append({
            "algorithm": algo,
            "display_name": name,
            "mean_fitness": round(mean_score, 4),
            "std_fitness": round(std_score, 4),
            "mean_iterations_to_95": round(mean_iter_95, 2),
            "std_iterations_to_95": round(std_iter_95, 2),
            "hit_rate_strict_pct": round(hit_rate, 2),
            "hit_rate_1pct_pct": round(hit_rate_1pct, 2),
            "avg_runtime_ms": round(avg_time_ms, 2),
        })

    print("-" * len(header))

    # Generate convergence plot
    plot_path = Path(output_plot)
    plot_convergence(histories, plot_path)

    # Also save to outputs/ directory if different
    alt_plot_path = Path("outputs/convergence_comparison.png")
    if alt_plot_path != plot_path:
        plot_convergence(histories, alt_plot_path)

    # Save metrics JSON & CSV
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "num_stops": n,
                "num_seeds": num_seeds,
                "tier": tier,
                "tolerance": tolerance,
                "global_best_known": global_best_known,
            },
            "algorithms": metrics_export,
        }, f, indent=2)
    print(f"Metrics JSON saved to: {json_path.resolve()}")

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_export).to_csv(csv_path, index=False)
    print(f"Metrics CSV saved to: {csv_path.resolve()}")

    return {
        "histories": histories,
        "final_scores": final_scores,
        "ref_bests": ref_bests,
        "metrics": metrics_export,
    }


def build_live_distance_and_congestion(
    network_graph: NetworkGraph,
    state: Dict[str, Any],
    stops: List[str],
) -> Tuple[np.ndarray, CongestionLookup]:
    """
    Build live-weighted travel-time matrix and per-leg congestion lookup from TraCI state.
    """
    edge_weights: Dict[str, float] = {}
    for edge_id, metrics in state["edges"].items():
        speed = metrics["mean_speed"]
        if speed > 0.1:
            edge_weights[edge_id] = network_graph.edges[edge_id]["length"] / speed

    adjacency = adjacency_from_network_graph(network_graph, edge_weights)
    distance_matrix = compute_distance_matrix(adjacency, stops)

    congestion_lookup: CongestionLookup = {}
    n = len(stops)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            source_node = stops[i]
            target_node = stops[j]
            edge_records = []
            for edge_id, e_data in network_graph.edges.items():
                if e_data["from"] == source_node or e_data["to"] == target_node:
                    occ = state["edges"].get(edge_id, {}).get("occupancy", 0.0)
                    edge_records.append({"occupancy": occ, "capacity": 1.0})
            if edge_records:
                congestion_lookup[(i, j)] = edge_records

    return distance_matrix, congestion_lookup


def run_single_trial(
    tier: str,
    seed: int,
    algorithm: str,
    duration: int,
    stops: List[str],
    network_graph: NetworkGraph,
    use_libsumo: bool = False,
) -> Dict[str, Any]:
    """
    Execute one simulation run for a given (tier, seed, algorithm) configuration.
    Supports all 5 algorithms: va_qpso, fixed_beta_qpso, standard_pso, ga, dijkstra_nn.
    """
    sumocfg_path = f"networks/delhi/scenarios/{tier}/scenario.sumocfg"
    if not os.path.exists(sumocfg_path):
        raise FileNotFoundError(f"Scenario configuration not found: {sumocfg_path}")

    edge_ids = list(network_graph.edges.keys())
    extractor = SubscriptionStateExtractor(edge_ids, use_libsumo=use_libsumo)
    extractor.connect(sumocfg_path, use_gui=False)

    volatility_calc = NetworkVolatilityIndex(
        window_size=VOLATILITY_WINDOW,
        reference_variance=REFERENCE_VARIANCE,
    )
    arbiter = ReplanArbiter(ARBITER_WINDOW_SECONDS, ARBITER_REROUTE_THRESHOLD)

    vehicle_routes: Dict[str, List[str]] = {}
    known_vehicle_ids: set = set()

    next_replan_time = 0.0
    reroute_count = 0
    replan_count = 0
    sim_time = 0.0

    current_best_order = np.arange(len(stops))
    last_T = 0.0
    last_D = 0.0
    last_C = 0.0

    cumulative_T = 0.0
    cumulative_D = 0.0
    cumulative_C = 0.0
    active_steps = 0

    try:
        while sim_time < duration:
            try:
                sim_time = extractor.step()
            except (FatalTraCIError, TraCIException) as exc:
                print(f"    [Warning] SUMO stepped early at {sim_time:.0f}s: {exc}")
                break

            state = extractor.get_state()
            edge_mean_speeds = {e: m["mean_speed"] for e, m in state["edges"].items()}
            volatility_index = volatility_calc.update(edge_mean_speeds)

            # Reactive vehicle rerouting
            current_vehicle_ids = set(state["vehicles"].keys())
            for veh_id in current_vehicle_ids - known_vehicle_ids:
                try:
                    vehicle_routes[veh_id] = list(extractor.traci.vehicle.getRoute(veh_id))
                except Exception:
                    pass
            for veh_id in known_vehicle_ids - current_vehicle_ids:
                vehicle_routes.pop(veh_id, None)
            known_vehicle_ids = current_vehicle_ids

            for veh_id, route in list(vehicle_routes.items()):
                veh_info = state["vehicles"].get(veh_id, {})
                route_index = veh_info.get("route_index")
                if route_index is None or route_index + 1 >= len(route):
                    continue
                planned_next_edge = route[route_index + 1]

                decision = evaluate_vehicle_reroute(
                    veh_id, state, planned_next_edge, network_graph,
                    occupancy_threshold=OCCUPANCY_THRESHOLD,
                    min_occupancy_improvement=MIN_OCCUPANCY_IMPROVEMENT,
                    max_extra_distance_ratio=MAX_EXTRA_DISTANCE_RATIO,
                )
                if decision is not None:
                    new_route = list(route)
                    new_route[route_index + 1] = decision.to_edge
                    try:
                        extractor.traci.vehicle.setRoute(veh_id, new_route)
                        vehicle_routes[veh_id] = new_route
                        arbiter.record_reroute(sim_time)
                        reroute_count += 1
                    except Exception:
                        pass

            # Check replanning cadence
            is_scheduled = sim_time >= next_replan_time
            is_arbiter_triggered = arbiter.should_trigger_early_replan(sim_time)

            if is_scheduled or is_arbiter_triggered:
                distance_matrix, congestion_lookup = build_live_distance_and_congestion(
                    network_graph, state, stops
                )

                replan_seed = (seed * 10007 + replan_count * 31) % (2**31 - 1)

                if algorithm in ("va_qpso", "fixed_beta_qpso"):
                    best_order, best_score = qpso_replan(
                        stops,
                        distance_matrix,
                        congestion_lookup,
                        volatility_index=volatility_index,
                        algorithm=algorithm,
                        seed=replan_seed,
                    )
                elif algorithm in ("standard_pso", "pso"):
                    best_order, best_score = pso_replan(
                        stops,
                        distance_matrix,
                        congestion_lookup,
                        volatility_index=volatility_index,
                        seed=replan_seed,
                    )
                elif algorithm in ("ga", "permutation_ga"):
                    best_order, best_score = ga_replan(
                        stops,
                        distance_matrix,
                        congestion_lookup,
                        volatility_index=volatility_index,
                        seed=replan_seed,
                    )
                elif algorithm in ("dijkstra_nn", "dijkstra"):
                    best_order, best_score = dijkstra_replan(
                        stops,
                        distance_matrix,
                        congestion_lookup,
                        volatility_index=volatility_index,
                        start_idx=0,
                    )
                else:
                    raise ValueError(f"Unknown simulation algorithm: {algorithm}")

                current_best_order = best_order
                last_T, last_D, last_C = route_components(
                    current_best_order, distance_matrix, congestion_lookup
                )

                if algorithm == "va_qpso":
                    interval = replan_interval(volatility_index)
                else:
                    interval = N_FIXED

                next_replan_time = sim_time + interval
                arbiter.notify_replanned(sim_time)
                replan_count += 1

            if last_T > 0.0:
                cumulative_T += last_T
                cumulative_D += last_D
                cumulative_C += last_C
                active_steps += 1

    finally:
        extractor.close()

    avg_T = (cumulative_T / active_steps) if active_steps > 0 else last_T
    avg_D = (cumulative_D / active_steps) if active_steps > 0 else last_D
    avg_C = (cumulative_C / active_steps) if active_steps > 0 else last_C

    return {
        "tier": tier,
        "seed": seed,
        "algorithm": algorithm,
        "total_route_completion_time": round(avg_T, 2),
        "total_distance": round(avg_D, 2),
        "congestion_exposure_score": round(avg_C, 4),
        "reroute_count": reroute_count,
        "replan_count": replan_count,
    }


def print_statistical_summary(df: pd.DataFrame):
    """
    Format and print paired comparison statistics across evaluated algorithms.
    """
    print("\n" + "=" * 80)
    print("SIMULATION STATISTICAL COMPARISON SUMMARY")
    print("=" * 80)

    for tier in df["tier"].unique():
        sub_df = df[df["tier"] == tier]
        print(f"\n[Tier: {tier.upper()}]")
        for algo in sub_df["algorithm"].unique():
            algo_df = sub_df[sub_df["algorithm"] == algo]
            tt = algo_df["total_route_completion_time"]
            cong = algo_df["congestion_exposure_score"]
            print(f"  {algo:<20} | Time: {tt.mean():.2f} +/- {tt.std():.2f}s | Congestion: {cong.mean():.4f} +/- {cong.std():.4f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Route Optimization Experiments & Convergence Benchmarks")
    parser.add_argument("--mode", choices=["convergence", "simulation", "both"], default="convergence",
                        help="Execution mode: 'convergence' benchmark, 'simulation', or 'both' (default: convergence)")
    parser.add_argument("--num-seeds", type=int, default=30, help="Number of seeds for convergence benchmark (default: 30)")
    parser.add_argument("--start-seed", type=int, default=42, help="Starting seed value (default: 42)")
    parser.add_argument("--tiers", nargs="+", default=["medium"], help="Tiers to evaluate (default: medium)")
    parser.add_argument("--duration", type=int, default=300, help="Simulation duration in seconds (default: 300)")
    parser.add_argument("--output", type=str, default="results/experiments.csv", help="Simulation CSV path")
    parser.add_argument("--json-output", type=str, default="results/experiments.json", help="Simulation JSON path")
    parser.add_argument("--output-plot", type=str, default="results/convergence_comparison.png", help="Convergence plot PNG path")
    parser.add_argument("--use-libsumo", action="store_true", help="Use libsumo if installed")
    args = parser.parse_args()

    network_graph = NetworkGraph(NET_FILE)
    stops = pick_mutually_reachable_stops(
        adjacency_from_network_graph(network_graph, edge_weights={}),
        NUM_STOPS,
    )

    if args.mode in ("convergence", "both"):
        tier_to_run = args.tiers[0] if len(args.tiers) == 1 else "medium"
        run_convergence_experiment(
            stops=stops,
            network_graph=network_graph,
            num_seeds=args.num_seeds,
            start_seed=args.start_seed,
            tier=tier_to_run,
            output_plot=args.output_plot,
        )

    if args.mode in ("simulation", "both"):
        seeds = [args.start_seed + i for i in range(min(args.num_seeds, 10))]
        algorithms = ["va_qpso", "fixed_beta_qpso", "standard_pso", "ga", "dijkstra_nn"]
        results: List[Dict[str, Any]] = []

        total_runs = len(args.tiers) * len(seeds) * len(algorithms)
        run_idx = 0

        for tier in args.tiers:
            print(f"\n>>> Running SUMO Simulation Tier: [{tier.upper()}] <<<")
            for seed in seeds:
                for algo in algorithms:
                    run_idx += 1
                    t0 = time.time()
                    res = run_single_trial(
                        tier=tier,
                        seed=seed,
                        algorithm=algo,
                        duration=args.duration,
                        stops=stops,
                        network_graph=network_graph,
                        use_libsumo=args.use_libsumo,
                    )
                    elapsed = time.time() - t0
                    results.append(res)
                    print(
                        f"  [{run_idx:02d}/{total_runs:02d}] Tier: {tier:<6} | Seed: {seed:<3} | Algo: {algo:<15} "
                        f"| Time: {res['total_route_completion_time']:6.2f}s | Cong: {res['congestion_exposure_score']:7.4f} ({elapsed:.1f}s)"
                    )

        df = pd.DataFrame(results)
        out_csv = Path(args.output)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nSimulation CSV saved to: {out_csv.resolve()}")

        if args.json_output:
            out_json = Path(args.json_output)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Simulation JSON saved to: {out_json.resolve()}")

        print_statistical_summary(df)


if __name__ == "__main__":
    main()
