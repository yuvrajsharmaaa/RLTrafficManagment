#!/usr/bin/env python3
"""
Real-world travel-time benchmark across 3 traffic tiers and 6 routing algorithms
using authentic SUMO ambulance drive-throughs (src/simulation/dispatch.py).

Defines arrival time unambiguously as the measured SUMO arrival time if the
vehicle reaches the destination within the 900 s cap; otherwise 'did not arrive'
with the distance covered in meters. No planner-estimate averaging.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from export_for_frontend import (
    DEFAULT_NET_FILE,
    HERO_PICKUP_JUNCTION,
    NUM_STOPS,
    load_hospitals,
)
from src.simulation.dispatch import DispatchSession, RoadModel, TIERS

ALGORITHMS = [
    "va_qpso",
    "fixed_beta_qpso",
    "standard_pso",
    "ga",
    "sa",
    "dijkstra_nn",
]

ALGO_DISPLAY_NAMES = {
    "va_qpso": "VA-QPSO (Volatility-Adaptive)",
    "fixed_beta_qpso": "Fixed-Beta QPSO (Linear Anneal)",
    "standard_pso": "Standard PSO",
    "ga": "Permutation GA",
    "sa": "Simulated Annealing",
    "dijkstra_nn": "Dijkstra (nearest-neighbor)",
}


def run_benchmark(
    tiers: List[str] = list(TIERS),
    algorithms: List[str] = ALGORITHMS,
    num_seeds: int = 5,
    start_seed: int = 42,
    output_csv: str = "results/real_travel_time_benchmark.csv",
    output_json: str = "results/real_travel_time_benchmark.json",
) -> List[Dict[str, Any]]:
    model = RoadModel.load(Path(DEFAULT_NET_FILE))
    hospitals = load_hospitals()
    hospital = hospitals[0]
    exit_junction = hospital["exit_junction"]
    waypoints = model.default_waypoints(
        NUM_STOPS - 2, exclude=[HERO_PICKUP_JUNCTION, exit_junction]
    )
    stops = (
        [("junction", HERO_PICKUP_JUNCTION)]
        + [("junction", w) for w in waypoints]
        + [("junction", exit_junction)]
    )

    seeds = [start_seed + i for i in range(num_seeds)]
    total_runs = len(tiers) * len(seeds) * len(algorithms)
    run_idx = 0

    print("=" * 88)
    print(" REAL-WORLD SUMO DRIVE-THROUGH BENCHMARK (MEASURED ARRIVALS)")
    print(f" Tiers: {tiers} | Seeds ({num_seeds}): {seeds} | Algorithms: {len(algorithms)}")
    print(f" Destination: {hospital['name']} (Exit junction: {exit_junction})")
    print(f" Max simulated drive cap: 900 s | Total runs to execute: {total_runs}")
    print("=" * 88 + "\n")

    results: List[Dict[str, Any]] = []

    for tier in tiers:
        print(f"\n>>> Running Traffic Tier: [{tier.upper()}] <<<")
        for seed in seeds:
            for algo in algorithms:
                run_idx += 1
                t0_wall = time.time()
                with DispatchSession(tier, model, sirens=False) as session:
                    plan = session.plan(stops, algo, seed)
                    res = session.drive(plan, algo, seed, replans=True, cap_s=900)
                elapsed_wall = time.time() - t0_wall

                arrived = res.arrived
                arr_time = res.elapsed_s if arrived else None
                rec = {
                    "tier": tier,
                    "seed": seed,
                    "algorithm": algo,
                    "display_name": ALGO_DISPLAY_NAMES.get(algo, algo),
                    "status": res.status,
                    "arrived": arrived,
                    "arrival_time_s": arr_time,
                    "simulated_seconds": res.elapsed_s,
                    "driven_length_m": res.driven_length_m,
                    "route_length_m": res.route_length_m,
                    "average_speed_kmh": res.average_speed_kmh,
                    "replans": sum(1 for e in res.events if e.get("type") == "replan"),
                    "wall_clock_s": round(elapsed_wall, 2),
                }
                results.append(rec)

                status_str = f"ARRIVED in {arr_time:.1f}s" if arrived else f"{res.status} ({res.driven_length_m:.0f}m covered)"
                print(
                    f"  [{run_idx:02d}/{total_runs:02d}] Tier: {tier:<6} | Seed: {seed:<2} | {algo:<16} | "
                    f"{status_str:<32} (wall: {elapsed_wall:.1f}s)"
                )

    # Save CSV and JSON
    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\nRaw results saved to: {out_csv.resolve()}")

    out_json = Path(output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"JSON results saved to: {out_json.resolve()}")

    # Print summary table
    print_summary_table(df)
    return results


def print_summary_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 96)
    print(" REAL-WORLD PERFORMANCE SUMMARY TABLE (MEASURED SUMO DRIVE-THROUGHS)")
    print("=" * 96)
    header = f"{'Tier':<8} | {'Algorithm':<28} | {'Completed':<12} | {'Mean Time (s)':<14} | {'Min (s)':<8} | {'Max (s)':<8} | {'Mean Driven (m)'}"
    print(header)
    print("-" * len(header))

    summary_rows = []
    for tier in df["tier"].unique():
        sub_tier = df[df["tier"] == tier]
        for algo in sub_tier["algorithm"].unique():
            rows = sub_tier[sub_tier["algorithm"] == algo]
            n_total = len(rows)
            arrived_rows = rows[rows["arrived"] == True]
            n_arrived = len(arrived_rows)
            comp_rate = (n_arrived / n_total) * 100.0

            if n_arrived > 0:
                mean_time_str = f"{arrived_rows['arrival_time_s'].mean():.1f} s"
                min_time_str = f"{arrived_rows['arrival_time_s'].min():.1f}"
                max_time_str = f"{arrived_rows['arrival_time_s'].max():.1f}"
            else:
                mean_time_str = "Did not arrive"
                min_time_str = "—"
                max_time_str = "—"

            mean_dist = rows["driven_length_m"].mean()
            print(
                f"{tier.upper():<8} | {algo:<28} | {n_arrived}/{n_total} ({comp_rate:4.0f}%) | "
                f"{mean_time_str:<14} | {min_time_str:<8} | {max_time_str:<8} | {mean_dist:6.0f} m"
            )
        print("-" * len(header))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run real-world SUMO drive-through benchmark")
    parser.add_argument("--tiers", nargs="+", default=["low", "medium", "high"], help="Tiers to run")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds (default: 5)")
    parser.add_argument("--start-seed", type=int, default=42, help="Start seed (default: 42)")
    args = parser.parse_args()

    run_benchmark(tiers=args.tiers, num_seeds=args.num_seeds, start_seed=args.start_seed)
