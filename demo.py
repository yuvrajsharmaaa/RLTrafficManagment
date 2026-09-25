"""
Streamlit Live & Replay Dashboard for QPSO Delivery Route Optimization.

Designed specifically for Demo Day:
- Displays live metrics (Volatility Index, Adaptive Cadence N, Replan/Reroute counts).
- Shows current delivery tour stop sequence.
- Running live feed of events (scheduled replans, arbiter-triggered replans, reactive reroutes).
- Supports both Live SUMO execution and instant zero-risk Replay from event log.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.planner.fitness import CongestionLookup, route_components, score_route
from src.planner.qpso import replan as qpso_replan
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

# Page Configuration
st.set_page_config(
    page_title="Ambulance Emergency Route Dispatch — Live Monitor",
    page_icon="🚑",
    layout="wide",
)

# Constants
NET_FILE = "networks/delhi/delhi_intersection.net.xml"
DEFAULT_SUMOCFG = "networks/delhi/scenarios/medium/scenario.sumocfg"
DEFAULT_LOG = "logs/hybrid_run.jsonl"
NUM_STOPS = 8
N_MAX = 120.0
N_MIN = 20.0
OCCUPANCY_THRESHOLD = 0.8
MIN_OCCUPANCY_IMPROVEMENT = 0.15
MAX_EXTRA_DISTANCE_RATIO = 0.3
ARBITER_WINDOW_SECONDS = 60.0
ARBITER_REROUTE_THRESHOLD = 5
VOLATILITY_WINDOW = 15
REFERENCE_VARIANCE = 0.002


def replan_interval(volatility_index: float) -> float:
    return N_MAX - (N_MAX - N_MIN) * volatility_index


def build_live_distance_matrix(network_graph: NetworkGraph, state: Dict[str, Any], stops: List[str]):
    edge_weights: Dict[str, float] = {}
    for edge_id, metrics in state["edges"].items():
        speed = metrics["mean_speed"]
        if speed > 0.1:
            edge_weights[edge_id] = network_graph.edges[edge_id]["length"] / speed
    adjacency = adjacency_from_network_graph(network_graph, edge_weights)
    return compute_distance_matrix(adjacency, stops)


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🚑 Ambulance Emergency Route Dispatch — Live Monitor")
st.caption("Tracks live traffic volatility, adapts replanning cadence, and executes real-time detour routing for Delhi hospital corridors.")

# ----------------------------------------------------------------------
# Sidebar Controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Simulation Controls")
    mode = st.radio(
        "Execution Mode",
        ["Live SUMO Simulation", "Replay Event Log (logs/hybrid_run.jsonl)"],
        index=0,
        help="Select Live to run TraCI in real-time, or Replay to inspect recorded events with zero latency.",
    )

    if mode == "Live SUMO Simulation":
        tier = st.selectbox(
            "Scenario Volatility Tier",
            ["medium", "high", "low"],
            index=0,
            format_func=lambda x: f"{x.upper()} Volatility" + (" (Road Closure @ 120s)" if x == "medium" else " (Compound Closure + Surge)" if x == "high" else " (Smooth Flow)")
        )
        sim_duration = st.slider("Simulation Duration (seconds)", min_value=60, max_value=600, value=300, step=30)
        sim_speed = st.slider("Step Sleep (sec)", min_value=0.0, max_value=0.2, value=0.01, step=0.01, help="Slow down simulation for visual inspection.")
        start_btn = st.button("Start Live Simulation", type="primary")
    else:
        log_path_input = st.text_input("Event Log Path", value=DEFAULT_LOG)
        replay_speed = st.slider("Replay Delay (sec)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        start_btn = st.button("Start Replay", type="primary")

# ----------------------------------------------------------------------
# Dashboard Layout Placeholders
# ----------------------------------------------------------------------
kpi_cols = st.columns(5)
metric_sim_time = kpi_cols[0].empty()
metric_volatility = kpi_cols[1].empty()
metric_interval = kpi_cols[2].empty()
metric_replans = kpi_cols[3].empty()
metric_reroutes = kpi_cols[4].empty()

st.divider()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📍 Active Corridor Stops")
    route_status_box = st.empty()
    route_table_box = st.empty()

with col_right:
    st.subheader("⚡ Decision Event Feed")
    event_summary_box = st.empty()
    event_table_box = st.empty()


def render_kpis(sim_time: float, vol_idx: float, interval: float, replan_count: int, reroute_count: int):
    metric_sim_time.metric("Simulation Time", f"{sim_time:.0f} s")
    vol_label = "🔴 HIGH" if vol_idx > 0.45 else ("🟡 MODERATE" if vol_idx > 0.25 else "🟢 CALM")
    metric_volatility.metric("Volatility Index", f"{vol_idx:.4f}", vol_label)
    metric_interval.metric("Adaptive Cadence N", f"{interval:.1f} s")
    metric_replans.metric("Total Replans", replan_count)
    metric_reroutes.metric("Reactive Reroutes", reroute_count)


def render_route(stops: List[str], best_order: List[int], fitness: float):
    route_status_box.info(f"**Current Tour Fitness (Travel Time)**: `{fitness:.2f} s` | Stops: `{len(stops)}`")
    ordered_stops = [stops[i] for i in best_order]
    route_df = pd.DataFrame({
        "Stop #": [f"Stop {k+1}" for k in range(len(ordered_stops))],
        "Node ID": ordered_stops,
        "Order Index": best_order,
    })
    route_table_box.dataframe(route_df, width="stretch", hide_index=True)


def render_event_feed(events: List[Dict[str, Any]]):
    if not events:
        event_table_box.write("No events recorded yet.")
        return

    # Show latest events first
    display_rows = []
    for ev in reversed(events[-20:]):
        ev_type = ev.get("event", "").upper()
        sim_t = ev.get("sim_time", 0.0)
        vol = ev.get("volatility_index", 0.0)

        if ev_type == "REPLAN":
            trigger = ev.get("trigger", "scheduled").upper()
            fit = ev.get("fitness", 0.0)
            next_int = ev.get("next_interval_seconds", 0.0)
            detail = f"Trigger: {trigger} | Fitness: {fit:.2f}s | Next Cadence: {next_int:.1f}s"
        else:
            veh = ev.get("vehicle_id", "")
            from_e = ev.get("from_edge", "")
            to_e = ev.get("to_edge", "")
            detail = f"Vehicle: {veh} | Detour: {from_e} -> {to_e}"

        display_rows.append({
            "Time (s)": f"{sim_t:.0f}",
            "Event": ev_type,
            "Volatility": f"{vol:.4f}",
            "Details": detail,
        })

    event_df = pd.DataFrame(display_rows)
    event_table_box.dataframe(event_df, width="stretch", hide_index=True)


# ----------------------------------------------------------------------
# Execution Logic
# ----------------------------------------------------------------------
if start_btn:
    if mode == "Replay Event Log (logs/hybrid_run.jsonl)":
        if not os.path.exists(log_path_input):
            st.error(f"Event log not found at: {log_path_input}. Run `run_hybrid.py` first to generate a log.")
        else:
            with open(log_path_input, "r", encoding="utf-8") as f:
                lines = f.readlines()

            st.success(f"Loaded {len(lines)} events from `{log_path_input}`. Replay in progress...")
            all_events = []
            replan_count = 0
            reroute_count = 0

            for line in lines:
                ev = json.loads(line.strip())
                all_events.append(ev)

                ev_type = ev.get("event")
                sim_time = ev.get("sim_time", 0.0)
                vol_idx = ev.get("volatility_index", 0.0)
                interval = ev.get("next_interval_seconds", replan_interval(vol_idx))

                if ev_type == "replan":
                    replan_count += 1
                    stops = ev.get("stops", [])
                    best_order = ev.get("best_order", list(range(len(stops))))
                    fitness = ev.get("fitness", 0.0)
                    render_route(stops, best_order, fitness)
                elif ev_type == "reroute":
                    reroute_count += 1

                render_kpis(sim_time, vol_idx, interval, replan_count, reroute_count)
                render_event_feed(all_events)
                if replay_speed > 0:
                    time.sleep(replay_speed)

            st.success("Replay complete.")

    else:
        # Live Simulation Mode
        sumocfg_path = f"networks/delhi/scenarios/{tier}/scenario.sumocfg"
        network_graph = NetworkGraph(NET_FILE)
        edge_ids = list(network_graph.edges.keys())
        free_flow_adj = adjacency_from_network_graph(network_graph, edge_weights={})
        stops = pick_mutually_reachable_stops(free_flow_adj, NUM_STOPS)

        extractor = SubscriptionStateExtractor(edge_ids, use_libsumo=False)
        try:
            extractor.connect(sumocfg_path, use_gui=False)
        except Exception as exc:
            st.error(f"Failed to connect to SUMO: {exc}")
            st.stop()

        volatility_calc = NetworkVolatilityIndex(
            window_size=VOLATILITY_WINDOW,
            reference_variance=REFERENCE_VARIANCE,
        )
        arbiter = ReplanArbiter(ARBITER_WINDOW_SECONDS, ARBITER_REROUTE_THRESHOLD)

        vehicle_routes: Dict[str, List[str]] = {}
        known_vehicle_ids: set = set()
        all_events = []

        next_replan_time = 0.0
        replan_count = 0
        reroute_count = 0
        sim_time = 0.0
        current_order = list(range(NUM_STOPS))
        current_fitness = 0.0

        progress_bar = st.progress(0.0)

        try:
            while sim_time < sim_duration:
                try:
                    sim_time = extractor.step()
                except (FatalTraCIError, TraCIException) as exc:
                    st.warning(f"Simulation concluded at {sim_time:.0f}s: {exc}")
                    break

                state = extractor.get_state()
                edge_mean_speeds = {e: m["mean_speed"] for e, m in state["edges"].items()}
                volatility_index = volatility_calc.update(edge_mean_speeds)
                interval = replan_interval(volatility_index)

                # Reactive next-hop evaluation
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
                            ev = {
                                "event": "reroute",
                                "sim_time": sim_time,
                                "volatility_index": volatility_index,
                                "vehicle_id": decision.vehicle_id,
                                "from_edge": decision.from_edge,
                                "to_edge": decision.to_edge,
                            }
                            all_events.append(ev)
                        except Exception:
                            pass

                # Replanning evaluation
                is_scheduled = sim_time >= next_replan_time
                is_arbiter_triggered = arbiter.should_trigger_early_replan(sim_time)

                if is_scheduled or is_arbiter_triggered:
                    distance_matrix = build_live_distance_matrix(network_graph, state, stops)
                    congestion_lookup: CongestionLookup = {}

                    best_order, best_score = qpso_replan(
                        stops, distance_matrix, congestion_lookup,
                        volatility_index=volatility_index,
                    )
                    current_order = best_order.tolist()
                    current_fitness = best_score
                    replan_count += 1

                    ev = {
                        "event": "replan",
                        "sim_time": sim_time,
                        "volatility_index": volatility_index,
                        "trigger": "arbiter" if is_arbiter_triggered else "scheduled",
                        "num_stops": NUM_STOPS,
                        "stops": stops,
                        "best_order": current_order,
                        "fitness": current_fitness,
                        "next_interval_seconds": interval,
                    }
                    all_events.append(ev)

                    next_replan_time = sim_time + interval
                    arbiter.notify_replanned(sim_time)

                # Update live UI every few steps
                if int(sim_time) % 2 == 0 or is_scheduled or is_arbiter_triggered:
                    render_kpis(sim_time, volatility_index, interval, replan_count, reroute_count)
                    render_route(stops, current_order, current_fitness)
                    render_event_feed(all_events)
                    progress_bar.progress(min(1.0, sim_time / sim_duration))

                if sim_speed > 0:
                    time.sleep(sim_speed)

        finally:
            extractor.close()

        progress_bar.progress(1.0)
        st.success(f"Live simulation complete: {sim_time:.0f}s simulated, {replan_count} replans, {reroute_count} reactive detours.")
else:
    # Initial state display before start is clicked
    st.info("Select a scenario and click **Start Live Simulation** or **Start Replay** to begin.")
    if os.path.exists(DEFAULT_LOG):
        try:
            with open(DEFAULT_LOG, "r", encoding="utf-8") as f:
                lines = [json.loads(line.strip()) for line in f if line.strip()]
            if lines:
                last_ev = lines[-1]
                render_kpis(
                    last_ev.get("sim_time", 0.0),
                    last_ev.get("volatility_index", 0.0),
                    last_ev.get("next_interval_seconds", 120.0),
                    sum(1 for e in lines if e.get("event") == "replan"),
                    sum(1 for e in lines if e.get("event") == "reroute"),
                )
                # Find last replan event for route display
                last_replan = next((e for e in reversed(lines) if e.get("event") == "replan"), None)
                if last_replan:
                    render_route(
                        last_replan.get("stops", []),
                        last_replan.get("best_order", list(range(len(last_replan.get("stops", []))))),
                        last_replan.get("fitness", 0.0),
                    )
                render_event_feed(lines)
        except Exception:
            pass
    else:
        render_kpis(0.0, 0.0, 120.0, 0, 0)
        route_status_box.info("No active corridor selected. Click Start Live Simulation or Start Replay in sidebar.")
        event_table_box.write("No events recorded yet.")
