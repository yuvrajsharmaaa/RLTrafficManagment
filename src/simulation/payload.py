"""
Frontend payloads for dispatch runs (recorded hero files and live responses).

Field names match the existing contract (stops, path, metrics_over_time,
events, completion_time, ...). New fields say where each number comes from:

- timing.kind: "simulated_drive" (an ambulance driven through SUMO; the
  time is SUMO's) or "planner_estimate" (sum of per-edge travel times at the
  speeds measured at dispatch; no vehicle was driven).
- timing.status: "arrived", "not_arrived_within_cap", "teleported" or
  "estimate". completion_time is the measured arrival only when status is
  "arrived"; otherwise it is the time simulated before the run stopped, or
  the planner estimate.
- final_leg: the unsimulated part of the trip, from the network exit to the
  hospital, as a straight-line distance. It is never converted to time and
  never added to completion_time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.planner.qpso import DEFAULT_BETA_MIN, va_beta_floor
from src.simulation.dispatch import (
    DISPATCH_S,
    DRIVE_CAP_S,
    SUMO_SEED,
    DriveResult,
    RoadModel,
    TrafficSample,
)

HOSPITAL_PUBLIC_KEYS = (
    "name", "lat", "lon", "level", "coord_source", "in_network", "exit_junction",
    "exit_lat", "exit_lon", "straight_line_from_exit_m",
)


def tier_for(volatility: float) -> str:
    """Unpredictability tier of V (not congestion): calm / moderate / turbulent."""
    if volatility < 0.25:
        return "calm"
    if volatility < 0.50:
        return "moderate"
    return "turbulent"


def beta_floor(algorithm: str, volatility: float) -> float:
    return va_beta_floor(volatility) if algorithm == "va_qpso" else DEFAULT_BETA_MIN


def hospital_record(h: Dict[str, Any], selected_name: str) -> Dict[str, Any]:
    rec = {k: h[k] for k in HOSPITAL_PUBLIC_KEYS if k in h}
    rec["status"] = "selected_destination" if h["name"] == selected_name else "alternate_destination"
    return rec


def stop_records(
    model: RoadModel,
    ordered: Sequence[Tuple[str, str]],
    hospital: Dict[str, Any],
    pickup_label: str,
    reached: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Stops in visit order: pickup, planner waypoints, network exit."""
    records = []
    last = len(ordered) - 1
    for idx, (_, jid) in enumerate(ordered):
        lat, lon = model.junction_lat_lon(jid)
        if idx == 0:
            kind, label = "pickup", pickup_label
        elif idx == last:
            kind = "network_exit"
            label = f"Network exit toward {hospital['name']}"
        else:
            kind, label = "waypoint", f"Planner waypoint {idx} (auto-selected junction, not a real place)"
        rec = {"id": jid, "lat": round(lat, 6), "lon": round(lon, 6), "label": label, "kind": kind}
        if jid in reached and idx > 0:
            rec["reached_t"] = reached[jid]
        records.append(rec)
    return records


def traffic_record(sample: TrafficSample, algorithm: str) -> Dict[str, Any]:
    return {
        "t": sample.t,
        "volatility_index": round(sample.volatility_index, 4),
        "beta": round(beta_floor(algorithm, sample.volatility_index), 4),
        "tier": tier_for(sample.volatility_index),
        "vehicles": sample.vehicles,
        "mean_vehicle_speed_kmh": sample.mean_vehicle_speed_kmh,
        "stopped_vehicles": sample.stopped_vehicles,
    }


def planned_path(model: RoadModel, route_edges: Sequence[str], edge_time) -> List[Dict[str, float]]:
    """Route geometry timed by the planner's per-edge estimate (not a drive)."""
    points: List[Dict[str, float]] = []
    t = 0.0
    for eid in route_edges:
        edge = model.net.getEdge(eid)
        shape = edge.getShape()
        seg = [((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a, b in zip(shape[1:], shape[:-1])]
        total = sum(seg) or 1.0
        dt = edge_time(edge)
        acc = 0.0
        for k, xy in enumerate(shape):
            if k:
                acc += seg[k - 1]
            lat, lon = model.lat_lon(xy)
            pt = {"t": round(t + dt * acc / total, 2), "lat": round(lat, 6), "lon": round(lon, 6)}
            if not points or (pt["lat"], pt["lon"]) != (points[-1]["lat"], points[-1]["lon"]):
                points.append(pt)
        t += dt
    return points


def build_payload(
    *,
    model: RoadModel,
    scenario: str,
    tier: str,
    algorithm: str,
    seed: int,
    plan: Dict[str, Any],
    dispatch_sample: TrafficSample,
    hospital: Dict[str, Any],
    hospitals: List[Dict[str, Any]],
    pickup_label: str,
    pickup_snap_m: Optional[float],
    drive: Optional[DriveResult],
    edge_time,
    sirens: bool,
) -> Dict[str, Any]:
    v0 = dispatch_sample.volatility_index
    dispatch_event = {
        "t": 0.0,
        "type": "dispatch",
        "volatility_index": round(v0, 4),
        "beta": round(beta_floor(algorithm, v0), 4),
        "detail": (
            f"Ambulance dispatched toward {hospital['name']} (network exit {hospital['exit_junction']}). "
            f"Traffic at dispatch: {dispatch_sample.vehicles} vehicles, mean speed "
            f"{dispatch_sample.mean_vehicle_speed_kmh} km/h, {dispatch_sample.stopped_vehicles} stopped; "
            f"unpredictability {v0:.2f} ({tier_for(v0)})."
        ),
    }
    if drive is not None:
        metrics = [traffic_record(dispatch_sample, algorithm)] + [traffic_record(s, algorithm) for s in drive.samples]
        path = drive.trajectory
        events = [dispatch_event] + drive.events
        reached = drive.waypoint_times
        seconds = drive.elapsed_s
        timing = {
            "kind": "simulated_drive",
            "status": drive.status,
            "seconds_to_exit": drive.elapsed_s if drive.arrived else None,
            "simulated_seconds": drive.elapsed_s,
            "cap_s": DRIVE_CAP_S,
            "driven_length_m": drive.driven_length_m,
            "average_speed_kmh": round(drive.average_speed_kmh, 1) if drive.average_speed_kmh else None,
            "planner_estimate_s": round(plan["estimate_s"], 1),
        }
        path_source = "simulated_ambulance_trajectory"
    else:
        metrics = [traffic_record(dispatch_sample, algorithm)]
        path = planned_path(model, plan["route_edges"], edge_time)
        events = [dispatch_event]
        reached = {}
        seconds = round(plan["estimate_s"], 1)
        timing = {
            "kind": "planner_estimate",
            "status": "estimate",
            "seconds_to_exit": seconds,
            "simulated_seconds": None,
            "cap_s": None,
            "driven_length_m": None,
            "average_speed_kmh": None,
            "planner_estimate_s": seconds,
        }
        path_source = "planned_route_timed_by_planner_estimate"
    timing.update({
        "route_length_m": round(plan["route_length_m"], 1),
        "dispatch_sim_time_s": DISPATCH_S,
        "sumo_seed": SUMO_SEED,
        "sirens": sirens,
    })
    return {
        "scenario": scenario,
        "tier": tier,
        "algorithm": algorithm,
        "seed": seed,
        "stops": stop_records(model, plan["ordered_stops"], hospital, pickup_label, reached),
        "selected_hospital": hospital_record(hospital, hospital["name"]),
        "hospital_candidates": [hospital_record(h, hospital["name"]) for h in hospitals],
        "path": path,
        "path_source": path_source,
        "metrics_over_time": metrics,
        "events": events,
        "completion_time": seconds,
        "timing": timing,
        "traffic_at_dispatch": traffic_record(dispatch_sample, algorithm),
        "final_leg": {
            "kind": "straight_line_estimate",
            "from_junction": hospital["exit_junction"],
            "to": hospital["name"],
            "straight_line_m": hospital["straight_line_from_exit_m"],
            "simulated": False,
        },
        "pickup_snap_m": None if pickup_snap_m is None else round(pickup_snap_m, 1),
        "best_score_history": [round(x, 3) for x in plan["best_score_history"]],
    }
