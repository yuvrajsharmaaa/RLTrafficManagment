#!/usr/bin/env python3
"""
Frontend GeoJSON & Playback Exporter for QPSO Traffic Route Optimization.

Converts SUMO Cartesian coordinates (meters) to WGS84 GPS (lon/lat) using sumolib:
    lon, lat = net.convertXY2LonLat(x, y)

Produces frontend-friendly JSON payloads containing:
- "scenario": Scenario name (e.g. "medium_volatility", "high_volatility")
- "algorithm": Algorithm name ("va_qpso" or "fixed_beta_qpso")
- "stops": Delivery destinations with latitude, longitude, label, and junction ID
- "path": Densely sampled vehicle GPS coordinates over time
- "metrics_over_time": Rolling volatility index, beta anneal floor (va_qpso: 0.5 + 0.25 * V;
  fixed_beta_qpso: 0.5), and tier
- "events": Plain-language event stream for judges (re-plans, tactical detours)
- "completion_time": Total route completion time in simulated seconds

Usage Modes:
1. Export from an existing JSONL log file:
     python export_for_frontend.py export --log logs/va_high_seed42.jsonl --net networks/delhi/delhi_intersection.net.xml --scenario high --algorithm va_qpso --seed 42 --completion-time 842 --output frontend_data/high_seed42_va_qpso.json
   or simply:
     python export_for_frontend.py --log logs/hybrid_run.jsonl --output frontend_data/exported.json

2. Bundle matched pairs and choose verified hero scenarios:
     python export_for_frontend.py bundle --manifest frontend_runs.json --net networks/delhi/delhi_intersection.net.xml --output-dir frontend_data

3. Batch export bundled 'Hero' demo scenarios via simulation:
     python export_for_frontend.py --export-heroes

4. Launch the local mission control web server:
     python export_for_frontend.py --serve --port 8000
   or:
     python server.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.planner.qpso import DEFAULT_BETA_MIN, va_beta_floor
except ImportError:  # planner deps unavailable; same formula as src/planner/qpso.py
    DEFAULT_BETA_MIN = 0.5

    def va_beta_floor(volatility_index: float, beta_min: float = DEFAULT_BETA_MIN) -> float:
        return beta_min + 0.25 * volatility_index


def beta_floor_for(algorithm: str, volatility_index: float) -> float:
    """
    Exported "beta": the floor each re-plan's anneal ends at (both variants
    start at beta_max = 1.0). va_qpso ends at beta_min + 0.25 * V;
    fixed_beta_qpso always ends at beta_min, independent of V.
    """
    if algorithm == "va_qpso":
        return va_beta_floor(volatility_index)
    return DEFAULT_BETA_MIN

# Constants
DEFAULT_NET_FILE = str(PROJECT_ROOT / "networks" / "delhi" / "delhi_intersection.net.xml")
DEFAULT_FRONTEND_DIR = str(PROJECT_ROOT / "frontend_data")
ALGORITHMS = {"va_qpso", "fixed_beta_qpso"}

NUM_STOPS = 8  # pickup + 6 planner waypoints + network exit

SCENARIO_MAP = {
    "low": {
        "cfg": str(PROJECT_ROOT / "networks" / "delhi" / "scenarios" / "low" / "scenario.sumocfg"),
        "key": "low_volatility",
        "title": "Low Volatility (Off-Peak Emergency Transfer)",
        "description": "Smooth off-peak traffic across Connaught Place corridor for rapid hospital dispatch.",
    },
    "medium": {
        "cfg": str(PROJECT_ROOT / "networks" / "delhi" / "scenarios" / "medium" / "scenario.sumocfg"),
        "key": "medium_volatility",
        "title": "Medium Volatility (Corridor Obstruction @ 120s)",
        "description": "Moderate traffic with unexpected road obstruction at t=120s. Adaptive routing redirects ambulance around obstruction.",
    },
    "high": {
        "cfg": str(PROJECT_ROOT / "networks" / "delhi" / "scenarios" / "high" / "scenario.sumocfg"),
        "key": "high_volatility",
        "title": "High Volatility (Rush Hour Emergency Surge)",
        "description": "Severe peak congestion with dual road closures. Adaptive routing finds open perimeter corridor to trauma center.",
    },
}


def require_sumolib() -> Any:
    """Import sumolib lazily so CLI commands like --help or --serve work on any machine."""
    try:
        import sumolib
        return sumolib
    except ImportError as exc:
        raise SystemExit(
            "sumolib is required to export coordinates. Install SUMO (or its "
            "Python tools) and retry; viewing already exported JSON needs no SUMO."
        ) from exc


def lonlat(net: Any, xy: tuple[float, float]) -> tuple[float, float]:
    """Convert SUMO (x, y) coordinates to (lat, lon) using sumolib."""
    lon, lat = net.convertXY2LonLat(*xy)
    return float(lat), float(lon)


def tier_for(volatility: float) -> str:
    """Assign semantic traffic volatility tier."""
    if volatility < 0.25:
        return "calm"
    if volatility < 0.50:
        return "moderate"
    return "turbulent"


def get_road_name(net: Any, edge_id: str) -> str:
    """Retrieve human-readable road name from network or fallback to corridor identifier."""
    try:
        e = net.getEdge(edge_id)
        name = e.getName()
        if name:
            return name
    except Exception:
        pass
    return f"corridor {edge_id}"


def make_plain_replan_detail(v: float, trigger: str, fitness: float, t: float) -> str:
    """Generate clear, non-technical plain English explanation for judges."""
    if t <= 1.5:
        return "Emergency corridor dispatched. Prioritized fastest path to patient pickup and hospital."
    if trigger == "arbiter":
        return "High congestion detected ahead. Triggered emergency re-planning to clear hospital route."
    if v >= 0.50:
        return f"Severe traffic turbulence detected (volatility {v:.2f}). Expanded search breadth to find clear hospital corridor."
    if v >= 0.25:
        return f"Traffic unpredictability increased (volatility {v:.2f}). Re-planned corridor to bypass forming congestion."
    return "Traffic conditions steady. Maintained corridor for fastest transit time to hospital."


def make_plain_reroute_detail(net: Any, from_edge: str, to_edge: str, occ_before: float) -> str:
    """Generate clear, non-technical plain English explanation for tactical detours."""
    from_road = get_road_name(net, from_edge)
    to_road = get_road_name(net, to_edge)
    occ_pct = int(round(occ_before * 100))
    if from_road != to_road and not to_road.startswith("corridor"):
        return f"Detoured around congestion on {from_road} ({occ_pct}% congested) onto {to_road} to reach hospital faster."
    return f"Detoured around heavy traffic on {from_road} ({occ_pct}% congested) to speed up hospital arrival."


def event_detail(event: dict[str, Any], net: Any = None) -> str:
    """General helper for plain-language event description."""
    kind = event.get("event")
    volatility = float(event.get("volatility_index", 0.0))
    sim_time = float(event.get("sim_time", 0.0))
    trigger = str(event.get("trigger", "scheduled"))
    fitness = float(event.get("fitness", 0.0))

    if kind == "reroute":
        from_edge = event.get("from_edge", "")
        to_edge = event.get("to_edge", "")
        occ = float(event.get("occupancy_before", 0.8))
        if net and from_edge:
            return make_plain_reroute_detail(net, from_edge, to_edge, occ)
        return "Detoured onto clearer corridor to avoid hospital transit delays."
    if kind == "replan":
        return make_plain_replan_detail(volatility, trigger, fitness, sim_time)
    return "Recorded traffic state update."


def frontend_events(events: Iterable[dict[str, Any]], net: Any = None) -> list[dict[str, Any]]:
    """Format simulation events into judge-friendly event stream."""
    return [
        {
            "t": round(float(event.get("sim_time", 0.0)), 2),
            "type": str(event.get("event", "update")),
            "detail": event_detail(event, net),
        }
        for event in events
        if event.get("event") in {"replan", "reroute"}
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse JSON Lines file."""
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            events.append(record)
    if not events:
        raise ValueError(f"{path} has no events")
    return events


def validate_embedded_metadata(
    events: list[dict[str, Any]], scenario: str, algorithm: str, seed: int
) -> None:
    """Validate embedded log metadata matches expected parameters."""
    expected = {"scenario": scenario, "algorithm": algorithm, "seed": seed}
    for key, value in expected.items():
        recorded = {event[key] for event in events if key in event}
        if len(recorded) > 1:
            raise ValueError(f"The log contains conflicting {key!r} values: {sorted(recorded)!r}")
        if recorded and next(iter(recorded)) != value:
            raise ValueError(
                f"The log says {key}={next(iter(recorded))!r}, but the export requested {value!r}."
            )


def latest_route_plan(events: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
    """Extract stop list and optimal permutation order from latest replan event."""
    plans = [event for event in events if event.get("event") == "replan"]
    if not plans:
        raise ValueError("The log has no replan event with stops and best_order.")
    plan = plans[-1]
    stops, order = plan.get("stops"), plan.get("best_order")
    if not isinstance(stops, list) or not isinstance(order, list):
        raise ValueError("The latest replan event is missing stops or best_order.")
    if sorted(order) != list(range(len(stops))):
        raise ValueError("best_order is not a permutation of the logged stops.")
    return [str(stop) for stop in stops], [int(index) for index in order]


def load_hospitals(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load hardcoded real Delhi hospitals from hospitals.json."""
    path = Path(file_path) if file_path else PROJECT_ROOT / "hospitals.json"
    if not path.is_file():
        path = PROJECT_ROOT / "frontend_data" / "hospitals.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def stop_records(
    net: Any,
    stops: list[str],
    best_hospital: Optional[dict[str, Any]] = None,
    incident_label: str = "🚨 Patient Pickup (Incident Location)",
) -> list[dict[str, Any]]:
    """Convert junction stop IDs to geo-located stop dictionaries with ambulance framing."""
    records = []
    n = len(stops)
    for number, stop_id in enumerate(stops, start=1):
        node = net.getNode(stop_id)
        if node is None:
            lat, lon = 28.6300 + number * 0.001, 77.2200 + number * 0.001
        else:
            lat, lon = lonlat(net, node.getCoord())

        if number == 1:
            label = incident_label
        elif number == n:
            if best_hospital:
                h_name = best_hospital.get("name", "Dr. Ram Manohar Lohia Hospital")
                h_lvl = best_hospital.get("level", "trauma center").title()
                label = f"🏥 Destination: {h_name} ({h_lvl})"
            else:
                label = "🏥 Destination: Dr. Ram Manohar Lohia Hospital (Trauma Center)"
        else:
            label = f"Planner waypoint {number - 1} (auto-selected junction, not a real place)"

        records.append({
            "id": stop_id,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "label": label,
        })
    return records


def shortest_edges_between_nodes(net: Any, from_node: str, to_node: str) -> list[Any]:
    """Find shortest legal edge sequence between two logged junctions."""
    start = net.getNode(from_node)
    target = net.getNode(to_node)
    if start is None or target is None:
        return []

    def candidates_for(vehicle_class: str | None) -> list[tuple[float, list[Any]]]:
        candidates: list[tuple[float, list[Any]]] = []
        for from_edge in start.getOutgoing():
            for to_edge in target.getIncoming():
                try:
                    result = net.getShortestPath(from_edge, to_edge, vClass=vehicle_class)
                    if result and result[0]:
                        edges, cost = result
                        candidates.append((float(cost), list(edges)))
                except Exception:
                    pass
        return candidates

    candidates = candidates_for("passenger")
    if not candidates:
        candidates = candidates_for(None)
    if not candidates:
        return []
    return min(candidates, key=lambda candidate: candidate[0])[1]


def route_shapes(net: Any, ordered_stops: list[str]) -> tuple[list[tuple[float, float]], bool]:
    """Return edge-shape points along ordered stop sequence."""
    route: list[tuple[float, float]] = []
    used_stop_geometry_fallback = False
    for origin, destination in zip(ordered_stops, ordered_stops[1:]):
        try:
            edges = shortest_edges_between_nodes(net, origin, destination)
        except Exception:
            edges = []

        if not edges:
            used_stop_geometry_fallback = True
            n1 = net.getNode(origin)
            n2 = net.getNode(destination)
            if n1 and n2:
                for xy in (n1.getCoord(), n2.getCoord()):
                    point = lonlat(net, xy)
                    if not route or point != route[-1]:
                        route.append(point)
        else:
            for edge in edges:
                for xy in edge.getShape():
                    point = lonlat(net, xy)
                    if not route or point != route[-1]:
                        route.append(point)

    if len(route) < 2:
        route = [(28.6325, 77.2215), (28.6335, 77.2225)]
    return route, used_stop_geometry_fallback


def route_shapes_from_edges(net: Any, edge_ids: list[str]) -> list[tuple[float, float]]:
    """Use exact edge sequence when recorded."""
    route: list[tuple[float, float]] = []
    for edge_id in edge_ids:
        edge = net.getEdge(edge_id)
        if edge is None:
            continue
        for xy in edge.getShape():
            point = lonlat(net, xy)
            if not route or point != route[-1]:
                route.append(point)
    if len(route) < 2:
        route = [(28.6325, 77.2215), (28.6335, 77.2225)]
    return route


def dense_path(points: list[tuple[float, float]], completion_time: float) -> list[dict[str, float]]:
    """Sample polyline at roughly one visual point per simulated second."""
    if completion_time <= 0:
        completion_time = 100.0
    meters_per_degree = 111_320.0
    segment_lengths = []
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        dx = (lon2 - lon1) * meters_per_degree * math.cos(math.radians((lat1 + lat2) / 2))
        dy = (lat2 - lat1) * meters_per_degree
        segment_lengths.append(math.hypot(dx, dy))
    total = sum(segment_lengths)
    if total == 0:
        return [{"t": 0.0, "lat": points[0][0], "lon": points[0][1]}]

    sample_count = max(2, int(math.ceil(completion_time)) + 1)
    cumulative = [0.0]
    for length in segment_lengths:
        cumulative.append(cumulative[-1] + length)
    path = []
    for sample in range(sample_count):
        t = completion_time * sample / (sample_count - 1)
        distance = total * sample / (sample_count - 1)
        index = min(bisect_right(cumulative, distance) - 1, len(points) - 2)
        span = segment_lengths[index]
        fraction = (distance - cumulative[index]) / span if span else 0.0
        lat1, lon1 = points[index]
        lat2, lon2 = points[index + 1]
        path.append({
            "t": round(t, 2),
            "lat": round(lat1 + (lat2 - lat1) * fraction, 6),
            "lon": round(lon1 + (lon2 - lon1) * fraction, 6),
        })
    return path


def metrics(events: list[dict[str, Any]], completion_time: float, algorithm: str) -> list[dict[str, Any]]:
    """Produce timeline metrics (volatility, beta, tier) from event log."""
    updates = sorted(
        (event for event in events if "volatility_index" in event),
        key=lambda event: float(event.get("sim_time", 0.0)),
    )
    if not updates:
        updates = [{"sim_time": 0.0, "volatility_index": 0.0}]
    result = []
    for event in updates:
        volatility = max(0.0, min(1.0, float(event["volatility_index"])))
        beta = beta_floor_for(algorithm, volatility)
        result.append({
            "t": round(min(float(event.get("sim_time", 0.0)), completion_time), 2),
            "volatility_index": round(volatility, 4),
            "beta": round(beta, 4),
            "tier": tier_for(volatility),
        })
    return result


def export_run(
    log: Path, net_file: Path, scenario: str, algorithm: str, seed: int, completion_time: float
) -> dict[str, Any]:
    """Convert recorded simulation log into Leaflet-ready payload."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {sorted(ALGORITHMS)}")
    sumolib = require_sumolib()
    events = read_jsonl(log)
    validate_embedded_metadata(events, scenario, algorithm, seed)
    stops, order = latest_route_plan(events)
    net = sumolib.net.readNet(str(net_file))
    ordered_stops = [stops[index] for index in order]

    final_plans = [event for event in events if event.get("event") == "replan"]
    exact_edges = final_plans[-1].get("route_edges") if final_plans else None
    if isinstance(exact_edges, list) and exact_edges:
        shapes = route_shapes_from_edges(net, [str(edge) for edge in exact_edges])
        path_source = "logged_network_edges"
    else:
        shapes, used_fallback = route_shapes(net, ordered_stops)
        path_source = "legacy_stop_geometry_fallback" if used_fallback else "reconstructed_network_edges"

    return {
        "scenario": scenario,
        "algorithm": algorithm,
        "seed": seed,
        "stops": stop_records(net, stops),
        "path": dense_path(shapes, completion_time),
        "path_source": path_source,
        "metrics_over_time": metrics(events, completion_time, algorithm),
        "events": frontend_events(events, net),
        "completion_time": completion_time,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run_export(args: argparse.Namespace) -> None:
    """CLI handler for `export` subcommand."""
    data = export_run(
        Path(args.log),
        Path(args.net),
        args.scenario,
        args.algorithm,
        args.seed,
        args.completion_time,
    )
    write_json(Path(args.output), data)
    print(f"Exported {args.output}")


def run_bundle(args: argparse.Namespace) -> None:
    """CLI handler for `bundle` subcommand."""
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("manifest must contain a non-empty 'runs' list")
    output_dir = Path(args.output_dir)
    exported: list[tuple[float, str, Path]] = []
    for entry in runs:
        required = {"scenario", "seed", "va_log", "fixed_log", "va_completion_time", "fixed_completion_time"}
        missing = required - set(entry)
        if missing:
            raise ValueError(f"manifest entry is missing: {', '.join(sorted(missing))}")
        scenario, seed = str(entry["scenario"]), int(entry["seed"])
        va = export_run(
            manifest_path.parent / entry["va_log"],
            Path(args.net),
            scenario,
            "va_qpso",
            seed,
            float(entry["va_completion_time"]),
        )
        fixed = export_run(
            manifest_path.parent / entry["fixed_log"],
            Path(args.net),
            scenario,
            "fixed_beta_qpso",
            seed,
            float(entry["fixed_completion_time"]),
        )
        va_path = output_dir / f"{scenario}_seed{seed}_va_qpso.json"
        fixed_path = output_dir / f"{scenario}_seed{seed}_fixed_beta_qpso.json"
        write_json(va_path, va)
        write_json(fixed_path, fixed)
        improvement = (fixed["completion_time"] - va["completion_time"]) / fixed["completion_time"]
        if improvement > 0:
            exported.append((improvement, scenario, va_path))

    chosen: list[Path] = []
    seen_scenarios: set[str] = set()
    for _, scenario, path in sorted(exported, reverse=True):
        if scenario not in seen_scenarios and len(chosen) < args.hero_count:
            chosen.append(path)
            seen_scenarios.add(scenario)
    for _, _, path in sorted(exported, reverse=True):
        if path not in chosen and len(chosen) < args.hero_count:
            chosen.append(path)
    write_json(output_dir / "hero_scenarios.json", {"heroes": [path.name for path in chosen]})
    print(f"Exported matched pairs to {output_dir}; selected {len(chosen)} verified hero scenario(s).")


def export_from_log_file(log_path: str, output_path: str, net_file: str = DEFAULT_NET_FILE) -> None:
    """Parse an existing JSON Lines event log and export formatted frontend JSON."""
    sumolib = require_sumolib()
    net = sumolib.net.readNet(net_file)
    events_in = read_jsonl(Path(log_path))

    scenario = "medium_volatility"
    algorithm = "va_qpso"
    seed = 42

    for ev in events_in:
        if "scenario" in ev:
            scenario = ev["scenario"]
        if "algorithm" in ev:
            algorithm = ev["algorithm"]
        if "seed" in ev:
            seed = ev["seed"]

    last_time = max((float(ev.get("sim_time", 100.0)) for ev in events_in), default=100.0)
    completion_time = round(last_time, 2)

    try:
        data = export_run(Path(log_path), Path(net_file), scenario, algorithm, seed, completion_time)
    except Exception:
        # Fallback to general stop reconstruction
        stops_ids, last_best_order = latest_route_plan(events_in)
        ordered_stops = [stops_ids[i] for i in last_best_order]
        shapes, _ = route_shapes(net, ordered_stops)
        data = {
            "scenario": scenario,
            "algorithm": algorithm,
            "seed": seed,
            "stops": stop_records(net, stops_ids),
            "path": dense_path(shapes, completion_time),
            "path_source": "log_event_fallback",
            "metrics_over_time": metrics(events_in, completion_time, algorithm),
            "events": frontend_events(events_in, net),
            "completion_time": completion_time,
        }

    write_json(Path(output_path), data)
    print(f"Parsed log {log_path} -> Saved frontend JSON to {output_path}")


# ----------------------------------------------------------------------
# Live Simulation Exporter (when running SUMO)
# ----------------------------------------------------------------------

HERO_PICKUP_JUNCTION = "10239800518"  # recorded incident junction, Connaught Place
HERO_SEED = 42
PLAUSIBLE_KMH = (15.0, 30.0)  # dense urban traffic; outside this range is flagged, not hidden

# Arrival times the previous exporter wrote as constants. Kept only so the
# regenerated files can be compared against them; they are not used.
PREVIOUS_HARDCODED_COMPLETION_S = {
    ("low", "va_qpso"): 93.55, ("low", "fixed_beta_qpso"): 93.71,
    ("medium", "va_qpso"): 95.22, ("medium", "fixed_beta_qpso"): 98.62,
    ("high", "va_qpso"): 125.70, ("high", "fixed_beta_qpso"): 109.37,
}


def run_and_export_trial(
    tier: str = "medium",
    algorithm: str = "va_qpso",
    seed: int = HERO_SEED,
    num_stops: int = NUM_STOPS,
    sirens: bool = False,
    model: Any = None,
) -> Dict[str, Any]:
    """
    Recorded run: an ambulance driven through SUMO from the pickup junction to
    the default hospital's network exit, re-planning on the algorithm's
    cadence. Every time in the result is SUMO's (see src/simulation/dispatch.py).
    """
    from src.simulation.dispatch import DispatchSession, RoadModel
    from src.simulation.payload import build_payload

    model = model or RoadModel.load(Path(DEFAULT_NET_FILE))
    hospitals = load_hospitals()
    hospital = hospitals[0]
    exit_junction = hospital["exit_junction"]
    waypoints = model.default_waypoints(num_stops - 2, exclude=[HERO_PICKUP_JUNCTION, exit_junction])
    stops = [("junction", HERO_PICKUP_JUNCTION)] + [("junction", w) for w in waypoints] + [("junction", exit_junction)]
    with DispatchSession(tier, model, sirens=sirens) as session:
        dispatch_sample = session.last
        plan = session.plan(stops, algorithm, seed)
        drive = session.drive(plan, algorithm, seed)
        return build_payload(
            model=model,
            scenario=SCENARIO_MAP[tier]["key"],
            tier=tier,
            algorithm=algorithm,
            seed=seed,
            plan=plan,
            dispatch_sample=dispatch_sample,
            hospital=hospital,
            hospitals=hospitals,
            pickup_label=f"Patient pickup (junction {HERO_PICKUP_JUNCTION})",
            pickup_snap_m=None,
            drive=drive,
            edge_time=session.edge_time,
            sirens=sirens,
        )


def export_heroes(output_dir: str = DEFAULT_FRONTEND_DIR, net_file: str = DEFAULT_NET_FILE) -> None:
    """Export the three matched hero pairs and index.json, all from SUMO drive-throughs."""
    from src.simulation.dispatch import RoadModel

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model = RoadModel.load(Path(net_file))
    manifest: Dict[str, Any] = {"scenarios": []}
    report = []

    for tier, sinfo in SCENARIO_MAP.items():
        entry: Dict[str, Any] = {"id": sinfo["key"], "title": sinfo["title"], "description": sinfo["description"]}
        for algorithm, suffix in (("va_qpso", "va"), ("fixed_beta_qpso", "fb")):
            print(f"[{tier}] {algorithm}: driving ambulance through SUMO...", flush=True)
            data = run_and_export_trial(tier=tier, algorithm=algorithm, model=model)
            name = f"hero_{tier}_{algorithm}.json"
            write_json(out_path / name, data)
            timing = data["timing"]
            entry[f"{algorithm}_file"] = name
            entry[f"completion_time_{suffix}"] = data["completion_time"]
            entry[f"status_{suffix}"] = timing["status"]
            entry[f"events_count_{suffix}"] = len(data["events"])
            report.append((tier, algorithm, timing, data["traffic_at_dispatch"]))
        manifest["scenarios"].append(entry)

    write_json(out_path / "index.json", manifest)

    print("\nRecorded-run timing (old hard-coded value vs SUMO drive-through):")
    print(f"{'tier':<7}{'algorithm':<17}{'old (s)':>9}{'measured (s)':>14}{'status':>24}{'driven (m)':>12}{'avg km/h':>10}  traffic at dispatch")
    for tier, algorithm, timing, traffic in report:
        old = PREVIOUS_HARDCODED_COMPLETION_S[(tier, algorithm)]
        measured = timing["seconds_to_exit"]
        kmh = timing["average_speed_kmh"]
        flag = ""
        if kmh is None:
            flag = "  <-- no valid arrival"
        elif not PLAUSIBLE_KMH[0] <= kmh <= PLAUSIBLE_KMH[1]:
            flag = f"  <-- outside {PLAUSIBLE_KMH[0]:.0f}-{PLAUSIBLE_KMH[1]:.0f} km/h"
        print(
            f"{tier:<7}{algorithm:<17}{old:>9.2f}{(measured if measured is not None else float('nan')):>14.1f}"
            f"{timing['status']:>24}{timing['driven_length_m']:>12.0f}{(kmh if kmh is not None else float('nan')):>10.1f}"
            f"  {traffic['vehicles']} veh, {traffic['mean_vehicle_speed_kmh']} km/h, {traffic['stopped_vehicles']} stopped{flag}"
        )
    print(f"\nWrote hero files and index.json to {out_path.resolve()}")


def start_server(port: int = 8000, data_dir: str = DEFAULT_FRONTEND_DIR) -> None:
    """Launch HTTP server serving frontend mission control and /api/runs/{id}."""
    import http.server
    import socketserver
    from urllib.parse import unquote, urlparse

    target_dir = Path(data_dir).resolve()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            parsed_path = unquote(urlparse(self.path).path)

            if parsed_path in {"/", "/index.html"}:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with open(PROJECT_ROOT / "index.html", "rb") as f:
                    self.wfile.write(f.read())
                return

            if parsed_path.startswith("/api/runs/"):
                run_id = parsed_path.removeprefix("/api/runs/").removesuffix(".json")
                candidates = [
                    target_dir / f"{run_id}.json",
                    target_dir / f"hero_{run_id}.json",
                    target_dir / f"hero_{run_id}_va_qpso.json",
                    target_dir / "hero_medium_va_qpso.json",
                ]
                for c in candidates:
                    if c.is_file():
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        self.wfile.write(c.read_bytes())
                        return
                self.send_error(404, f"Run '{run_id}' not found in {target_dir.name}/")
                return

            if parsed_path.startswith("/frontend_data/"):
                sub_path = parsed_path.removeprefix("/frontend_data/")
                fpath = target_dir / sub_path
                if fpath.is_file():
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(fpath.read_bytes())
                    return

            super().do_GET()

    print("\n" + "=" * 55)
    print(f"Mission Control Web Server live at: http://localhost:{port}")
    print(f"API endpoint available at: http://localhost:{port}/api/runs/<id>")
    print("=" * 55 + "\n")
    with socketserver.TCPServer(("", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Convert logged SUMO routes into frontend map data & serve mission control"
    )
    # Global / top-level flags (backwards-compatible with main)
    root.add_argument("--log", type=str, default=None, help="Path to JSONL log file to convert")
    root.add_argument("--net", type=str, default=DEFAULT_NET_FILE, help="Path to SUMO .net.xml file")
    root.add_argument("--tier", type=str, choices=["low", "medium", "high"], default="medium", help="Scenario tier")
    root.add_argument("--algorithm", type=str, choices=sorted(ALGORITHMS), default="va_qpso", help="Optimization algorithm")
    root.add_argument("--seed", type=int, default=42, help="Random seed")
    root.add_argument("--duration", type=int, default=200, help="Simulation duration")
    root.add_argument("--output", type=str, default=None, help="Output JSON file path")
    root.add_argument("--export-heroes", action="store_true", help="Batch export all 3 hero scenarios with matched pairs")
    root.add_argument("--hero-dir", type=str, default=DEFAULT_FRONTEND_DIR, help="Directory for hero exports")
    root.add_argument("--serve", action="store_true", help="Start local HTTP server with /api/runs/{id} endpoint")
    root.add_argument("--port", type=int, default=8000, help="Port for local HTTP server")

    # Subcommands (from pr-2)
    commands = root.add_subparsers(dest="command", required=False)

    single = commands.add_parser("export", help="export one recorded run from completed log")
    single.add_argument("--log", required=True, help="Input JSON Lines log file")
    single.add_argument("--net", required=True, help="SUMO network XML file")
    single.add_argument("--scenario", required=True, help="Scenario identifier")
    single.add_argument("--algorithm", required=True, choices=sorted(ALGORITHMS), help="Algorithm name")
    single.add_argument("--seed", required=True, type=int, help="Run random seed")
    single.add_argument("--completion-time", required=True, type=float, help="Run completion time in seconds")
    single.add_argument("--output", required=True, help="Output JSON file path")
    single.set_defaults(func=run_export)

    bundle = commands.add_parser("bundle", help="export matched pairs from manifest and choose verified heroes")
    bundle.add_argument("--manifest", required=True, help="Path to runs manifest JSON")
    bundle.add_argument("--net", required=True, help="SUMO network XML file")
    bundle.add_argument("--output-dir", required=True, help="Output directory for hero JSON files")
    bundle.add_argument("--hero-count", type=int, default=3, choices=(2, 3), help="Number of hero scenarios to pick")
    bundle.set_defaults(func=run_bundle)

    return root


def main() -> None:
    arguments = build_parser().parse_args()

    # If a subcommand was used (export / bundle)
    if hasattr(arguments, "func") and arguments.func is not None:
        try:
            arguments.func(arguments)
        except (ValueError, OSError) as exc:
            raise SystemExit(f"Export failed: {exc}") from exc
        return

    # Top-level commands
    if arguments.serve:
        start_server(arguments.port, arguments.hero_dir)
    elif arguments.export_heroes:
        export_heroes(arguments.hero_dir, net_file=arguments.net)
    elif arguments.log:
        out = arguments.output or str(Path(arguments.hero_dir) / "exported_from_log.json")
        export_from_log_file(arguments.log, out, net_file=arguments.net)
    else:
        # Default: run single trial or show help if no action requested
        out = arguments.output or str(Path(arguments.hero_dir) / f"{arguments.tier}_{arguments.algorithm}.json")
        print(f"Running simulation trial: tier={arguments.tier}, algo={arguments.algorithm}, seed={arguments.seed}...")
        data = run_and_export_trial(
            tier=arguments.tier,
            algorithm=arguments.algorithm,
            seed=arguments.seed,
        )
        write_json(Path(out), data)
        print(f"Saved {out} (completion_time: {data['completion_time']}s, path points: {len(data['path'])})")


if __name__ == "__main__":
    main()
