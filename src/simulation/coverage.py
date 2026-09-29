"""
Network coverage for ambulance destinations.

The SUMO network covers central Connaught Place only. Every hospital in
hospitals.json lies outside the road-covered area, so a route can only be
simulated as far as the network junction closest to the hospital (the
"exit junction"). The rest of the trip is not simulated; it is reported
separately as a straight-line distance, never as time.

Everything here is derived from the .net.xml and the hospital coordinates,
so a judge can regenerate hospitals.json with:

    python -m src.simulation.coverage --write
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NET_FILE = PROJECT_ROOT / "networks" / "delhi" / "delhi_intersection.net.xml"
HOSPITALS_FILE = PROJECT_ROOT / "hospitals.json"
VEHICLE_CLASS = "emergency"
EARTH_RADIUS_M = 6371008.8


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def drivable_core_edges(net: Any, vclass: str = VEHICLE_CLASS) -> Set[Any]:
    """
    Largest strongly connected set of edges an `vclass` vehicle can drive,
    following SUMO lane connections (turn restrictions included). Any edge
    in it can reach, and be reached from, every other.
    """
    edges = [e for e in net.getEdges() if not e.isSpecial() and e.allows(vclass)]

    def reach(start: Any, forward: bool) -> Set[Any]:
        seen, stack = {start}, [start]
        while stack:
            e = stack.pop()
            nxt = e.getOutgoing() if forward else e.getIncoming()
            for o in nxt:
                if o.allows(vclass) and o not in seen:
                    seen.add(o)
                    stack.append(o)
        return seen

    best: Set[Any] = set()
    assigned: Set[Any] = set()
    for e in edges:
        if e in assigned:
            continue
        scc = reach(e, True) & reach(e, False)
        assigned |= scc
        if len(scc) > len(best):
            best = scc
    return best


def core_junctions(core: Set[Any]) -> Dict[str, Any]:
    """Junctions reachable in the drivable core: the to-node of a core edge."""
    return {e.getToNode().getID(): e.getToNode() for e in core}


def node_lat_lon(net: Any, node: Any) -> tuple[float, float]:
    lon, lat = net.convertXY2LonLat(*node.getCoord())
    return float(lat), float(lon)


def hospital_coverage(net: Any, hospital: Dict[str, Any], core: Set[Any]) -> Dict[str, Any]:
    """Coverage facts for one hospital, all measured from the network file."""
    lat, lon = float(hospital["lat"]), float(hospital["lon"])
    x, y = net.convertLonLat2XY(lon, lat)
    xmin, ymin, xmax, ymax = net.getBoundary()
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)

    nearest_edge_m = None
    for radius in (100, 500, 2000, 20000):
        found = net.getNeighboringEdges(x, y, r=radius)
        if found:
            nearest_edge_m = min(d for _, d in found)
            break

    junctions = core_junctions(core)
    exit_id, exit_node = min(
        junctions.items(),
        key=lambda item: haversine_m(lat, lon, *node_lat_lon(net, item[1])),
    )
    exit_lat, exit_lon = node_lat_lon(net, exit_node)
    remaining_m = haversine_m(lat, lon, exit_lat, exit_lon)
    return {
        # Inside only if the hospital is within snapping distance of a drivable road.
        "in_network": remaining_m <= 100.0,
        "exit_junction": exit_id,
        "exit_lat": round(exit_lat, 6),
        "exit_lon": round(exit_lon, 6),
        "straight_line_from_exit_m": round(remaining_m, 1),
        "nearest_edge_m": round(nearest_edge_m, 1) if nearest_edge_m is not None else None,
        "outside_boundary_box_m": round(math.hypot(dx, dy), 1),
    }


def load_net(net_file: Path = DEFAULT_NET_FILE) -> Any:
    import sumolib

    return sumolib.net.readNet(str(net_file))


def annotate_hospitals(net: Any, hospitals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return hospitals with fresh coverage fields, nearest-to-network first."""
    core = drivable_core_edges(net)
    out = []
    for h in hospitals:
        rec = {k: v for k, v in h.items() if k not in _COVERAGE_KEYS}
        rec.update(hospital_coverage(net, rec, core))
        out.append(rec)
    out.sort(key=lambda r: r["straight_line_from_exit_m"])
    return out


_COVERAGE_KEYS = {
    "in_network",
    "exit_junction",
    "exit_lat",
    "exit_lon",
    "straight_line_from_exit_m",
    "nearest_edge_m",
    "outside_boundary_box_m",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--net", default=str(DEFAULT_NET_FILE))
    parser.add_argument("--write", action="store_true", help="rewrite hospitals.json with the computed fields")
    args = parser.parse_args()

    net = load_net(Path(args.net))
    hospitals = json.loads(HOSPITALS_FILE.read_text(encoding="utf-8"))
    annotated = annotate_hospitals(net, hospitals)
    for h in annotated:
        print(
            f"{h['name'][:44]:<44} in_network={h['in_network']!s:<5} exit={h['exit_junction']:<12} "
            f"straight-line from exit={h['straight_line_from_exit_m']:>8.1f} m  "
            f"nearest edge={h['nearest_edge_m']:>8.1f} m  outside box={h['outside_boundary_box_m']:>7.1f} m"
        )
    if args.write:
        HOSPITALS_FILE.write_text(json.dumps(annotated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {HOSPITALS_FILE}")


if __name__ == "__main__":
    main()
