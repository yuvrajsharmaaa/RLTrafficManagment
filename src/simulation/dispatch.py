"""
Ambulance dispatch on the SUMO Connaught Place scenarios.

One code path for recorded (hero) runs and live requests, so both report the
same kind of numbers:

1. Warm-up. Each scenario is simulated from t = 0 with SUMO seed 42 and its
   state saved at STATE_SAVE_S. The scenarios generate demand until t = 300
   and their scripted incidents are over by t = 220 (medium: closure at 120;
   high: closures at 90, surge 100-220), so dispatch happens at
   DISPATCH_S = 240: after every incident, while traffic is still arriving.
   Saved states are cached under outputs/sumo_states/ and keyed by a hash of
   the network, scenario and SUMO options, so a changed input never reuses a
   stale state. Anyone can regenerate them; nothing is committed.
2. Snapshot. The state is loaded and simulated for NVI_WINDOW steps, which
   fills the volatility index's rolling window, so V at dispatch is measured
   the same way the recorded runs always measured it.
3. Plan. VA-QPSO (or the fixed schedule) orders the waypoints between the
   pickup and the network exit (fixed_endpoints=True) on a travel-time
   matrix built from the edge speeds measured at dispatch, routed over SUMO
   lane connections for an emergency vehicle.
4. Drive (optional). An ambulance vehicle is inserted and driven along the
   plan until it reaches the exit, is teleported by SUMO, or DRIVE_CAP_S
   passes. The arrival time is whatever SUMO reports; it is never filled in.

Traffic statistics per step come from per-vehicle speed subscriptions:
vehicle count, mean vehicle speed and stopped vehicles (speed < 0.1 m/s),
over vehicles on the road network (a vehicle SUMO is teleporting has no
valid speed and is left out).
They are not derived from edge mean speeds: SUMO reports an empty edge's
mean speed as its speed limit, and an occupied edge's "last step mean speed"
also averages vehicles that left the edge during the step. Edge speeds are
used only where SUMO means them: the volatility index (as in the recorded
runs) and the planner's per-edge travel times.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.simulation.coverage import DEFAULT_NET_FILE, PROJECT_ROOT, drivable_core_edges, haversine_m

SCENARIO_DIR = PROJECT_ROOT / "networks" / "delhi" / "scenarios"
STATE_DIR = PROJECT_ROOT / "outputs" / "sumo_states"
TIERS = ("low", "medium", "high")

SUMO_SEED = 42
DISPATCH_S = 240
NVI_WINDOW = 15
STATE_SAVE_S = DISPATCH_S - NVI_WINDOW
DRIVE_CAP_S = 900
REFERENCE_VARIANCE = 0.002
AMBULANCE_ID = "ambulance_0"
AMBULANCE_TYPE = "ambulance"
STOPPED_SPEED = 0.1  # m/s, SUMO's halting threshold
MIN_EDGE_SPEED = 0.1  # m/s, floor so a jammed edge has a finite travel time

# Replan cadence, as in export_for_frontend.py: VA-QPSO replans every
# N_MAX - (N_MAX - N_MIN) * V seconds, the fixed schedule every N_FIXED.
N_MAX, N_MIN, N_FIXED = 120.0, 20.0, 70.0

_state_lock = threading.Lock()


def replan_interval(algorithm: str, volatility_index: float) -> float:
    if algorithm == "va_qpso":
        return N_MAX - (N_MAX - N_MIN) * volatility_index
    return N_FIXED


# ---------------------------------------------------------------------------
# SUMO process setup and warm-up state
# ---------------------------------------------------------------------------

def scenario_cfg(tier: str) -> Path:
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    return SCENARIO_DIR / tier / "scenario.sumocfg"


def sumo_command(tier: str, sirens: bool = False) -> List[str]:
    import sumolib

    cmd = [
        sumolib.checkBinary("sumo"),
        "-c", str(scenario_cfg(tier)),
        "--seed", str(SUMO_SEED),
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--ignore-route-errors", "true",
    ]
    if sirens:
        cmd += ["--device.bluelight.explicit", AMBULANCE_ID]
    return cmd


# Saving the RNG state makes a loaded run continue exactly like an unbroken one.
WARMUP_OPTIONS = ["--save-state.rng", "true"]


def _inputs_digest(tier: str, sirens: bool) -> str:
    import sumolib

    h = hashlib.sha256()
    tier_dir = scenario_cfg(tier).parent
    for path in sorted([DEFAULT_NET_FILE, *tier_dir.glob("*.xml"), scenario_cfg(tier)]):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    h.update(" ".join(sumo_command(tier, sirens)[1:] + WARMUP_OPTIONS).encode())
    h.update(str(sumolib.checkBinary("sumo")).encode())
    return h.hexdigest()[:12]


def warmup_state(tier: str, sirens: bool = False) -> Path:
    """Path of the saved state at STATE_SAVE_S, simulating it first if missing."""
    import traci

    path = STATE_DIR / f"{tier}_seed{SUMO_SEED}_t{STATE_SAVE_S}_{_inputs_digest(tier, sirens)}.xml.gz"
    with _state_lock:
        if path.is_file():
            return path
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        label = f"warmup_{tier}_{threading.get_ident()}"
        traci.start(sumo_command(tier, sirens) + WARMUP_OPTIONS, label=label)
        conn = traci.getConnection(label)
        try:
            while conn.simulation.getTime() < STATE_SAVE_S:
                conn.simulationStep()
            tmp = path.with_suffix(".tmp.gz")
            conn.simulation.saveState(str(tmp))
        finally:
            conn.close()
        tmp.replace(path)
        return path


# ---------------------------------------------------------------------------
# Routing on SUMO lane connections
# ---------------------------------------------------------------------------

@dataclass
class RoadModel:
    """Drivable emergency-vehicle core of the network plus coordinates."""

    net: Any
    core: Set[Any]
    junctions: Dict[str, Any]

    @classmethod
    def load(cls, net_file: Path = DEFAULT_NET_FILE) -> "RoadModel":
        import sumolib

        net = sumolib.net.readNet(str(net_file))
        core = drivable_core_edges(net)
        junctions = {e.getToNode().getID(): e.getToNode() for e in core}
        return cls(net=net, core=core, junctions=junctions)

    def lat_lon(self, xy: Tuple[float, float]) -> Tuple[float, float]:
        lon, lat = self.net.convertXY2LonLat(*xy)
        return float(lat), float(lon)

    def junction_lat_lon(self, junction_id: str) -> Tuple[float, float]:
        return self.lat_lon(self.junctions[junction_id].getCoord())

    def nearest_junction(self, lat: float, lon: float) -> Tuple[str, float]:
        """Nearest drivable-core junction to a point and its distance in meters."""
        jid = min(self.junctions, key=lambda j: haversine_m(lat, lon, *self.junction_lat_lon(j)))
        return jid, haversine_m(lat, lon, *self.junction_lat_lon(jid))

    def default_waypoints(self, count: int, exclude: Sequence[str]) -> List[str]:
        """
        Planner waypoints: the first `count` drivable junctions by ID. They are
        not real places; they exist so the planner has a visit order to
        optimize, and are labelled as auto-selected waypoints everywhere.
        """
        return [j for j in sorted(self.junctions) if j not in exclude][: max(0, count)]

    def shortest_to_junction(
        self,
        start: Tuple[str, str],
        target: str,
        edge_time: Callable[[Any], float],
    ) -> Tuple[List[str], float, float]:
        """
        Fastest core path to `target` junction. `start` is ("junction", id) to
        depart from a junction, or ("edge", id) when already on that edge (the
        edge itself is the first route element and costs nothing).
        Returns (edge ids, travel time s, length m).
        """
        kind, sid = start
        counter = itertools.count()
        # (cost, tiebreak, edge, predecessor); predecessor is fixed when settled.
        heap: List[Tuple[float, int, Any, Optional[Any]]] = []
        if kind == "edge":
            e0 = self.net.getEdge(sid)
            if e0.getToNode().getID() == target:
                return [sid], 0.0, 0.0
            heap.append((0.0, next(counter), e0, None))
        else:
            for e in self.junctions_out(sid):
                heapq.heappush(heap, (edge_time(e), next(counter), e, None))
        parent: Dict[Any, Optional[Any]] = {}
        while heap:
            cost, _, e, pred = heapq.heappop(heap)
            if e in parent:
                continue
            parent[e] = pred
            if e.getToNode().getID() == target and not (kind == "edge" and pred is None):
                path = []
                node: Optional[Any] = e
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                length = sum(p.getLength() for p in (path[1:] if kind == "edge" else path))
                return [p.getID() for p in path], cost, length
            for nxt in e.getOutgoing():
                if nxt in self.core and nxt not in parent:
                    heapq.heappush(heap, (cost + edge_time(nxt), next(counter), nxt, e))
        raise ValueError(f"No drivable route from {start} to junction {target}")

    def junctions_out(self, junction_id: str) -> List[Any]:
        return [e for e in self.junctions[junction_id].getOutgoing() if e in self.core]


def travel_matrices(
    model: RoadModel,
    stops: Sequence[Tuple[str, str]],
    edge_time: Callable[[Any], float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    (n, n) travel-time (s) and length (m) matrices between stops, for a
    fixed-endpoints plan: stops[0] is only ever an origin (it may be the edge
    the ambulance is on) and stops[-1] only a destination, so column 0 and
    row n-1 are never read and are left at 0.
    """
    n = len(stops)
    tm = np.zeros((n, n))
    dm = np.zeros((n, n))
    for i, a in enumerate(stops):
        for j, b in enumerate(stops):
            if i == j or j == 0 or i == n - 1:
                continue
            _, t, d = model.shortest_to_junction(a, b[1], edge_time)
            tm[i, j], dm[i, j] = t, d
    return tm, dm


def build_route(
    model: RoadModel,
    ordered: Sequence[Tuple[str, str]],
    edge_time: Callable[[Any], float],
) -> Tuple[List[str], float, float]:
    """Concatenate the fastest legs through `ordered` stops into one route."""
    route: List[str] = []
    total_t = total_d = 0.0
    current = ordered[0]
    for stop in ordered[1:]:
        edges, t, d = model.shortest_to_junction(current, stop[1], edge_time)
        route.extend(edges[1:] if route and edges and edges[0] == route[-1] else edges)
        total_t += t
        total_d += d
        current = ("edge", edges[-1])
    return route, total_t, total_d


# ---------------------------------------------------------------------------
# Live connection: snapshot, plan, drive
# ---------------------------------------------------------------------------

@dataclass
class TrafficSample:
    t: float  # seconds since dispatch
    volatility_index: float
    vehicles: int
    mean_vehicle_speed_kmh: Optional[float]
    stopped_vehicles: int


@dataclass
class DriveResult:
    status: str  # "arrived" | "not_arrived_within_cap" | "teleported"
    elapsed_s: float
    route_length_m: float
    driven_length_m: float
    trajectory: List[Dict[str, float]]
    samples: List[TrafficSample]
    events: List[Dict[str, Any]]
    waypoint_times: Dict[str, float] = field(default_factory=dict)

    @property
    def arrived(self) -> bool:
        return self.status == "arrived"

    @property
    def average_speed_kmh(self) -> Optional[float]:
        if not self.arrived or self.elapsed_s <= 0:
            return None
        return 3.6 * self.driven_length_m / self.elapsed_s


class DispatchSession:
    """A SUMO connection loaded at the warm-up state for one request/run."""

    def __init__(self, tier: str, model: RoadModel, sirens: bool = False) -> None:
        self.tier = tier
        self.model = model
        self.sirens = sirens
        self.conn: Any = None
        self._normal_edges: List[str] = []
        self._history: List[float] = []

    # -- lifecycle --------------------------------------------------------
    def __enter__(self) -> "DispatchSession":
        import traci
        import traci.constants as tc

        state = warmup_state(self.tier, self.sirens)
        label = f"dispatch_{self.tier}_{threading.get_ident()}_{id(self)}"
        traci.start(
            sumo_command(self.tier, self.sirens) + ["--load-state", str(state), "--begin", str(STATE_SAVE_S)],
            label=label,
        )
        self.conn = traci.getConnection(label)
        for eid in self.conn.edge.getIDList():
            self.conn.edge.subscribe(eid, [tc.LAST_STEP_MEAN_SPEED])
            if not eid.startswith(":"):
                self._normal_edges.append(eid)
        for vid in self.conn.vehicle.getIDList():  # vehicles restored from the state
            self.conn.vehicle.subscribe(vid, [tc.VAR_SPEED])
        while self.conn.simulation.getTime() < DISPATCH_S:
            self._step_sample()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    # -- measurement ------------------------------------------------------
    def _step_sample(self) -> TrafficSample:
        import traci.constants as tc

        self.conn.simulationStep()
        res = self.conn.edge.getAllSubscriptionResults()
        speeds = [res[e][tc.LAST_STEP_MEAN_SPEED] for e in self._normal_edges if e in res]
        self._history.append(float(np.mean(speeds)))
        self._history = self._history[-NVI_WINDOW:]
        if len(self._history) < 2:
            v = 0.0
        else:
            var = float(np.var(self._history))
            v = var / (var + REFERENCE_VARIANCE)
        for vid in self.conn.simulation.getDepartedIDList():
            self.conn.vehicle.subscribe(vid, [tc.VAR_SPEED])
        # A vehicle being teleported reports SUMO's INVALID_DOUBLE_VALUE (-2**30)
        # as its speed; count only vehicles on the road network.
        vehicle_speeds = [
            r[tc.VAR_SPEED]
            for r in self.conn.vehicle.getAllSubscriptionResults().values()
            if r and r[tc.VAR_SPEED] >= 0.0
        ]
        n = len(vehicle_speeds)
        halted = sum(1 for v in vehicle_speeds if v < STOPPED_SPEED)
        self.last = TrafficSample(
            t=round(self.conn.simulation.getTime() - DISPATCH_S, 2),
            volatility_index=float(min(1.0, max(0.0, v))),
            vehicles=int(n),
            mean_vehicle_speed_kmh=round(3.6 * sum(vehicle_speeds) / n, 1) if n else None,
            stopped_vehicles=int(halted),
        )
        self._speeds = {e: float(res[e][tc.LAST_STEP_MEAN_SPEED]) for e in self._normal_edges if e in res}
        return self.last

    def _remaining_on(self, road: Optional[str], pos: float) -> float:
        """Length left on the last edge seen before SUMO removed the vehicle."""
        if not road or road.startswith(":"):
            return 0.0
        return max(0.0, self.model.net.getEdge(road).getLength() - pos)

    def edge_time(self, edge: Any) -> float:
        """Travel time over an edge at the speed SUMO measured this step."""
        return edge.getLength() / max(MIN_EDGE_SPEED, self._speeds.get(edge.getID(), edge.getSpeed()))

    # -- planning ---------------------------------------------------------
    def plan(
        self,
        stops: Sequence[Tuple[str, str]],
        algorithm: str,
        seed: int,
        num_particles: int = 15,
        max_iterations: int = 30,
    ) -> Dict[str, Any]:
        """Order the middle stops; route; planner estimate from measured speeds."""
        from src.planner.qpso import replan

        tm, dm = travel_matrices(self.model, stops, self.edge_time)
        v = self.last.volatility_index
        order, score, history = replan(
            stops=[s[1] for s in stops],
            distance_matrix=tm,
            congestion_lookup={},
            volatility_index=v if algorithm == "va_qpso" else 0.0,
            weights=(1.0, 0.0, 0.0),
            num_particles=num_particles,
            max_iterations=max_iterations,
            algorithm=algorithm,
            seed=seed,
            return_history=True,
            fixed_endpoints=True,
        )
        ordered = [stops[i] for i in order]
        route, est_t, length = build_route(self.model, ordered, self.edge_time)
        return {
            "order": [int(i) for i in order],
            "ordered_stops": ordered,
            "route_edges": route,
            "estimate_s": est_t,
            "route_length_m": length,
            "best_score_history": [float(x) for x in history],
        }

    # -- driving ----------------------------------------------------------
    def drive(
        self,
        plan: Dict[str, Any],
        algorithm: str,
        seed: int,
        replans: bool = True,
        cap_s: int = DRIVE_CAP_S,
    ) -> DriveResult:
        conn = self.conn
        route = list(plan["route_edges"])
        conn.route.add("ambulance_route", route)
        conn.vehicle.add(AMBULANCE_ID, "ambulance_route", typeID=AMBULANCE_TYPE, depart="now", arrivalPos="max")
        pending = [s[1] for s in plan["ordered_stops"][1:]]  # waypoints then exit
        exit_junction = pending[-1]
        trajectory: List[Dict[str, float]] = []
        samples: List[TrafficSample] = []
        events: List[Dict[str, Any]] = []
        waypoint_times: Dict[str, float] = {}
        driven = 0.0
        last_road: Optional[str] = None
        last_pos = 0.0
        next_replan = replan_interval(algorithm, self.last.volatility_index) if replans else math.inf
        status = "not_arrived_within_cap"

        for _ in range(cap_s):
            sample = self._step_sample()
            samples.append(sample)
            t = sample.t
            if AMBULANCE_ID in conn.simulation.getStartingTeleportIDList():
                status = "teleported"
                events.append({"t": t, "type": "teleported", "detail": "SUMO teleported the ambulance out of a jam, so this run has no valid arrival time."})
                break
            if AMBULANCE_ID in conn.simulation.getArrivedIDList():
                status = "arrived"
                driven += self._remaining_on(last_road, last_pos)
                waypoint_times[exit_junction] = t
                events.append({"t": t, "type": "arrival", "detail": "Reached the network exit."})
                break
            if AMBULANCE_ID not in conn.vehicle.getIDList():
                continue  # still waiting to enter the network
            road = conn.vehicle.getRoadID(AMBULANCE_ID)
            last_pos = conn.vehicle.getLanePosition(AMBULANCE_ID)
            driven = conn.vehicle.getDistance(AMBULANCE_ID)
            lat, lon = self.model.lat_lon(conn.vehicle.getPosition(AMBULANCE_ID))
            trajectory.append({"t": t, "lat": round(lat, 6), "lon": round(lon, 6)})
            if road != last_road:
                if last_road and not last_road.startswith(":"):
                    to_node = self.model.net.getEdge(last_road).getToNode().getID()
                    if len(pending) > 1 and to_node == pending[0]:
                        waypoint_times[pending.pop(0)] = t
                        events.append({"t": t, "type": "waypoint", "junction": to_node, "detail": f"Passed planner waypoint {to_node}."})
                last_road = road

            if t >= next_replan and not road.startswith(":") and len(pending) > 1:
                sub_stops = [("edge", road)] + [("junction", j) for j in pending]
                new_plan = self.plan(sub_stops, algorithm, seed + int(t))
                old_rest = pending[:]
                pending = [s[1] for s in new_plan["ordered_stops"][1:]]
                conn.vehicle.setRoute(AMBULANCE_ID, new_plan["route_edges"])
                changed = pending != old_rest
                events.append({
                    "t": t,
                    "type": "replan",
                    "volatility_index": round(sample.volatility_index, 4),
                    "order_changed": changed,
                    "planner_estimate_remaining_s": round(new_plan["estimate_s"], 1),
                    "detail": ("Re-planned: waypoint order changed." if changed else "Re-planned: kept the same waypoint order.")
                    + f" Traffic unpredictability {sample.volatility_index:.2f}.",
                })
                next_replan = t + replan_interval(algorithm, sample.volatility_index)

        elapsed = samples[-1].t if samples else 0.0
        if status == "not_arrived_within_cap":
            events.append({"t": elapsed, "type": "not_arrived", "detail": f"Did not reach the network exit within {cap_s} s of simulated time."})
        return DriveResult(
            status=status,
            elapsed_s=elapsed,
            route_length_m=plan["route_length_m"],
            driven_length_m=round(driven, 1),
            trajectory=trajectory,
            samples=samples,
            events=events,
            waypoint_times=waypoint_times,
        )
