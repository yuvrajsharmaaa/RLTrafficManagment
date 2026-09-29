#!/usr/bin/env python3
"""
Reproduce the demo's numbers from scratch and check them against the repo.

    python reproduce_demo.py            # primary case: Heavy traffic, default hospital
    python reproduce_demo.py --all      # also re-drive every recorded run (slower)

Checks, in order:
1. Hospital coverage: hospitals.json matches what the network file implies,
   and the default destination's network exit and straight-line final leg.
2. Traffic at dispatch per tier (t = 240 s) next to what the previous
   5-step snapshot saw, both measured now in SUMO.
3. Live request: POST /api/plan-route twice with the same body; the two
   responses must be identical.
4. Recorded runs: re-drive the ambulance in SUMO and compare with the
   committed frontend_data/hero_*.json (must match exactly).

Exit code is non-zero if any check fails. Requires SUMO (sumolib, traci).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def check_coverage() -> bool:
    from src.simulation.coverage import HOSPITALS_FILE, annotate_hospitals, load_net

    stored = json.loads(HOSPITALS_FILE.read_text(encoding="utf-8"))
    fresh = annotate_hospitals(load_net(), stored)
    ok = fresh == stored
    print("\n1. Hospital coverage (straight-line distance from each hospital's network exit)")
    for h in fresh:
        print(f"   {h['name'][:48]:<48} in_network={h['in_network']!s:<5} exit={h['exit_junction']:<14} {h['straight_line_from_exit_m']:>8.1f} m")
    print(f"   hospitals.json matches the network: {'PASS' if ok else 'FAIL'}")
    return ok


def check_traffic() -> None:
    import traci

    from src.simulation.dispatch import TIERS, DispatchSession, RoadModel, sumo_command

    model = RoadModel.load()
    print("\n2. Traffic per tier: previous 5-step snapshot vs dispatch at t = 240 s (both measured now)")
    print(f"   {'tier':<7}{'t=5 vehicles':>13}{'t=5 km/h':>10}{'t=240 vehicles':>16}{'t=240 km/h':>12}{'stopped':>9}{'V':>7}")
    for tier in TIERS:
        traci.start(sumo_command(tier), label=f"old_{tier}")
        conn = traci.getConnection(f"old_{tier}")
        for _ in range(5):
            conn.simulationStep()
        speeds = [conn.vehicle.getSpeed(v) for v in conn.vehicle.getIDList()]
        conn.close()
        with DispatchSession(tier, model) as s:
            now = s.last
        old_kmh = 3.6 * sum(speeds) / len(speeds) if speeds else float("nan")
        print(f"   {tier:<7}{len(speeds):>13}{old_kmh:>10.1f}{now.vehicles:>16}{now.mean_vehicle_speed_kmh:>12.1f}{now.stopped_vehicles:>9}{now.volatility_index:>7.3f}")


def check_live() -> bool:
    from fastapi.testclient import TestClient

    from server import app

    client = TestClient(app)
    body = {"incident_lat": 28.632587, "incident_lon": 77.222985, "scenario_tier": "high", "seed": 42, "use_live_sumo": True, "num_stops": 8}
    a = client.post("/api/plan-route", json=body).json()
    b = client.post("/api/plan-route", json=body).json()
    same = a == b
    t, d = a["timing"], a["traffic_at_dispatch"]
    print("\n3. Live request (Heavy, seed 42, planner estimate)")
    print(f"   destination: {a['stops'][-1]['label']}; final leg {a['final_leg']['straight_line_m']:.0f} m straight-line (not simulated)")
    print(f"   traffic at dispatch: {d['vehicles']} vehicles, {d['mean_vehicle_speed_kmh']} km/h, {d['stopped_vehicles']} stopped, unpredictability {d['volatility_index']} ({d['tier']})")
    print(f"   time to exit: {t['seconds_to_exit']} s ({t['kind']}), route {t['route_length_m']:.0f} m")
    print(f"   identical on a second request: {'PASS' if same else 'FAIL'}")
    return same


def check_recorded(runs) -> bool:
    import export_for_frontend as eff
    from src.simulation.dispatch import RoadModel

    model = RoadModel.load()
    print("\n4. Recorded runs re-driven in SUMO vs committed files")
    ok_all = True
    for tier, algorithm in runs:
        fresh = eff.run_and_export_trial(tier=tier, algorithm=algorithm, model=model)
        path = ROOT / "frontend_data" / f"hero_{tier}_{algorithm}.json"
        committed = json.loads(path.read_text(encoding="utf-8"))
        same = json.loads(json.dumps(fresh)) == committed
        ok_all &= same
        t = fresh["timing"]
        result = f"{t['seconds_to_exit']:.0f} s" if t["seconds_to_exit"] is not None else f"not at exit after {t['simulated_seconds']:.0f} s"
        speed = f"{t['average_speed_kmh']} km/h" if t["average_speed_kmh"] else f"{3.6 * t['driven_length_m'] / t['simulated_seconds']:.1f} km/h so far"
        print(f"   {tier:<7}{algorithm:<17} {t['status']:<24} {result:<28} {t['driven_length_m']:>6.0f} m  {speed:<18} matches file: {'PASS' if same else 'FAIL'}")
    return ok_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="re-drive all six recorded runs, not just Heavy")
    args = parser.parse_args()
    tiers = ("low", "medium", "high") if args.all else ("high",)
    runs = [(t, a) for t in tiers for a in ("va_qpso", "fixed_beta_qpso")]

    ok = check_coverage()
    check_traffic()
    ok &= check_live()
    ok &= check_recorded(runs)
    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
