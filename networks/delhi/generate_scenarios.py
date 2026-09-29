#!/usr/bin/env python3
"""
Generate reproducible Low, Medium, and High Volatility traffic scenarios
for the Delhi Connaught Place network.

Features:
- Three distinct volatility tiers (Low, Medium, High).
- Varied vehicle spawn rates / periods.
- Scripted incidents:
  * Low: Free-flowing, no disruptions.
  * Medium: Dynamic road closure starting partway through (t=120s) on key connector.
  * High: Heavy peak congestion, compound road closure at t=90s, and surge burst (t=100-220s).
- Explicit --seed argument for 100% reproducibility.
- Automated duarouter compilation into validated routes.rou.xml for each tier.
- Self-contained scenario.sumocfg generated per tier.
"""

import os
import sys
import argparse
import subprocess
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Locate SUMO_HOME
if "SUMO_HOME" not in os.environ:
    possible_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
        "/usr/share/sumo",
        "/opt/sumo",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            os.environ["SUMO_HOME"] = p
            break

SUMO_HOME = os.environ.get("SUMO_HOME", "")
RANDOM_TRIPS = os.path.join(SUMO_HOME, "tools", "randomTrips.py")

def get_binary(name: str) -> str:
    """Find SUMO executable."""
    exe_name = f"{name}.exe" if sys.platform == "win32" else name
    if SUMO_HOME:
        candidate = os.path.join(SUMO_HOME, "bin", exe_name)
        if os.path.exists(candidate):
            return candidate
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"Binary {name} not found. Please verify SUMO installation.")

# Pedestrian footway edges in Connaught Place network
PEDESTRIAN_EDGES = [
    "1000443874", "1000443877", "1000443878", "1001015117#0", "1010477871",
    "1119548714", "1119548715#0", "1119548715#1", "1119548716", "1119548717",
    "1119548718", "1119548719", "1119548720", "1119548721#0", "1119548721#1",
    "1119548722", "1119548723", "1119548724", "1119548725", "1119548726",
    "1119548727", "1119548733", "1119548734", "1119548735", "1119548736",
    "1119638714", "1119638715", "583501528#1", "660372880#3", "164675821#0",
    "80499772#1", "660372913", "1233744938#4", "164675822#0", "778204111#1",
    "1393736863", "611102337", "660372919#1", "1234918808#0", "582355769#0",
    "164675823#0", "1290403899", "1300354458", "1302777699#0", "80499763#2",
    "1234918817#1", "582355769#1", "1234918818#1", "1466288327#1", "164675818",
]

TIER_CONFIGS = {
    "low": {
        "description": "Off-peak light flow (~1000 veh/hr rate), no incidents, stable traffic",
        "pedestrians_per_hour": 300,
        "vehicles": [
            ("two_wheeler", "two_wheeler", 5.5, 10, "motorcycle"),
            ("car", "car", 8.5, 10, "passenger"),
            ("three_wheeler", "three_wheeler", 15.0, 10, "passenger"),
            ("e_rickshaw", "e_rickshaw", 22.0, 8, "passenger"),
            ("cycle_rickshaw", "cycle_rickshaw", 35.0, 6, "bicycle"),
            ("bus", "bus", 28.0, 5, "bus"),
            ("truck", "truck", 60.0, 5, "truck"),
            ("ambulance", "ambulance", 250.0, 10, "emergency"),
        ],
        "surge": None,
        "closures": []
    },
    "medium": {
        "description": "Moderate traffic (~2500 veh/hr rate) + mid-run lane closure at t=120s on Connaught connector",
        "pedestrians_per_hour": 1200,
        "vehicles": [
            ("two_wheeler", "two_wheeler", 2.6, 10, "motorcycle"),
            ("car", "car", 4.0, 10, "passenger"),
            ("three_wheeler", "three_wheeler", 7.5, 10, "passenger"),
            ("e_rickshaw", "e_rickshaw", 10.5, 8, "passenger"),
            ("cycle_rickshaw", "cycle_rickshaw", 16.0, 6, "bicycle"),
            ("bus", "bus", 13.0, 5, "bus"),
            ("truck", "truck", 30.0, 5, "truck"),
            ("ambulance", "ambulance", 120.0, 10, "emergency"),
        ],
        "surge": None,
        "closures": [
            {
                "id": "incident_med_closure",
                "edges": ["164675827#2", "164675827#3", "-164675827#2", "-164675827#3"],
                "begin": 120,
                "end": 3600,
                "allow": "emergency"
            }
        ]
    },
    "high": {
        "description": "Heavy peak traffic (~4500 veh/hr rate) + compound closure at t=90s + sudden demand surge (t=100-220s)",
        "pedestrians_per_hour": 2400,
        "vehicles": [
            ("two_wheeler", "two_wheeler", 1.5, 10, "motorcycle"),
            ("car", "car", 2.4, 10, "passenger"),
            ("three_wheeler", "three_wheeler", 4.5, 10, "passenger"),
            ("e_rickshaw", "e_rickshaw", 6.5, 8, "passenger"),
            ("cycle_rickshaw", "cycle_rickshaw", 10.0, 6, "bicycle"),
            ("bus", "bus", 8.0, 5, "bus"),
            ("truck", "truck", 18.0, 5, "truck"),
            ("ambulance", "ambulance", 80.0, 10, "emergency"),
        ],
        "surge": {
            "begin": 100,
            "end": 220,
            "vehicles": [
                ("surge_bike", "two_wheeler", 1.2, 12, "motorcycle"),
                ("surge_auto", "three_wheeler", 3.0, 10, "passenger"),
                ("surge_car", "car", 2.0, 10, "passenger"),
            ]
        },
        "closures": [
            {
                "id": "incident_high_closure_1",
                "edges": ["253307767#1", "253307767#2", "253307767#3", "-253307767#1", "-253307767#2", "-253307767#3"],
                "begin": 90,
                "end": 3600,
                "allow": ""
            },
            {
                "id": "incident_high_closure_2",
                "edges": ["164675827#2", "164675827#3", "-164675827#2", "-164675827#3"],
                "begin": 90,
                "end": 3600,
                "allow": "emergency"
            }
        ]
    }
}


def write_incidents_file(closures: List[Dict[str, Any]], filepath: str, duration: int):
    """Write SUMO additional file defining rerouters / road closures."""
    root = ET.Element("additional")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/additional_file.xsd")

    for c in closures:
        rerouter = ET.SubElement(root, "rerouter")
        rerouter.set("id", c["id"])
        rerouter.set("edges", " ".join(c["edges"]))

        interval = ET.SubElement(rerouter, "interval")
        interval.set("begin", str(c["begin"]))
        interval.set("end", str(max(duration, c["end"])))

        for edge in c["edges"]:
            closing = ET.SubElement(interval, "closingReroute")
            closing.set("id", edge)
            closing.set("allow", c.get("allow", ""))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(filepath, encoding="utf-8", xml_declaration=True)


def write_sumocfg(
    cfg_path: str,
    net_path: str,
    routes_path: str,
    additional_paths: List[str],
    duration: int
):
    """Write self-contained .sumocfg file for the scenario."""
    root = ET.Element("configuration")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/sumoConfiguration.xsd")

    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file").set("value", net_path)
    ET.SubElement(inp, "route-files").set("value", routes_path)
    if additional_paths:
        ET.SubElement(inp, "additional-files").set("value", ",".join(additional_paths))

    t = ET.SubElement(root, "time")
    ET.SubElement(t, "begin").set("value", "0")
    ET.SubElement(t, "end").set("value", str(duration))
    ET.SubElement(t, "step-length").set("value", "1.0")

    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "time-to-teleport").set("value", "300")
    ET.SubElement(proc, "collision.action").set("value", "warn")
    ET.SubElement(proc, "pedestrian.model").set("value", "striping")
    ET.SubElement(proc, "ignore-route-errors").set("value", "true")

    routing = ET.SubElement(root, "routing")
    ET.SubElement(routing, "device.rerouting.probability").set("value", "1.0")
    ET.SubElement(routing, "device.rerouting.adaptation-interval").set("value", "5")

    rep = ET.SubElement(root, "report")
    ET.SubElement(rep, "verbose").set("value", "false")
    ET.SubElement(rep, "no-step-log").set("value", "true")
    ET.SubElement(rep, "no-warnings").set("value", "true")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(cfg_path, encoding="utf-8", xml_declaration=True)


def build_scenario_tier(
    tier: str,
    duration: int,
    seed: int,
    net_file: str,
    vtypes_file: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Generate trips, compile with duarouter, and write config for a single tier.
    """
    cfg = TIER_CONFIGS[tier]
    tier_dir = os.path.join(output_dir, tier)
    os.makedirs(tier_dir, exist_ok=True)

    random.seed(seed)

    # 1. Incidents file
    incidents_file = os.path.join(tier_dir, "incidents.add.xml")
    write_incidents_file(cfg["closures"], incidents_file, duration)

    # 2. Generate vehicle trips using randomTrips.py
    temp_trip_files = []
    veh_seed = seed

    for prefix, vtype, period, fringe, vclass in cfg["vehicles"]:
        veh_seed += 11
        trip_file = os.path.join(tier_dir, f"temp_{prefix}.trips.xml")
        temp_trip_files.append(trip_file)

        cmd = [
            sys.executable, RANDOM_TRIPS,
            "-n", net_file,
            "--additional-file", vtypes_file,
            "-o", trip_file,
            "--prefix", f"{tier}_{prefix}",
            "--trip-attributes", f'type="{vtype}"',
            "-b", "0",
            "-e", str(duration),
            "-p", str(period),
            "--fringe-factor", str(fringe),
            "-s", str(veh_seed),
            "--allow-fringe.min-length", "50",
            "--min-distance", "80",
            "--edge-permission", vclass,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  [Warning {prefix}]: {res.stderr[:200]}")

    # 3. Add surge trips if applicable
    if cfg["surge"]:
        s_cfg = cfg["surge"]
        for prefix, vtype, period, fringe, vclass in s_cfg["vehicles"]:
            veh_seed += 19
            trip_file = os.path.join(tier_dir, f"temp_surge_{prefix}.trips.xml")
            temp_trip_files.append(trip_file)

            cmd = [
                sys.executable, RANDOM_TRIPS,
                "-n", net_file,
                "--additional-file", vtypes_file,
                "-o", trip_file,
                "--prefix", f"{tier}_{prefix}",
                "--trip-attributes", f'type="{vtype}"',
                "-b", str(s_cfg["begin"]),
                "-e", str(s_cfg["end"]),
                "-p", str(period),
                "--fringe-factor", str(fringe),
                "-s", str(veh_seed),
                "--allow-fringe.min-length", "50",
                "--min-distance", "60",
                "--edge-permission", vclass,
            ]
            subprocess.run(cmd, capture_output=True, text=True)

    # 4. Generate pedestrian trips
    peds_per_hr = cfg["pedestrians_per_hour"]
    num_peds = int(peds_per_hr * duration / 3600)
    ped_elements = []

    if num_peds > 0:
        ped_types = ["pedestrian", "pedestrian_slow", "pedestrian_fast"]
        ped_weights = [0.55, 0.20, 0.25]
        ped_step = duration / num_peds

        for i in range(num_peds):
            depart = max(0.0, min(duration - 1, i * ped_step + random.uniform(-0.5, 0.5)))
            ptype = random.choices(ped_types, weights=ped_weights, k=1)[0]
            from_e = random.choice(PEDESTRIAN_EDGES)
            to_e = random.choice(PEDESTRIAN_EDGES)
            while to_e == from_e:
                to_e = random.choice(PEDESTRIAN_EDGES)

            person = ET.Element("person")
            person.set("id", f"{tier}_ped_{i}")
            person.set("depart", f"{depart:.2f}")
            person.set("type", ptype)

            walk = ET.SubElement(person, "walk")
            walk.set("from", from_e)
            walk.set("to", to_e)
            ped_elements.append(person)

    # 5. Merge all trips & pedestrians into scenario_trips.xml
    all_trips = []
    for tf in temp_trip_files:
        if os.path.exists(tf):
            tree = ET.parse(tf)
            for t in tree.getroot().findall("trip"):
                all_trips.append(t)

    all_trips.sort(key=lambda t: float(t.get("depart", 0.0)))
    ped_elements.sort(key=lambda p: float(p.get("depart", 0.0)))

    combined_root = ET.Element("routes")
    combined_root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    combined_root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")

    v_i, p_i = 0, 0
    while v_i < len(all_trips) and p_i < len(ped_elements):
        v_dep = float(all_trips[v_i].get("depart", 0.0))
        p_dep = float(ped_elements[p_i].get("depart", 0.0))
        if v_dep <= p_dep:
            combined_root.append(all_trips[v_i])
            v_i += 1
        else:
            combined_root.append(ped_elements[p_i])
            p_i += 1

    while v_i < len(all_trips):
        combined_root.append(all_trips[v_i])
        v_i += 1
    while p_i < len(ped_elements):
        combined_root.append(ped_elements[p_i])
        p_i += 1

    trips_output = os.path.join(tier_dir, "scenario_trips.xml")
    tree = ET.ElementTree(combined_root)
    ET.indent(tree, space="    ")
    tree.write(trips_output, encoding="utf-8", xml_declaration=True)

    # Cleanup temp trip files
    for tf in temp_trip_files:
        if os.path.exists(tf):
            os.remove(tf)
        alt = tf.replace(".trips.xml", ".trips.alt.xml")
        if os.path.exists(alt):
            os.remove(alt)

    # 6. Route compilation using duarouter as dumb baseline router
    routes_output = os.path.join(tier_dir, "routes.rou.xml")
    duarouter_bin = get_binary("duarouter")
    additional_list = [vtypes_file]
    if os.path.exists(incidents_file) and os.path.getsize(incidents_file) > 100:
        additional_list.append(incidents_file)

    dua_cmd = [
        duarouter_bin,
        "--net-file", net_file,
        "--additional-files", ",".join(additional_list),
        "--route-files", trips_output,
        "--output-file", routes_output,
        "--begin", "0",
        "--end", str(duration),
        "--ignore-errors", "true",
        "--no-warnings", "true",
    ]
    dua_res = subprocess.run(dua_cmd, capture_output=True, text=True)
    if dua_res.returncode != 0:
        print(f"  [Duarouter Error in {tier}]: {dua_res.stderr[:300]}")

    # 7. Write scenario.sumocfg
    sumocfg_path = os.path.join(tier_dir, "scenario.sumocfg")
    # Only incidents file is needed in additional_files, because duarouter
    # already embeds the vType definitions directly inside routes.rou.xml.
    add_paths = []
    if os.path.exists(incidents_file) and os.path.getsize(incidents_file) > 100:
        add_paths.append(os.path.relpath(incidents_file, tier_dir))

    write_sumocfg(
        cfg_path=sumocfg_path,
        # Relative to the .sumocfg, so the scenario works from any checkout.
        net_path=os.path.relpath(net_file, tier_dir),
        routes_path=os.path.relpath(routes_output, tier_dir),
        additional_paths=add_paths,
        duration=duration
    )

    result_info = {
        "tier": tier,
        "vehicles_generated": len(all_trips),
        "pedestrians_generated": len(ped_elements),
        "trips_file": trips_output,
        "routes_file": routes_output,
        "incidents_file": incidents_file,
        "sumocfg": sumocfg_path,
        "duarouter_ok": (dua_res.returncode == 0)
    }

    print(f"  Tier [{tier.upper()}]: {len(all_trips)} vehicles, {len(ped_elements)} pedestrians compiled via duarouter -> {routes_output}")
    return result_info


def main():
    parser = argparse.ArgumentParser(description="Generate Delhi Traffic Scenarios (Low, Medium, High Volatility)")
    parser.add_argument("--tier", choices=["low", "medium", "high", "all"], default="all", help="Scenario tier")
    parser.add_argument("--duration", type=int, default=300, help="Simulation duration in seconds (default 300 = 5 min)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for exact reproducibility")
    parser.add_argument("--output-dir", type=str, default="networks/delhi/scenarios", help="Output directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    net_file = str(project_root / "networks" / "delhi" / "delhi_intersection.net.xml")
    vtypes_file = str(project_root / "networks" / "delhi" / "delhi_vtypes.add.xml")
    out_dir = str(project_root / args.output_dir)

    tiers = ["low", "medium", "high"] if args.tier == "all" else [args.tier]

    print("=" * 65)
    print("Delhi Scenario Demand Generator (QPSO Benchmarking)")
    print(f"Tiers: {tiers} | Duration: {args.duration}s | Seed: {args.seed}")
    print("=" * 65)

    for tier in tiers:
        build_scenario_tier(
            tier=tier,
            duration=args.duration,
            seed=args.seed,
            net_file=net_file,
            vtypes_file=vtypes_file,
            output_dir=out_dir
        )

    print("\nAll scenario tiers built and compiled successfully.")


if __name__ == "__main__":
    main()
