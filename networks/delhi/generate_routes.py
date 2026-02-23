#!/usr/bin/env python3
"""
Generate realistic Delhi traffic routes with Indian vehicle types.

Traffic composition (based on Delhi traffic surveys):
  - Two-wheelers (motorcycles/scooters): 40%
  - Cars (4-wheelers): 30%
  - Three-wheelers (auto-rickshaws): 15%
  - Buses (DTC/Cluster): 10%
  - Ambulances (emergency): 1%
  - Remaining: mixed cars (~4%)

Simulates 1 hour (3600 seconds) of heavy metro traffic.
"""

import subprocess
import os
import sys
import xml.etree.ElementTree as ET

SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
RANDOM_TRIPS = os.path.join(SUMO_HOME, "tools", "randomTrips.py")
NET_FILE = "delhi_intersection.net.xml"
VTYPES_FILE = "delhi_vtypes.add.xml"
OUTPUT_ROUTE = "delhi_intersection.rou.xml"
DURATION = 3600  # 1 hour simulation

# Delhi-level heavy traffic: ~3000 vehicles/hour total
# period = 3600 / vehicles_per_hour_for_this_type
VEHICLE_CONFIGS = [
    # (prefix, vtype, period, fringe_factor - higher means more from boundary)
    ("two_wheeler", "two_wheeler", 3.0, 10),    # ~1200 veh/hr (40%)
    ("car", "car", 4.0, 10),                     # ~900 veh/hr (30%)
    ("three_wheeler", "three_wheeler", 8.0, 10),  # ~450 veh/hr (15%)
    ("bus", "bus", 12.0, 5),                      # ~300 veh/hr (10%)
    ("ambulance", "ambulance", 120.0, 10),        # ~30 veh/hr (1%)
]


def generate_trips():
    """Generate trips for each vehicle type."""
    trip_files = []

    for prefix, vtype, period, fringe in VEHICLE_CONFIGS:
        trip_file = f"trips_{prefix}.trips.xml"
        trip_files.append(trip_file)

        cmd = [
            sys.executable, RANDOM_TRIPS,
            "-n", NET_FILE,
            "--additional-file", VTYPES_FILE,
            "-o", trip_file,
            "--prefix", prefix,
            "--trip-attributes", f'type="{vtype}"',
            "-b", "0",
            "-e", str(DURATION),
            "-p", str(period),
            "--fringe-factor", str(fringe),
            "--validate",
            "--random",
            "--allow-fringe.min-length", "50",
            "--min-distance", "100",
        ]

        print(f"Generating {prefix} trips (period={period}s, ~{int(DURATION/period)}/hr)...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Warning: {result.stderr[:200]}")
        else:
            print(f"  Done: {trip_file}")

    return trip_files


def merge_and_sort_routes(trip_files):
    """Merge all trip files into one sorted route file."""
    all_trips = []

    for tf in trip_files:
        if not os.path.exists(tf):
            print(f"  Skipping missing file: {tf}")
            continue
        tree = ET.parse(tf)
        root = tree.getroot()
        for trip in root.findall("trip"):
            all_trips.append(trip)

    # Sort by departure time
    all_trips.sort(key=lambda t: float(t.get("depart", "0")))

    # Build combined route file
    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation",
             "http://sumo.dlr.de/xsd/routes_file.xsd")

    for trip in all_trips:
        root.append(trip)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(OUTPUT_ROUTE, encoding="unicode", xml_declaration=True)
    print(f"\nMerged {len(all_trips)} trips into {OUTPUT_ROUTE}")

    # Cleanup temp files
    for tf in trip_files:
        if os.path.exists(tf):
            os.remove(tf)
        # Remove associated .alt files
        alt = tf.replace(".trips.xml", ".trips.alt.xml")
        if os.path.exists(alt):
            os.remove(alt)

    return len(all_trips)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 60)
    print("Delhi Traffic Route Generator")
    print(f"Duration: {DURATION}s | Target: ~3000 veh/hr (heavy metro)")
    print("=" * 60)

    trip_files = generate_trips()
    total = merge_and_sort_routes(trip_files)

    print(f"\nTotal vehicles generated: {total}")
    print(f"Effective rate: {total * 3600 / DURATION:.0f} vehicles/hour")
    print("Done!")


if __name__ == "__main__":
    main()
