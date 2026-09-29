#!/usr/bin/env python3
"""
Rebuild the Delhi Connaught Place SUMO road network from raw OSM data.

Changes:
- Sets --tls.default-type to "static" (Indian traffic lights are fixed-timer, not actuated).
"""

import os
import sys
import subprocess
import shutil

# Locate SUMO_HOME
if 'SUMO_HOME' not in os.environ:
    possible_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
        "/usr/share/sumo",
        "/opt/sumo",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['SUMO_HOME'] = path
            break

def get_netconvert_binary() -> str:
    """Find netconvert executable."""
    if 'SUMO_HOME' in os.environ:
        binary_name = "netconvert.exe" if sys.platform == "win32" else "netconvert"
        candidate = os.path.join(os.environ['SUMO_HOME'], 'bin', binary_name)
        if os.path.exists(candidate):
            return candidate

    found = shutil.which("netconvert")
    if found:
        return found

    raise FileNotFoundError("netconvert executable not found. Please install SUMO or set SUMO_HOME.")

def build_network():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    osm_file = os.path.join(script_dir, "delhi_intersection.osm")
    net_file = os.path.join(script_dir, "delhi_intersection.net.xml")

    if not os.path.exists(osm_file):
        raise FileNotFoundError(f"OSM input file not found: {osm_file}")

    netconvert_bin = get_netconvert_binary()
    print(f"Using netconvert at: {netconvert_bin}")
    print(f"Building network from: {osm_file}")
    print(f"Outputting to: {net_file}")
    print("Setting --tls.default-type to 'static' (Fixed-timer Indian traffic signals)...")

    cmd = [
        netconvert_bin,
        "--osm-files", osm_file,
        # Delhi speed limits override netconvert's generic OSM defaults
        # (see delhi_speed_limits.typ.xml for the source).
        "--type-files", ",".join([
            os.path.join(os.environ.get("SUMO_HOME", ""), "data", "typemap", "osmNetconvert.typ.xml"),
            os.path.join(script_dir, "delhi_speed_limits.typ.xml"),
        ]),
        "--output-file", net_file,
        "--output.street-names", "true",
        "--proj.utm", "true",
        "--geometry.remove", "true",
        "--roundabouts.guess", "true",
        "--ramps.guess", "true",
        "--edges.join", "true",
        "--tls.discard-simple", "true",
        "--tls.guess", "true",
        "--tls.join", "true",
        "--tls.guess-signals", "true",
        "--tls.default-type", "static",  # Fixed-timer traffic lights per project spec
        "--remove-edges.isolated", "true",
        "--no-turnarounds", "true",
        "--junctions.join", "true",
        "--junctions.corner-detail", "5",
        "--osm.sidewalks", "false",
        "--no-warnings", "true",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR building network:")
        print(result.stderr)
        sys.exit(result.returncode)

    print("Success! Network rebuilt successfully.")
    print(result.stdout)

if __name__ == "__main__":
    build_network()
