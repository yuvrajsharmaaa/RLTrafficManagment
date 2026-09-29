"""hospitals.json coverage fields must match what the network file implies."""

import json

from src.simulation.coverage import HOSPITALS_FILE, annotate_hospitals, load_net


def test_hospitals_json_matches_network():
    stored = json.loads(HOSPITALS_FILE.read_text(encoding="utf-8"))
    assert annotate_hospitals(load_net(), stored) == stored


def test_no_hospital_is_inside_current_network():
    stored = json.loads(HOSPITALS_FILE.read_text(encoding="utf-8"))
    assert stored and not any(h["in_network"] for h in stored)
    # Default demo destination: shortest unsimulated final leg, listed first.
    assert stored[0]["name"] == "Dr. Ram Manohar Lohia Hospital"
