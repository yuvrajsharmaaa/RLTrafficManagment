"""Recorded hero runs: every stored number must be a plausible SUMO measurement."""

import json
from pathlib import Path

import pytest

HERO_FILES = sorted(Path("frontend_data").glob("hero_*.json"))


@pytest.mark.parametrize("path", HERO_FILES, ids=lambda p: p.stem)
def test_hero_traffic_samples_are_valid(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    for m in data["metrics_over_time"]:
        assert 0.0 <= m["volatility_index"] <= 1.0
        assert m["vehicles"] >= m["stopped_vehicles"] >= 0
        if m["mean_vehicle_speed_kmh"] is not None:
            assert 0.0 <= m["mean_vehicle_speed_kmh"] < 150.0


@pytest.mark.parametrize("path", HERO_FILES, ids=lambda p: p.stem)
def test_hero_timing_is_consistent(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    timing = data["timing"]
    assert timing["kind"] == "simulated_drive"
    assert data["stops"][0]["kind"] == "pickup" and data["stops"][-1]["kind"] == "network_exit"
    if timing["status"] == "arrived":
        assert data["completion_time"] == timing["seconds_to_exit"]
        assert data["stops"][-1]["reached_t"] == timing["seconds_to_exit"]
    else:
        assert timing["seconds_to_exit"] is None
        assert data["completion_time"] == timing["simulated_seconds"]
    # The unsimulated final leg is reported separately, never as time.
    assert data["final_leg"]["simulated"] is False
    assert data["path"][-1]["t"] <= data["completion_time"]
