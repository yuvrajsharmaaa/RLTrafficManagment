"""
Unit and integration tests for FastAPI server.py endpoints.
"""

import time
import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "trafficmgmt-adaptive-routing"
    assert "sumo_available" in data
    assert data["version"] == "1.1.0"


def test_get_hero_runs_default():
    response = client.get("/api/hero-runs")
    assert response.status_code == 200
    data = response.json()
    assert "algorithm" in data
    assert "stops" in data
    assert "path" in data
    assert "events" in data
    assert data["completion_time"] > 0


def test_get_hero_runs_catalog():
    response = client.get("/api/hero-runs?catalog=true")
    assert response.status_code == 200
    data = response.json()
    assert "scenarios" in data
    assert len(data["scenarios"]) > 0


def test_get_specific_hero_run():
    response = client.get("/api/hero-runs/hero_medium_va_qpso")
    assert response.status_code == 200
    data = response.json()
    assert data["algorithm"] == "va_qpso"


def test_get_legacy_run():
    response = client.get("/api/runs/hero_low_fixed_beta_qpso")
    assert response.status_code == 200
    data = response.json()
    assert data["algorithm"] == "fixed_beta_qpso"


def test_static_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Delhi Ambulance Dispatch" in response.text or "Delhi EMS Routing" in response.text


LIVE = {
    "incident_lat": 28.6325,
    "incident_lon": 77.2215,
    "scenario_tier": "medium",
    "seed": 42,
    "use_live_sumo": True,
    "num_stops": 8,
}


def test_plan_route_live_estimate():
    t0 = time.time()
    response = client.post("/api/plan-route", json=LIVE)
    elapsed = time.time() - t0
    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()

    assert data["algorithm"] == "va_qpso"
    assert data["scenario"] == "live_medium"
    assert data["source"] == "live_sumo_snapshot"
    assert len(data["stops"]) == 8
    # Route runs pickup -> planner waypoints -> network exit, in that order.
    assert [s["kind"] for s in data["stops"]] == ["pickup"] + ["waypoint"] * 6 + ["network_exit"]
    assert data["stops"][-1]["id"] == data["selected_hospital"]["exit_junction"]
    assert data["stops"][-1]["label"].startswith("Network exit toward ")
    assert data["selected_hospital"]["in_network"] is False
    assert data["final_leg"]["simulated"] is False
    assert data["final_leg"]["straight_line_m"] == data["selected_hospital"]["straight_line_from_exit_m"]
    # Estimate, labelled as one; never blended with the final leg.
    assert data["timing"]["kind"] == "planner_estimate"
    assert data["timing"]["status"] == "estimate"
    assert data["completion_time"] == data["eta_seconds"] == data["timing"]["planner_estimate_s"]
    assert data["path_source"] == "planned_route_timed_by_planner_estimate"
    assert len(data["path"]) > 10
    assert 0.5 <= data["beta"] <= 0.75
    print(f"\n[Test] Live estimate in {elapsed:.2f}s")
    assert elapsed < 10.0, f"Live estimate took too long ({elapsed:.2f}s)"


def test_plan_route_is_reproducible():
    a = client.post("/api/plan-route", json=LIVE).json()
    b = client.post("/api/plan-route", json=LIVE).json()
    assert a == b


def test_plan_route_traffic_is_measured_at_dispatch():
    data = client.post("/api/plan-route", json={**LIVE, "scenario_tier": "high"}).json()
    (sample,) = data["metrics_over_time"]
    assert sample == data["traffic_at_dispatch"]
    assert sample["t"] == 0.0
    assert sample["volatility_index"] == data["volatility_index"]
    assert sample["beta"] == data["beta"]
    assert abs(data["beta"] - (0.5 + 0.25 * data["volatility_index"])) < 1e-3
    assert sample["vehicles"] > 0 and sample["mean_vehicle_speed_kmh"] is not None
    history = data["best_score_history"]
    assert len(history) >= 1
    assert all(b <= a for a, b in zip(history, history[1:]))  # elite score never worsens


def test_plan_route_rejects_non_sumo_request():
    response = client.post("/api/plan-route", json={**LIVE, "use_live_sumo": False})
    assert response.status_code == 400
    assert "SUMO" in response.json()["detail"]


def test_plan_route_unknown_hospital_is_an_error():
    response = client.post("/api/plan-route", json={**LIVE, "hospital_name": "No Such Hospital"})
    assert response.status_code == 400


def test_exported_beta_is_anneal_floor():
    import export_for_frontend as eff

    assert eff.beta_floor_for("va_qpso", 0.0) == 0.5
    assert eff.beta_floor_for("va_qpso", 1.0) == 0.75
    assert eff.beta_floor_for("fixed_beta_qpso", 0.9) == 0.5
