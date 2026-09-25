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
    assert data["version"] == "1.0.0"


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


def test_plan_route_live_sumo():
    t0 = time.time()
    payload = {
        "incident_lat": 28.6325,
        "incident_lon": 77.2215,
        "hospital_name": "Dr. Ram Manohar Lohia Hospital",
        "scenario_tier": "medium",
        "seed": 42,
        "use_live_sumo": True,
        "num_stops": 8,
    }
    response = client.post("/api/plan-route", json=payload)
    elapsed = time.time() - t0
    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()

    assert data["algorithm"] == "va_qpso"
    assert data["scenario"] == "live_medium"
    assert len(data["stops"]) == 8
    assert len(data["path"]) > 10
    assert len(data["events"]) >= 3
    assert 0.0 <= data["volatility_index"] <= 1.0
    assert 0.5 <= data["beta"] <= 1.0
    assert data["eta_seconds"] > 0
    assert data["completion_time"] == data["eta_seconds"]
    assert data["source"] in {"live_bounded_sumo", "live_calibrated_scenario"}
    print(f"\n[Test] Live SUMO route planning completed in {elapsed:.3f}s (source: {data['source']})")
    assert elapsed < 5.0, f"Live request took too long ({elapsed:.2f}s > 5.0s)"


def test_plan_route_scenario_fallback():
    t0 = time.time()
    payload = {
        "incident_lat": 28.6300,
        "incident_lon": 77.2200,
        "hospital_name": None,
        "scenario_tier": "high",
        "seed": 101,
        "use_live_sumo": False,
        "num_stops": 6,
    }
    response = client.post("/api/plan-route", json=payload)
    elapsed = time.time() - t0
    assert response.status_code == 200
    data = response.json()

    assert data["algorithm"] == "va_qpso"
    assert data["source"] == "live_calibrated_scenario"
    assert len(data["stops"]) == 6
    assert data["selected_hospital"] is not None
    assert data["eta_seconds"] > 0
    print(f"\n[Test] Fallback scenario planning completed in {elapsed:.3f}s")
    assert elapsed < 1.0, f"Fallback planning should be sub-second ({elapsed:.2f}s)"
