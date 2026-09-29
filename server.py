#!/usr/bin/env python3
"""
FastAPI Server for Adaptive Emergency Ambulance Route Dispatch.

Provides:
- POST /api/plan-route: Loads the requested traffic scenario's SUMO state (seed 42,
  t = 240 s), plans the ambulance route with VA-QPSO, and returns either the
  planner estimate or, with drive_through=true, the time an ambulance vehicle
  actually took in SUMO. No synthetic fallback: without SUMO it returns 503.
- GET /api/hero-runs: Serves bundled hero-run JSON files as the fast-loading default view.
- GET /api/hero-runs/{run_id}: Serves specific precomputed hero run JSON.
- GET /api/runs/{run_id}: Backward compatibility with the existing route replay viewer.
- GET /health: Operational health and dependency diagnostic check.
- Static assets: Serves index.html, frontend_data/, PWA manifest, and icons.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("server")

SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_-]+$")
DEFAULT_DATA_DIR = PROJECT_ROOT / "frontend_data"
DEFAULT_NET_FILE = PROJECT_ROOT / "networks" / "delhi" / "delhi_intersection.net.xml"

# Optional simulation imports
HAS_SIMULATION = False
try:
    import sumolib  # noqa: F401
    import traci  # noqa: F401

    from src.simulation.dispatch import TIERS, DispatchSession, RoadModel
    from src.simulation.payload import beta_floor, build_payload

    HAS_SIMULATION = True
except Exception as exc:
    logger.warning("Simulation packages unavailable (%s); /api/plan-route will return 503.", exc)

# One SUMO session at a time per process.
_sumo_lock = threading.Lock()
_road_model: Any = None
_hospitals_cache: List[Dict[str, Any]] = []
HOSPITALS_FILE = PROJECT_ROOT / "hospitals.json"


def initialize_network_cache() -> None:
    """Load the drivable road model and hospital coverage once at startup."""
    global _road_model, _hospitals_cache
    if not HAS_SIMULATION or not DEFAULT_NET_FILE.is_file():
        logger.info("Skipping network caching: simulation dependencies or network file not found.")
        return
    try:
        t0 = time.time()
        _road_model = RoadModel.load(DEFAULT_NET_FILE)
        _hospitals_cache = json.loads(HOSPITALS_FILE.read_text(encoding="utf-8"))
        logger.info(
            "Loaded road model (%d drivable junctions) and %d hospitals in %.2fs.",
            len(_road_model.junctions), len(_hospitals_cache), time.time() - t0,
        )
    except Exception as exc:
        logger.error("Failed to initialize network cache: %s", exc)


# Initialize on module load
initialize_network_cache()


# ---------------------------------------------------------------------------
# FastAPI Application & Request / Response Models
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Delhi EMS Route Dispatch API",
    description="Adaptive Emergency Ambulance Routing powered by Volatility-Adaptive QPSO (VA-QPSO).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlanRouteRequest(BaseModel):
    incident_lat: float = Field(..., description="Latitude of the patient pickup / incident location")
    incident_lon: float = Field(..., description="Longitude of the patient pickup / incident location")
    hospital_name: Optional[str] = Field(None, description="Optional target hospital name (substring match)")
    scenario_tier: str = Field("medium", description="Traffic scenario ('low', 'medium', 'high')")
    seed: Optional[int] = Field(42, description="Optimizer seed (SUMO traffic always uses seed 42)")
    use_live_sumo: bool = Field(True, description="Must be true: routes are only computed from SUMO traffic")
    num_stops: int = Field(8, description="Stops including pickup and network exit (4-12)")
    drive_through: bool = Field(
        False,
        description=(
            "Drive an ambulance through SUMO to the network exit and report its measured time "
            "(slower; capped at 900 simulated seconds). False returns the planner estimate."
        ),
    )


class PlanRouteResponse(BaseModel):
    scenario: str
    algorithm: str
    seed: int
    stops: List[Dict[str, Any]]
    selected_hospital: Dict[str, Any]
    hospital_candidates: List[Dict[str, Any]]
    path: List[Dict[str, Any]]
    # One entry per measurement: the dispatch snapshot, plus one per simulated
    # second when drive_through is true.
    metrics_over_time: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    # Measured arrival (timing.status == "arrived"), time simulated before the
    # run stopped, or the planner estimate; timing.kind and timing.status say which.
    completion_time: float
    volatility_index: float
    beta: float
    eta_seconds: Optional[float]
    source: str
    # Best planner score (seconds) after each VA-QPSO iteration.
    best_score_history: Optional[List[float]] = None
    tier: Optional[str] = None
    path_source: Optional[str] = None
    timing: Optional[Dict[str, Any]] = None
    traffic_at_dispatch: Optional[Dict[str, Any]] = None
    final_leg: Optional[Dict[str, Any]] = None
    pickup_snap_m: Optional[float] = None


# ---------------------------------------------------------------------------
# Core Live Routing Pipeline
# ---------------------------------------------------------------------------

def resolve_hospital(requested_name: Optional[str]) -> Dict[str, Any]:
    """
    Requested hospital (substring match), else the one with the shortest
    unsimulated final leg (hospitals.json is sorted that way). No hospital is
    inside the simulated network, so the route always ends at the hospital's
    network exit junction.
    """
    if not _hospitals_cache:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "hospitals.json is not loaded.")
    if requested_name:
        wanted = requested_name.lower().strip()
        for h in _hospitals_cache:
            if wanted in h["name"].lower():
                return h
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"No hospital matches {requested_name!r}.")
    return _hospitals_cache[0]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Basic service health check")
def health_check() -> Dict[str, Any]:
    """Return health status, service identifiers, and SUMO availability."""
    return {
        "status": "healthy",
        "service": "trafficmgmt-adaptive-routing",
        "version": "1.1.0",
        "sumo_available": bool(HAS_SIMULATION and _road_model is not None),
        "live_pipeline": "sumo_warm_state_va_qpso",
    }


@app.post("/api/plan-route", response_model=PlanRouteResponse, summary="Compute live route via VA-QPSO")
def plan_route(payload: PlanRouteRequest) -> PlanRouteResponse:
    """
    Live route computation, deterministic for a given request:
    1. Snap the incident to the nearest drivable junction; pick the hospital.
    2. Load the tier's SUMO state (seed 42) and measure traffic at dispatch (t = 240 s).
    3. VA-QPSO orders the planner waypoints between pickup and network exit.
    4. drive_through=false: return the planner estimate (per-edge travel times at the
       measured speeds). drive_through=true: drive an ambulance through SUMO and return
       its measured time, or the time simulated before the cap or a teleport.
    There is no fallback: without SUMO the endpoint returns 503.
    """
    if not payload.use_live_sumo:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "use_live_sumo=false is no longer supported: routes are computed from SUMO traffic only.",
        )
    if not HAS_SIMULATION or _road_model is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "SUMO is not available on this host, so no live route can be computed.",
        )

    t_start = time.time()
    seed = payload.seed if payload.seed is not None else 42
    tier = payload.scenario_tier.lower()
    if tier not in TIERS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"scenario_tier must be one of {list(TIERS)}.")

    hospital = resolve_hospital(payload.hospital_name)
    pickup, snap_m = _road_model.nearest_junction(payload.incident_lat, payload.incident_lon)
    exit_junction = hospital["exit_junction"]
    num_stops = max(4, min(12, payload.num_stops))
    waypoints = _road_model.default_waypoints(num_stops - 2, exclude=[pickup, exit_junction])
    stops = [("junction", pickup)] + [("junction", w) for w in waypoints] + [("junction", exit_junction)]

    if not _sumo_lock.acquire(timeout=120):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "SUMO is busy with another request.")
    try:
        with DispatchSession(tier, _road_model) as session:
            dispatch_sample = session.last
            plan = session.plan(stops, "va_qpso", seed)
            drive = session.drive(plan, "va_qpso", seed) if payload.drive_through else None
            result = build_payload(
                model=_road_model,
                scenario=f"live_{tier}",
                tier=tier,
                algorithm="va_qpso",
                seed=seed,
                plan=plan,
                dispatch_sample=dispatch_sample,
                hospital=hospital,
                hospitals=_hospitals_cache,
                pickup_label=f"Patient pickup (junction {pickup}, {snap_m:.0f} m from the chosen point)",
                pickup_snap_m=snap_m,
                drive=drive,
                edge_time=session.edge_time,
                sirens=False,
            )
    finally:
        _sumo_lock.release()

    v0 = dispatch_sample.volatility_index
    timing = result["timing"]
    logger.info(
        "Live route %s tier=%s seed=%d: %s %s (%.0f s) in %.1f s wall.",
        hospital["name"], tier, seed, timing["kind"], timing["status"],
        result["completion_time"], time.time() - t_start,
    )
    return PlanRouteResponse(
        **result,
        volatility_index=round(v0, 4),
        beta=round(beta_floor("va_qpso", v0), 4),
        eta_seconds=timing["seconds_to_exit"],
        source="live_sumo_drive" if payload.drive_through else "live_sumo_snapshot",
    )


def resolve_run_file(run_id: str, data_dir: Path) -> Path:
    """Find the requested run JSON file with fallbacks."""
    clean_id = run_id.removesuffix(".json")
    if not SAFE_RUN_ID.fullmatch(clean_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid run ID.")

    candidates = [
        data_dir / f"{clean_id}.json",
        data_dir / f"hero_{clean_id}.json",
        data_dir / f"hero_{clean_id}_va_qpso.json",
        data_dir / f"hero_{clean_id}_fixed_beta_qpso.json",
        data_dir / clean_id,
    ]
    found = next((c for c in candidates if c.is_file()), None)
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found in {data_dir.name}/.",
        )
    return found


@app.get("/api/hero-runs", summary="Serve default hero run or catalog")
def get_hero_runs(
    run_id: Optional[str] = Query(None, description="Optional hero run ID"),
    catalog: bool = Query(False, description="Return catalog index manifest if true"),
) -> Any:
    """
    GET /api/hero-runs:
    - If `catalog=True`: returns available scenarios manifest (index.json).
    - If `run_id` is supplied: serves that specific precomputed run.
    - Default: serves `hero_medium_va_qpso.json`.
    """
    data_dir = getattr(app.state, "data_dir", DEFAULT_DATA_DIR)

    if catalog:
        index_file = data_dir / "index.json"
        if index_file.is_file():
            return json.loads(index_file.read_text(encoding="utf-8"))
        # Synthesize catalog from available JSON files
        hero_files = sorted(f.name for f in data_dir.glob("hero_*.json"))
        return {"scenarios": hero_files}

    target_id = run_id or "hero_medium_va_qpso"
    target_path = resolve_run_file(target_id, data_dir)
    return json.loads(target_path.read_text(encoding="utf-8"))


@app.get("/api/hero-runs/{run_id}", summary="Serve specific precomputed hero run JSON")
def get_specific_hero_run(run_id: str) -> Any:
    """Serve specific precomputed hero run JSON by run_id."""
    data_dir = getattr(app.state, "data_dir", DEFAULT_DATA_DIR)
    target_path = resolve_run_file(run_id, data_dir)
    return json.loads(target_path.read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}", summary="Legacy run replay endpoint")
def get_run(run_id: str) -> Any:
    """Backward compatibility with existing frontend /api/runs/{id} calls."""
    data_dir = getattr(app.state, "data_dir", DEFAULT_DATA_DIR)
    target_path = resolve_run_file(run_id, data_dir)
    return json.loads(target_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Static Web Assets (Frontend, PWA Manifest, Service Worker)
# ---------------------------------------------------------------------------

@app.get("/", summary="Serve the Mission Control UI")
@app.get("/index.html", summary="Serve the Mission Control UI")
def serve_index() -> FileResponse:
    web_dist = PROJECT_ROOT / "web" / "dist" / "index.html"
    index_path = web_dist if web_dist.is_file() else PROJECT_ROOT / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(index_path, media_type="text/html; charset=utf-8")


@app.get("/manifest.json")
def serve_manifest() -> FileResponse:
    path = PROJECT_ROOT / "manifest.json"
    return FileResponse(path, media_type="application/manifest+json")


@app.get("/sw.js")
def serve_service_worker() -> FileResponse:
    path = PROJECT_ROOT / "sw.js"
    return FileResponse(path, media_type="application/javascript")


@app.get("/icon-192.png")
def serve_icon_192() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "icon-192.png", media_type="image/png")


@app.get("/icon-512.png")
def serve_icon_512() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "icon-512.png", media_type="image/png")


@app.get("/icon.svg")
def serve_icon_svg() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "icon.svg", media_type="image/svg+xml")


# Mount frontend_data static folder for direct asset lookups
if DEFAULT_DATA_DIR.is_dir():
    app.mount("/frontend_data", StaticFiles(directory=str(DEFAULT_DATA_DIR)), name="frontend_data")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    port_default = int(os.environ.get("PORT", "8000"))
    host_default = os.environ.get("HOST", "0.0.0.0")
    parser = argparse.ArgumentParser(description="Serve the Adaptive Ambulance Route Control API & Web UI")
    parser.add_argument("--host", default=host_default, help=f"Host to bind (default: {host_default})")
    parser.add_argument("--port", default=port_default, type=int, help=f"Port to bind (default: {port_default})")
    parser.add_argument("--data-dir", default="frontend_data", help="Directory containing exported hero JSON")
    args = parser.parse_args()

    app.state.data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    logger.info("Starting FastAPI server at http://%s:%d (data: %s)", args.host, args.port, app.state.data_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
