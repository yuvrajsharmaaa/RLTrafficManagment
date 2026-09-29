# Bundled demo runs

`server.py` serves each file as `/api/runs/<filename-without-.json>` and the
browser can also fall back to `frontend_data/<filename>.json` when using a
static web server.

The `hero_*` files and `index.json` come from SUMO drive-throughs:

```sh
python export_for_frontend.py --export-heroes
```

Each run loads the tier's SUMO state (seed 42) at t = 225 s, measures traffic
for 15 s, dispatches at t = 240 s, plans with VA-QPSO or the fixed schedule,
and drives an ambulance vehicle through SUMO from the pickup junction to the
network exit of the default hospital, re-planning on the algorithm's cadence.
See `src/simulation/dispatch.py` for the full method. The export is
deterministic: re-running it reproduces the files exactly.

## Where each number comes from

| Field | Source |
| ----- | ------ |
| `completion_time` | SUMO arrival at the network exit when `timing.status` is `arrived`; otherwise the simulated time before the run stopped (900 s cap, or a teleport) |
| `timing` | `kind`, `status`, driven distance, average speed, planner estimate at dispatch |
| `path` | The ambulance's position each simulated second (`path_source: simulated_ambulance_trajectory`) |
| `metrics_over_time` | One SUMO measurement per simulated second: unpredictability `volatility_index`, vehicles, mean vehicle speed, stopped vehicles |
| `final_leg` | Straight-line distance from the network exit to the hospital. Not simulated and never added to any time |
| `stops` | `pickup`, auto-selected planner `waypoint`s (not real places), `network_exit` |
| `best_score_history` | Planner score (planned seconds to the exit) after each search iteration at dispatch |

`volatility_index` measures how much network speeds change; it is not a
congestion level. Congestion is `vehicles`, `mean_vehicle_speed_kmh` and
`stopped_vehicles`.

No hospital lies on the simulated roads (see `hospitals.json` and
`src/simulation/coverage.py`), so every run ends at the network exit toward
the hospital.

## `metrics_over_time[].beta`

`beta` is the floor that each re-plan's beta anneal ends at. Both variants
start at `beta_max = 1.0` and anneal linearly over QPSO iterations:

| algorithm         | exported `beta`            |
| ----------------- | -------------------------- |
| `va_qpso`         | `0.5 + 0.25 * V` (`va_beta_floor`, `src/planner/qpso.py`) |
| `fixed_beta_qpso` | `0.5` (`beta_min`; does not depend on V) |

## Live runs (`POST /api/plan-route`)

Live responses use the same field names. By default they return the planner
estimate (`timing.kind: planner_estimate`) with one traffic measurement at
dispatch. With `"drive_through": true` they drive an ambulance through SUMO
exactly as the hero export does. There is no synthetic fallback: without SUMO
the endpoint returns 503. A request with the same body returns the same
response.

## Reproducing the numbers

```sh
python reproduce_demo.py --all
```

re-checks hospital coverage, measures traffic per tier, sends a live request
twice, and re-drives every recorded run in SUMO, comparing each with the
committed file. It exits non-zero if anything differs.

## History

Files exported before 2026-09-29 used hard-coded arrival times (93.55 to
125.70 s), hand-written events with invented `eta_before`/`eta_after`, an
older beta formula, and routes that did not run from the pickup to the
hospital.
