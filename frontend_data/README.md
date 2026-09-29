# Bundled demo runs

Place the exported hero-run JSON files here. `server.py` serves each file as
`/api/runs/<filename-without-.json>` and the browser can also fall back to
`frontend_data/<filename>.json` when using a static web server.

Generate matched hero data with:

```sh
python3 export_for_frontend.py bundle --manifest frontend_runs.json \
  --net networks/delhi/delhi_intersection.net.xml --output-dir frontend_data
```

The `hero_*` files and `index.json` come from a live SUMO run per tier:

```sh
python export_for_frontend.py --export-heroes
```

## `metrics_over_time[].beta`

`beta` is the floor that each re-plan's beta anneal ends at. Both variants
start at `beta_max = 1.0` and anneal linearly over QPSO iterations:

| algorithm         | exported `beta`            |
| ----------------- | -------------------------- |
| `va_qpso`         | `0.5 + 0.25 * V` (`va_beta_floor`, `src/planner/qpso.py`) |
| `fixed_beta_qpso` | `0.5` (`beta_min`; does not depend on V) |

Files exported before 2026-09-29 used an older formula (`0.5 + 0.5 * V`) for
`va_qpso` and a constant placeholder `0.75` for `fixed_beta_qpso`.

## Live runs (`POST /api/plan-route`)

Live responses use the same field names. `metrics_over_time` holds one
sample at `t = 0`: the volatility measured for that request (bounded SUMO
snapshot or calibrated fallback). No per-second series is generated.
`best_score_history` (optional) lists the best fitness after each VA-QPSO
iteration.
