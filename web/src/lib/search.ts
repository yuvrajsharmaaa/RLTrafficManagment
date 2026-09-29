// Constants of the route search, copied from src/planner/qpso.py so the UI
// can explain the exported values. They are not sent by the server.
export const BETA_MAX = 1.0; // DEFAULT_BETA_MAX: every search starts here
export const BETA_MIN = 0.5; // DEFAULT_BETA_MIN: the fixed schedule ends here
export const VA_FLOOR_SPAN = 0.25; // va_qpso's floor rises by up to this much at V = 1

/**
 * Floor formula per data source. The recorded files were exported with an
 * older formula, 0.50 + 0.50 * V (every sample fits it exactly); the current
 * code, and so every live run, uses 0.50 + 0.25 * V.
 */
export const FLOOR_SPAN_RECORDED = 0.5;
export const FLOOR_SPAN_LIVE = VA_FLOOR_SPAN;

/** Value the fixed-schedule export writes for beta at every second (export_for_frontend.py). */
export const FIXED_EXPORT_BETA = 0.75;

/** Live searches are configured in server.py; the payload does not report them. */
export const LIVE_SEARCH_BUDGET = { particles: 15, iterations: 30 };

/**
 * Exploration kept at the end of a search, 0 to 1.
 *
 * Every search anneals beta from BETA_MAX down to its floor. The floor is the
 * exported "beta" for adaptive runs (beta_min + 0.25 * V). The share of the
 * full beta range the swarm still keeps at the end is
 *   (floor - BETA_MIN) / (BETA_MAX - BETA_MIN)
 * 0 = the search contracts fully (pure exploitation, as the fixed schedule
 * does); higher = it keeps searching wider streets.
 */
export function explorationKept(betaFloor: number): number {
  return Math.max(0, Math.min(1, (betaFloor - BETA_MIN) / (BETA_MAX - BETA_MIN)));
}
