// Constants of the route search, copied from src/planner/qpso.py so the UI
// can explain the exported values. They are not sent by the server.
export const BETA_MAX = 1.0; // DEFAULT_BETA_MAX: every search starts here
export const BETA_MIN = 0.5; // DEFAULT_BETA_MIN: the fixed schedule ends here
export const VA_FLOOR_SPAN = 0.25; // va_qpso's floor rises by up to this much at V = 1

/**
 * Exported "beta" is the floor each search's anneal ends at, for both
 * variants: 0.50 + 0.25 * V for adaptive runs (recorded and live), and
 * BETA_MIN for the fixed schedule, which ignores traffic.
 */
export const FIXED_EXPORT_BETA = BETA_MIN;

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
