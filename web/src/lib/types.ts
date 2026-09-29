// Types for the payloads the existing backend already returns. Field names
// match server.py (PlanRouteRequest / PlanRouteResponse) and the recorded
// run files in frontend_data/ exactly.

export type TierKey = 'calm' | 'moderate' | 'turbulent';
export type ScenarioTier = 'low' | 'medium' | 'high';

export interface Stop {
  id: string;
  lat: number;
  lon: number;
  label: string;
  /** Visit role. Older files have no kind: first = pickup, last = destination. */
  kind?: 'pickup' | 'waypoint' | 'network_exit';
  /** Simulated time the ambulance passed this stop (simulated drives only). */
  reached_t?: number;
}

export interface PathPoint {
  t: number;
  lat: number;
  lon: number;
}

export interface MetricPoint {
  t: number;
  /** Unpredictability of network speeds, 0-1. Not a congestion level. */
  volatility_index: number;
  beta: number;
  /** Tier of volatility_index (unpredictability), not of congestion. */
  tier: TierKey;
  /** Congestion, measured per vehicle in SUMO at this time. */
  vehicles?: number;
  mean_vehicle_speed_kmh?: number | null;
  stopped_vehicles?: number;
}

export interface DecisionEvent {
  t: number;
  /** 'replan' in recorded runs; live runs also send 'dispatch' and 'arrival'. */
  type: string;
  detail: string;
  eta_before?: number;
  eta_after?: number;
  volatility_index?: number;
  beta?: number;
}

export interface Hospital {
  name: string;
  lat: number;
  lon: number;
  level: string;
  status?: string;
  coord_source?: string;
  /** False for every hospital on the current map: none lies on a simulated road. */
  in_network?: boolean;
  exit_junction?: string;
  exit_lat?: number;
  exit_lon?: number;
  /** Straight-line distance from the network exit to the hospital. Not simulated. */
  straight_line_from_exit_m?: number;
}

export type TimingStatus = 'arrived' | 'not_arrived_within_cap' | 'teleported' | 'estimate';

export interface Timing {
  /** simulated_drive: an ambulance driven through SUMO. planner_estimate: no vehicle was driven. */
  kind: 'simulated_drive' | 'planner_estimate';
  status: TimingStatus;
  /** Measured (or estimated) seconds to the network exit; null when the drive did not arrive. */
  seconds_to_exit: number | null;
  simulated_seconds: number | null;
  cap_s: number | null;
  driven_length_m: number | null;
  average_speed_kmh: number | null;
  planner_estimate_s: number;
  route_length_m: number;
  dispatch_sim_time_s: number;
  sumo_seed: number;
  sirens: boolean;
}

export interface FinalLeg {
  kind: 'straight_line_estimate';
  from_junction: string;
  to: string;
  straight_line_m: number;
  simulated: false;
}

export interface RunData {
  scenario: string;
  algorithm: string;
  seed?: number;
  stops: Stop[];
  hospital_candidates: Hospital[];
  selected_hospital: Hospital;
  path: PathPoint[];
  metrics_over_time: MetricPoint[];
  events: DecisionEvent[];
  completion_time: number;
  /** Live responses only. */
  volatility_index?: number;
  beta?: number;
  eta_seconds?: number | null;
  source?: string;
  tier?: ScenarioTier;
  path_source?: string;
  timing?: Timing;
  traffic_at_dispatch?: MetricPoint;
  final_leg?: FinalLeg;
  pickup_snap_m?: number | null;
  /** Best planner score (seconds) after each search iteration. */
  best_score_history?: number[];
}

export interface PlanRouteRequest {
  incident_lat: number;
  incident_lon: number;
  scenario_tier: ScenarioTier;
  seed: number;
  use_live_sumo: boolean;
  num_stops: number;
  /** Drive an ambulance through SUMO (slower, exact) instead of the planner estimate. */
  drive_through?: boolean;
}
