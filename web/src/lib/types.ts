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
}

export interface PathPoint {
  t: number;
  lat: number;
  lon: number;
}

export interface MetricPoint {
  t: number;
  volatility_index: number;
  beta: number;
  tier: TierKey;
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
  gateway_node?: string;
  live_travel_time_sec?: number;
  status?: string;
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
  eta_seconds?: number;
  source?: string;
}

export interface PlanRouteRequest {
  incident_lat: number;
  incident_lon: number;
  scenario_tier: ScenarioTier;
  seed: number;
  use_live_sumo: boolean;
  num_stops: number;
}
