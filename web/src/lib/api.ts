import type { PlanRouteRequest, RunData, ScenarioTier } from './types';

// Ported from the current index.html without changing a request: the same
// URLs in the same order, the same method, headers and body keys.

export interface LoadedRun {
  data: RunData;
  /**
   * True when none of the requested run's URLs answered and the medium
   * adaptive run was loaded instead. The old app hid this; the UI must say it.
   */
  isDefaultFallback: boolean;
}

function normalizeRunData(data: RunData): RunData {
  return {
    ...data,
    path: (data.path || []).sort((a, b) => a.t - b.t),
    metrics_over_time: (data.metrics_over_time || []).sort((a, b) => a.t - b.t),
    events: (data.events || []).sort((a, b) => a.t - b.t),
    // No invented default: a file without completion_time plays to its last path sample.
    completion_time: data.completion_time ?? data.path?.at(-1)?.t ?? 0,
  };
}

export async function fetchRunData(runId: string): Promise<LoadedRun> {
  const cleanId = runId.replace(/\.json$/, '');
  const candidateUrls = [
    `/api/runs/${cleanId}`,
    `/api/runs/hero_${cleanId}`,
    `frontend_data/${cleanId}.json`,
    `frontend_data/hero_${cleanId}.json`,
    `frontend_data/hero_${cleanId}_va_qpso.json`,
  ];

  for (const url of candidateUrls) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        const data = (await res.json()) as RunData;
        return { data: normalizeRunData(data), isDefaultFallback: false };
      }
    } catch {
      // Continue to next candidate
    }
  }

  // Final fallback attempt
  try {
    const fallbackRes = await fetch('frontend_data/hero_medium_va_qpso.json');
    if (fallbackRes.ok) {
      return { data: normalizeRunData((await fallbackRes.json()) as RunData), isDefaultFallback: true };
    }
  } catch {
    // fall through
  }

  throw new Error(`Could not load run data for: ${runId}`);
}

export async function planRoute(req: PlanRouteRequest): Promise<RunData> {
  const res = await fetch('/api/plan-route', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      incident_lat: req.incident_lat,
      incident_lon: req.incident_lon,
      scenario_tier: req.scenario_tier,
      seed: req.seed,
      use_live_sumo: req.use_live_sumo,
      num_stops: req.num_stops,
      drive_through: req.drive_through ?? false,
    }),
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // keep statusText
    }
    throw new Error(`Server returned ${res.status}: ${detail}`);
  }
  return normalizeRunData((await res.json()) as RunData);
}

/** Same seed rule as the old app: a new random run number per request. */
export function randomSeed(): number {
  return Math.floor(Math.random() * 1000) + 1;
}

export function recordedRunId(tier: ScenarioTier, method: 'va_qpso' | 'fixed_beta_qpso'): string {
  return `hero_${tier}_${method}`;
}
