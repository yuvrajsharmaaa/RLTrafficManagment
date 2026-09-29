import type { DecisionEvent, MetricPoint, PathPoint, RunData } from './types';

/** Index of the last element with .t <= t, or -1. Arrays are sorted by t. */
function lastAtOrBefore<T extends { t: number }>(items: T[], t: number): number {
  let lo = 0;
  let hi = items.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const item = items[mid];
    if (item && item.t <= t) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

/** Position along the recorded path at time t (linear between samples). Ported from the old app. */
export function positionAt(path: PathPoint[], t: number): [number, number] | null {
  const first = path[0];
  const last = path.at(-1);
  if (!first || !last) return null;
  if (t <= first.t) return [first.lat, first.lon];
  if (t >= last.t) return [last.lat, last.lon];
  const i = Math.max(0, lastAtOrBefore(path, t));
  const p1 = path[i] ?? first;
  const p2 = path[Math.min(path.length - 1, i + 1)] ?? last;
  const dt = p2.t - p1.t;
  const alpha = dt > 0 ? (t - p1.t) / dt : 0;
  return [p1.lat + alpha * (p2.lat - p1.lat), p1.lon + alpha * (p2.lon - p1.lon)];
}

/** Index of the path segment the vehicle is on at time t. */
export function segmentIndexAt(path: PathPoint[], t: number): number {
  return Math.max(0, lastAtOrBefore(path, t));
}

/**
 * Metric sample in effect at time t. The old app indexed by floor(t), which
 * read one second late because recorded samples start at t = 1 (Phase 1 P19).
 */
export function metricAt(metrics: MetricPoint[], t: number): MetricPoint | null {
  if (metrics.length === 0) return null;
  const i = lastAtOrBefore(metrics, t);
  return metrics[Math.max(0, i)] ?? null;
}

/** Live responses carry `source` and a single measured volatility_index. */
export function isLiveRun(run: RunData): boolean {
  return run.source !== undefined && run.volatility_index !== undefined;
}

/**
 * Traffic reading to display at time t.
 *
 * Recorded runs: the per-second sample (logged during the SUMO run).
 * Live runs: server.py measures V once, then fills metrics_over_time with a
 * generated drift (V + 0.03 * sin(t / 10)) that is not a measurement. The UI
 * therefore uses the t = 0 sample, which equals the measured value, for the
 * whole trip.
 */
export function trafficAt(run: RunData, t: number): MetricPoint | null {
  if (isLiveRun(run)) {
    const first = run.metrics_over_time[0];
    return first ? { ...first, t } : null;
  }
  return metricAt(run.metrics_over_time, t);
}

export function eventsUpTo(events: DecisionEvent[], t: number): DecisionEvent[] {
  return events.filter((e) => e.t <= t);
}

/** Route changes after dispatch: 'replan' events with t > 0. */
export function replanEvents(events: DecisionEvent[]): DecisionEvent[] {
  return events.filter((e) => e.type === 'replan' && e.t > 0);
}

/** Compass bearing in degrees from a to b (0 = north), for the heading arrow. */
export function bearing(a: [number, number], b: [number, number]): number {
  const toRad = Math.PI / 180;
  const [lat1, lon1] = a;
  const [lat2, lon2] = b;
  const y = Math.sin((lon2 - lon1) * toRad) * Math.cos(lat2 * toRad);
  const x =
    Math.cos(lat1 * toRad) * Math.sin(lat2 * toRad) -
    Math.sin(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.cos((lon2 - lon1) * toRad);
  return (Math.atan2(y, x) / toRad + 360) % 360;
}
