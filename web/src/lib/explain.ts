import { SOURCE_WORD, TIER_WORD, formatDuration, shortName } from './format';
import { explorationKept } from './search';
import { trafficAt } from './timeline';
import { formatDistance, tripOf } from './trip';
import type { DecisionEvent, RunData } from './types';

// Plain-language text built from fixed templates and real fields only. A
// sentence is dropped when a field it needs is missing; nothing is generated
// freely.

export interface TimelineEntry {
  t: number;
  kind: 'dispatch' | 'replan' | 'arrival';
  title: string;
  /** Arrival estimate change, only when the event carries both values. */
  eta?: { before: number; after: number };
  /** The server's own wording, shown as secondary text. */
  serverNote?: string;
}

function hasEta(e: DecisionEvent): e is DecisionEvent & { eta_before: number; eta_after: number } {
  return typeof e.eta_before === 'number' && typeof e.eta_after === 'number';
}

function endTitle(run: RunData): string {
  const trip = tripOf(run);
  switch (trip.state) {
    case 'arrived':
      return `Reached: ${trip.destinationLabel}`;
    case 'estimate':
      return `Estimated arrival (planner estimate): ${trip.destinationLabel}`;
    case 'not_arrived_within_cap':
      return 'Simulation limit reached: the ambulance was not at the network exit';
    case 'teleported':
      return 'No valid arrival: SUMO teleported the ambulance out of a jam';
    default:
      return 'End of recording';
  }
}

export function buildTimeline(run: RunData): TimelineEntry[] {
  const end = run.completion_time;
  const destination = tripOf(run).destinationLabel;
  const entries: TimelineEntry[] = [];
  let sawEnd = false;

  // Recorded fixed-schedule files contain events after arrival (e.g. T+140 s); they are not part of this trip.
  run.events
    .filter((e) => e.t <= end)
    .forEach((e) => {
      if (e.type === 'arrival' || e.type === 'not_arrived' || e.type === 'teleported') {
        sawEnd = true;
        entries.push({ t: e.t, kind: 'arrival', title: endTitle(run), serverNote: e.detail });
      } else if (e.type === 'dispatch' || (e.t === 0 && e.type !== 'waypoint')) {
        entries.push({ t: e.t, kind: 'dispatch', title: `Dispatched. Destination: ${destination}`, serverNote: e.detail });
      } else if (e.type === 'replan') {
        const m = trafficAt(run, e.t);
        const traffic = m ? `Speeds ${TIER_WORD[m.tier].toLowerCase()}. ` : '';
        entries.push({
          t: e.t,
          kind: 'replan',
          title: `${traffic}Route re-planned`,
          eta: hasEta(e) ? { before: e.eta_before, after: e.eta_after } : undefined,
          serverNote: e.detail,
        });
      }
    });

  if (!sawEnd) entries.push({ t: end, kind: 'arrival', title: endTitle(run) });
  return entries.sort((a, b) => a.t - b.t);
}

/** "35 s sooner" / "4 s later" from an arrival-estimate change. */
export function etaChangeWords(eta: { before: number; after: number }): string {
  const d = eta.before - eta.after;
  return d >= 0 ? `arrival ${Math.round(d)} s sooner` : `arrival ${Math.round(-d)} s later`;
}

export interface Sentence {
  text: string;
  /** The exact fields and values used, for "Show the data behind this". */
  fields: string[];
}

const SOURCE_SENTENCE: Record<string, string> = {
  live_sumo_snapshot:
    'Traffic measured in SUMO at dispatch; the time is the planner estimate from those speeds, not a simulated drive.',
  live_sumo_drive: 'An ambulance was driven through SUMO from the measured traffic; the time is the simulated one.',
};

export function whyThisRoute(run: RunData, kind: 'live' | 'recorded', baseline: RunData | null): Sentence[] {
  const out: Sentence[] = [];
  const dest = run.selected_hospital;
  const destM = dest?.straight_line_from_exit_m;
  if (dest && destM !== undefined) {
    const known = run.hospital_candidates.filter((h) => h.straight_line_from_exit_m !== undefined);
    out.push({
      text: `${shortName(dest.name)} is the closest of ${known.length} hospitals to the simulated map: ${formatDistance(destM)} straight-line beyond its network exit. That distance is an estimate and is not part of the time.`,
      fields: ['selected_hospital.name', `hospital_candidates[].straight_line_from_exit_m = ${known.map((h) => h.straight_line_from_exit_m).join(', ')}`],
    });
  }

  const first = trafficAt(run, 0);
  if (first) {
    const congestion =
      first.vehicles !== undefined && first.mean_vehicle_speed_kmh != null
        ? `${first.vehicles} vehicles averaging ${first.mean_vehicle_speed_kmh.toFixed(1)} km/h (${first.stopped_vehicles ?? 0} stopped); `
        : '';
    out.push({
      text: `At dispatch: ${congestion}speeds were ${TIER_WORD[first.tier].toLowerCase()} (unpredictability ${first.volatility_index.toFixed(2)}).`,
      fields: [
        `metrics_over_time[0].tier = ${first.tier}`,
        `volatility_index = ${first.volatility_index}`,
        ...(first.vehicles !== undefined ? [`vehicles = ${first.vehicles}`, `mean_vehicle_speed_kmh = ${first.mean_vehicle_speed_kmh}`, `stopped_vehicles = ${first.stopped_vehicles}`] : []),
      ],
    });
  }

  buildTimeline(run)
    .filter((e) => e.kind === 'replan')
    .forEach((e) => {
      const m = trafficAt(run, e.t);
      out.push({
        text: `At T+${Math.round(e.t)} s speeds were ${m ? TIER_WORD[m.tier].toLowerCase() : 'changing'}${m && m.mean_vehicle_speed_kmh != null ? ` (network mean ${m.mean_vehicle_speed_kmh.toFixed(1)} km/h)` : ''}; the route was re-planned${e.eta ? `, ${etaChangeWords(e.eta)}` : ''}.`,
        fields: [`events[].t = ${e.t}`, ...(e.eta ? [`eta_before = ${e.eta.before}`, `eta_after = ${e.eta.after}`] : []), ...(m ? [`tier = ${m.tier}`] : [])],
      });
    });

  if (kind === 'recorded') {
    out.push({ text: 'Based on a recorded run.', fields: ['frontend_data/hero_*.json'] });
  } else if (run.source) {
    out.push({ text: SOURCE_SENTENCE[run.source] ?? `Based on ${SOURCE_WORD[run.source] ?? run.source}.`, fields: [`source = ${run.source}`] });
  }

  if (baseline) {
    const a = tripOf(run);
    const b = tripOf(baseline);
    const fields = [
      `timing.status = ${run.timing?.status}`,
      `fixed timing.status = ${baseline.timing?.status}`,
      `completion_time = ${run.completion_time}`,
      `fixed completion_time = ${baseline.completion_time}`,
    ];
    if (a.arrived && b.arrived) {
      const d = baseline.completion_time - run.completion_time;
      out.push({
        text: `On the same recorded traffic, the fixed schedule took ${formatDuration(baseline.completion_time)} to the network exit (adaptive ${formatDuration(Math.abs(d))} ${d >= 0 ? 'sooner' : 'later'}).`,
        fields,
      });
    } else {
      const word = (arrived: boolean, duration: number) => (arrived ? `reached the exit in ${formatDuration(duration)}` : 'did not reach the exit');
      out.push({
        text: `On the same recorded traffic the adaptive run ${word(a.arrived, a.duration)} and the fixed schedule ${word(b.arrived, b.duration)} within the ${formatDuration(a.capS ?? run.completion_time)} simulation limit, so no time difference can be given.`,
        fields,
      });
    }
  }
  return out;
}

/** "What the algorithm is doing now", at playback time t. */
export function doingNow(run: RunData, t: number): string[] {
  const out: string[] = [];
  const m = trafficAt(run, t);
  if (t >= run.completion_time) {
    const trip = tripOf(run);
    out.push(
      trip.arrived
        ? `The ambulance reached the network exit at T+${Math.round(run.completion_time)} s. No further searches run for this trip.`
        : trip.state === 'estimate'
          ? `End of the planner estimate (T+${Math.round(run.completion_time)} s). No vehicle was driven for this trip.`
          : `The simulation stopped at T+${Math.round(run.completion_time)} s with the ambulance not at the network exit.`,
    );
  }
  if (m) {
    out.push(`Speeds are ${TIER_WORD[m.tier].toLowerCase()} (unpredictability ${m.volatility_index.toFixed(2)})${m.mean_vehicle_speed_kmh != null ? `; network mean speed ${m.mean_vehicle_speed_kmh.toFixed(1)} km/h, ${m.stopped_vehicles ?? 0} vehicles stopped` : ''}.`);
    if (run.algorithm === 'va_qpso') {
      const kept = Math.round(explorationKept(m.beta) * 100);
      const floor = m.beta.toFixed(2);
      out.push(
        kept === 0
          ? 'A search started now would contract fully onto the best route found (search-breadth floor 0.50).'
          : `A search started now would keep ${kept}% of its exploration range at the end (search-breadth floor ${floor}), so it keeps trying wider streets.`,
      );
    } else {
      out.push('The fixed schedule ignores traffic: every search contracts on the same timetable.');
    }
  }
  const replans = buildTimeline(run).filter((e) => e.kind === 'replan');
  const last = [...replans].reverse().find((e) => e.t <= t);
  const next = replans.find((e) => e.t > t);
  if (last) out.push(`Last re-plan: T+${Math.round(last.t)} s${last.eta ? `, ${etaChangeWords(last.eta)}` : ''}.`);
  if (next && t < run.completion_time) out.push(`Next re-plan in this run: T+${Math.round(next.t)} s.`);
  if (!last && !next) out.push('This run had no re-plans after dispatch.');
  return out;
}
