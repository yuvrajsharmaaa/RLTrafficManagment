import { SOURCE_WORD, TIER_WORD, formatDuration, shortName } from './format';
import { explorationKept } from './search';
import { trafficAt } from './timeline';
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

export function buildTimeline(run: RunData): TimelineEntry[] {
  const end = run.completion_time;
  const hospital = run.selected_hospital ? shortName(run.selected_hospital.name) : 'the hospital';
  const entries: TimelineEntry[] = [];
  let sawArrival = false;

  // Recorded fixed-schedule files contain events after arrival (e.g. T+140 s); they are not part of this trip.
  run.events
    .filter((e) => e.t <= end)
    .forEach((e) => {
      if (e.type === 'arrival') {
        sawArrival = true;
        entries.push({ t: e.t, kind: 'arrival', title: `Arrived at ${hospital}`, serverNote: e.detail });
      } else if (e.type === 'dispatch' || e.t === 0) {
        entries.push({ t: e.t, kind: 'dispatch', title: `Dispatched to ${hospital}`, serverNote: e.detail });
      } else {
        const m = trafficAt(run, e.t);
        const traffic = m ? `Traffic ${TIER_WORD[m.tier].toLowerCase()}. ` : '';
        entries.push({
          t: e.t,
          kind: 'replan',
          title: `${traffic}Route re-planned`,
          eta: hasEta(e) ? { before: e.eta_before, after: e.eta_after } : undefined,
          serverNote: e.detail,
        });
      }
    });

  if (!sawArrival) entries.push({ t: end, kind: 'arrival', title: `Arrived at ${hospital}` });
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
  live_bounded_sumo: 'Based on a live simulation snapshot.',
  live_calibrated_scenario: 'Based on estimated traffic, because the simulation was unavailable; treat times as approximate.',
};

export function whyThisRoute(run: RunData, kind: 'live' | 'recorded', baseline: RunData | null): Sentence[] {
  const out: Sentence[] = [];
  const dest = run.selected_hospital;
  const destTime = run.hospital_candidates.find((h) => h.name === dest?.name)?.live_travel_time_sec;
  if (dest && destTime !== undefined) {
    const faster = run.hospital_candidates.filter((h) => h.live_travel_time_sec !== undefined);
    const tied = faster.filter((h) => h.name !== dest.name && h.live_travel_time_sec === destTime).map((h) => shortName(h.name));
    out.push({
      text: `${shortName(dest.name)} has the shortest road time of ${faster.length} hospitals (${destTime.toFixed(1)} s)${tied.length ? `, tied with ${tied.join(', ')}` : ''}.`,
      fields: ['selected_hospital.name', `hospital_candidates[].live_travel_time_sec = ${faster.map((h) => h.live_travel_time_sec).join(', ')}`],
    });
  }

  const first = trafficAt(run, 0);
  if (first) {
    out.push({
      text: `Traffic was ${TIER_WORD[first.tier].toLowerCase()} at dispatch.`,
      fields: [`metrics_over_time[0].tier = ${first.tier}`, `volatility_index = ${first.volatility_index}`],
    });
  }

  buildTimeline(run)
    .filter((e) => e.kind === 'replan')
    .forEach((e) => {
      const m = trafficAt(run, e.t);
      out.push({
        text: `At T+${Math.round(e.t)} s traffic was ${m ? TIER_WORD[m.tier].toLowerCase() : 'changing'}; the route was re-planned${e.eta ? `, ${etaChangeWords(e.eta)}` : ''}.`,
        fields: [`events[].t = ${e.t}`, ...(e.eta ? [`eta_before = ${e.eta.before}`, `eta_after = ${e.eta.after}`] : []), ...(m ? [`tier = ${m.tier}`] : [])],
      });
    });

  if (kind === 'recorded') {
    out.push({ text: 'Based on a recorded run.', fields: ['frontend_data/hero_*.json'] });
  } else if (run.source) {
    out.push({ text: SOURCE_SENTENCE[run.source] ?? `Based on ${SOURCE_WORD[run.source] ?? run.source}.`, fields: [`source = ${run.source}`] });
  }

  if (baseline) {
    const d = baseline.completion_time - run.completion_time;
    out.push({
      text: `On the same recorded traffic, the fixed schedule took ${formatDuration(baseline.completion_time)} (adaptive ${formatDuration(Math.abs(d))} ${d >= 0 ? 'sooner' : 'later'}).`,
      fields: [`completion_time = ${run.completion_time}`, `fixed completion_time = ${baseline.completion_time}`],
    });
  }
  return out;
}

/** "What the algorithm is doing now", at playback time t. */
export function doingNow(run: RunData, t: number): string[] {
  const out: string[] = [];
  const m = trafficAt(run, t);
  if (t >= run.completion_time) {
    out.push(`The ambulance arrived at T+${Math.round(run.completion_time)} s. No further searches run for this trip.`);
  }
  if (m) {
    out.push(`Traffic is ${TIER_WORD[m.tier].toLowerCase()} (unpredictability ${m.volatility_index.toFixed(2)}).`);
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
