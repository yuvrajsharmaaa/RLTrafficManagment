import type { ScenarioTier, TierKey } from './types';

/** Countdown / arrival time as mm:ss. */
export function formatClock(seconds: number): string {
  const rem = Math.max(0, seconds);
  const m = Math.floor(rem / 60);
  const s = Math.floor(rem % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

/** Durations: "38.0 s" under a minute, "2 min 06 s" from a minute. */
export function formatDuration(seconds: number): string {
  const s = Math.abs(seconds);
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  return `${m} min ${r.toString().padStart(2, '0')} s`;
}

export const TIER_WORD: Record<TierKey, string> = {
  calm: 'Steady',
  moderate: 'Changing',
  turbulent: 'Unstable',
};

export const SCENARIO_WORD: Record<ScenarioTier, string> = {
  low: 'Light',
  medium: 'Moderate',
  high: 'Heavy',
};

/** Labels in the data carry emoji; the UI never shows them. */
export function cleanLabel(label: string): string {
  return label.replace(/\p{Extended_Pictographic}️?/gu, '').replace(/\s+/g, ' ').trim();
}

/** "All India Institute of Medical Sciences (AIIMS), New Delhi" -> text before the first comma. */
export function shortName(name: string): string {
  return name.split(',')[0]?.trim() ?? name;
}

export function formatCoord(lat: number, lon: number): string {
  return `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
}

export const SOURCE_WORD: Record<string, string> = {
  live_bounded_sumo: 'Live simulation snapshot',
  live_calibrated_scenario: 'Estimated traffic',
};
