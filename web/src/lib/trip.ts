import { shortName } from './format';
import type { FinalLeg, Hospital, RunData, TimingStatus } from './types';

// What the UI may say about a trip, from the payload's own timing fields.
// Every screen that shows a time or a destination reads it from here, so the
// wording cannot drift between cards.

export type TripState = TimingStatus | 'unlabelled';

export interface Trip {
  state: TripState;
  /** Seconds on the playback clock (simulated, estimated, or simulated before the run stopped). */
  duration: number;
  /** True only when SUMO reported the ambulance reaching the exit. */
  arrived: boolean;
  /** "Network exit toward Dr. Ram Manohar Lohia Hospital" (or the hospital name for in-network hospitals). */
  destinationLabel: string;
  hospital: Hospital | null;
  /** The unsimulated remainder of the trip; never added to duration. */
  finalLeg: FinalLeg | null;
  /** One short phrase for where the time comes from. */
  timeSource: string;
  capS: number | null;
}

const TIME_SOURCE: Record<TripState, string> = {
  arrived: 'Simulated drive',
  not_arrived_within_cap: 'Simulated drive',
  teleported: 'Simulated drive',
  estimate: 'Planner estimate',
  unlabelled: 'Time source not reported',
};

export function tripOf(run: RunData): Trip {
  const timing = run.timing;
  const state: TripState = timing?.status ?? 'unlabelled';
  const hospital = run.selected_hospital ?? null;
  const outside = hospital?.in_network === false;
  return {
    state,
    duration: run.completion_time,
    arrived: state === 'arrived',
    destinationLabel: hospital
      ? outside
        ? `Network exit toward ${shortName(hospital.name)}`
        : shortName(hospital.name)
      : 'Destination',
    hospital,
    finalLeg: run.final_leg ?? null,
    timeSource: TIME_SOURCE[state],
    capS: timing?.cap_s ?? null,
  };
}

/** "1.7 km" / "640 m". */
export function formatDistance(m: number): string {
  return m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`;
}

/** Map-pin and status-bar wording for the end of playback. */
export function endWord(trip: Trip): string {
  switch (trip.state) {
    case 'arrived':
      return 'Reached network exit';
    case 'estimate':
      return 'Estimated arrival at exit';
    case 'not_arrived_within_cap':
      return 'Did not reach exit';
    case 'teleported':
      return 'No valid arrival';
    default:
      return 'End of recording';
  }
}

export const ABOUT_THIS_MAP =
  'The simulated road network covers central Connaught Place only (about 0.9 × 0.8 km of roads). ' +
  'No hospital lies on a simulated road, so every trip is simulated to the network exit closest to the ' +
  'hospital; the rest is shown as a straight-line distance, which is an estimate and is never added to the time.';
