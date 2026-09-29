import { Ambulance, CircleCheck, Info, OctagonAlert, Route, TriangleAlert } from 'lucide-react';
import { StatBlock, StatusChip, Tooltip } from '../ui';
import { formatClock, formatDuration } from '../../lib/format';
import { replanEvents } from '../../lib/timeline';
import { ABOUT_THIS_MAP, formatDistance, tripOf } from '../../lib/trip';
import type { RunData } from '../../lib/types';

const REROUTE_WINDOW_S = 3;

interface VehicleCardProps {
  run: RunData;
  /** Playback time (display rate). */
  t: number;
}

/** One ambulance per run: the data has no fleet (Phase 3, data limit 3). */
export function VehicleCard({ run, t }: VehicleCardProps) {
  const trip = tripOf(run);
  const duration = trip.duration;
  const remaining = Math.max(0, duration - t);
  const ended = t >= duration;
  const replans = replanEvents(run.events);
  const changesSoFar = replans.filter((e) => e.t <= t).length;
  const rerouting = !ended && replans.some((e) => t >= e.t && t - e.t < REROUTE_WINDOW_S);
  const progress = duration > 0 ? Math.min(1, t / duration) : 0;
  const stuck = trip.state === 'not_arrived_within_cap' || trip.state === 'teleported';
  const estimate = trip.state === 'estimate';
  const timing = run.timing;
  const covered =
    timing?.driven_length_m != null && timing.simulated_seconds
      ? { m: timing.driven_length_m, kmh: (3.6 * timing.driven_length_m) / timing.simulated_seconds }
      : null;

  const headline = (() => {
    if (stuck) {
      return ended
        ? { label: trip.state === 'teleported' ? 'No valid arrival' : 'Did not reach exit in', value: trip.state === 'teleported' ? '--:--' : formatClock(duration) }
        : { label: 'Simulated so far', value: formatClock(t) };
    }
    if (ended) return { label: estimate ? 'Estimated to exit' : 'Reached exit in', value: formatClock(duration) };
    return { label: estimate ? 'Estimated time to exit' : 'Time to exit', value: formatClock(remaining) };
  })();

  const announcement = stuck && ended
    ? trip.state === 'teleported'
      ? 'The ambulance was removed from a jam by the simulator; there is no valid arrival time.'
      : `The ambulance was not at the network exit when the simulation stopped at ${formatClock(duration)}.`
    : ended
      ? `${estimate ? 'Estimated arrival' : 'Reached network exit'} at ${formatClock(duration)}`
      : rerouting
        ? 'Route re-planned'
        : stuck
          ? `Simulated ${formatClock(Math.floor(t / 10) * 10)}`
          : `${headline.label} about ${formatClock(Math.ceil(remaining / 10) * 10)}`;

  const chip = stuck && ended ? (
    trip.state === 'teleported' ? (
      <StatusChip tone="danger" label="Teleported" icon={OctagonAlert} />
    ) : (
      <StatusChip tone="danger" label="Time limit" icon={OctagonAlert} />
    )
  ) : ended ? (
    estimate ? <StatusChip tone="info" label="Estimate" icon={Info} /> : <StatusChip tone="success" label="At exit" icon={CircleCheck} />
  ) : rerouting ? (
    <StatusChip tone="warning" label="Re-planning" icon={Route} />
  ) : (
    <StatusChip tone="info" label="En route" icon={Ambulance} />
  );

  return (
    <section aria-label="Ambulance status" className="flex flex-col gap-3">
      <span className="sr-only" aria-live="polite">{announcement}</span>
      <div className="flex items-start justify-between gap-2">
        <StatBlock label={headline.label} value={headline.value} size="xl" />
        {chip}
      </div>

      {trip.hospital && (
        <div className="flex flex-col gap-0.5">
          <Tooltip content={ABOUT_THIS_MAP}>
            <p tabIndex={0} className="flex w-fit items-center gap-1 text-title text-text-1">
              {trip.destinationLabel}
              <Info size={14} strokeWidth={1.75} aria-label="About this map" className="text-text-3" />
            </p>
          </Tooltip>
          {trip.finalLeg && (
            <p className="text-caption text-text-3">
              Then about <span className="num text-text-2">{formatDistance(trip.finalLeg.straight_line_m)}</span> straight-line
              to the hospital: estimated remaining distance, not simulated and not in the time above.
            </p>
          )}
        </div>
      )}

      {stuck && ended && (
        <p className="flex gap-1.5 rounded bg-danger-weak px-2 py-1.5 text-caption text-text-1">
          <TriangleAlert size={14} strokeWidth={1.75} aria-hidden className="mt-0.5 shrink-0 text-danger" />
          {trip.state === 'teleported'
            ? 'SUMO teleported the ambulance out of a jam, so this run has no valid arrival time.'
            : `Not at the network exit when the ${formatDuration(trip.capS ?? duration)} simulation limit was reached${
                covered ? `: it had driven ${formatDistance(covered.m)} at an average ${covered.kmh.toFixed(1)} km/h` : ''
              }. No arrival time exists for this run.`}
        </p>
      )}

      <div className="flex flex-col gap-1">
        <div className="flex justify-between text-label text-text-2">
          <span>{stuck ? 'Simulated time used' : 'Progress'}</span>
          <span className="num">{Math.round(progress * 100)}%</span>
        </div>
        <div
          role="progressbar"
          aria-label={stuck ? 'Simulated time used' : 'Route progress'}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(progress * 100)}
          className="h-1.5 overflow-hidden rounded-sm bg-raised"
        >
          <div className={stuck ? 'h-full bg-danger' : 'h-full bg-accent'} style={{ width: `${progress * 100}%` }} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <StatBlock
          label={`Time to exit (${trip.timeSource.toLowerCase()})`}
          value={trip.arrived || estimate ? formatDuration(duration) : 'No arrival'}
        />
        <StatBlock label="Route changes" value={`${changesSoFar} of ${replans.length}`} />
      </div>
    </section>
  );
}
