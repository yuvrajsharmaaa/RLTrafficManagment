import { Ambulance, CircleCheck, Route } from 'lucide-react';
import { StatBlock, StatusChip, Tooltip } from '../ui';
import { formatClock, formatDuration, shortName } from '../../lib/format';
import { replanEvents } from '../../lib/timeline';
import type { RunData } from '../../lib/types';

const REROUTE_WINDOW_S = 3;

interface VehicleCardProps {
  run: RunData;
  /** Playback time (display rate). */
  t: number;
}

/** One ambulance per run: the data has no fleet (Phase 3, data limit 3). */
export function VehicleCard({ run, t }: VehicleCardProps) {
  const duration = run.completion_time;
  const remaining = Math.max(0, duration - t);
  const arrived = t >= duration;
  const replans = replanEvents(run.events);
  const changesSoFar = replans.filter((e) => e.t <= t).length;
  const rerouting = !arrived && replans.some((e) => t >= e.t && t - e.t < REROUTE_WINDOW_S);
  const progress = duration > 0 ? Math.min(1, t / duration) : 0;
  const hospital = run.selected_hospital;
  // Screen readers hear the countdown in 10 s steps, on re-plans and on arrival, not every tick.
  const announcement = arrived
    ? `Arrived in ${formatClock(duration)}`
    : rerouting
      ? 'Route re-planned'
      : `Arrival in about ${formatClock(Math.ceil(remaining / 10) * 10)}`;

  return (
    <section aria-label="Ambulance status" className="flex flex-col gap-3">
      <span className="sr-only" aria-live="polite">{announcement}</span>
      <div className="flex items-start justify-between gap-2">
        {arrived ? (
          <StatBlock label="Arrived in" value={formatClock(duration)} size="xl" />
        ) : (
          <StatBlock label="Arrival in" value={formatClock(remaining)} size="xl" />
        )}
        {arrived ? (
          <StatusChip tone="success" label="Arrived" icon={CircleCheck} />
        ) : rerouting ? (
          <StatusChip tone="warning" label="Re-planning" icon={Route} />
        ) : (
          <StatusChip tone="info" label="En route" icon={Ambulance} />
        )}
      </div>

      {hospital && (
        <Tooltip content={`${hospital.name} (${hospital.level})`}>
          <p tabIndex={0} className="w-fit text-title text-text-1">
            {shortName(hospital.name)}
          </p>
        </Tooltip>
      )}

      <div className="flex flex-col gap-1">
        <div className="flex justify-between text-label text-text-2">
          <span>Progress</span>
          <span className="num">{Math.round(progress * 100)}%</span>
        </div>
        <div
          role="progressbar"
          aria-label="Route progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(progress * 100)}
          className="h-1.5 overflow-hidden rounded-sm bg-raised"
        >
          <div className="h-full bg-accent" style={{ width: `${progress * 100}%` }} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <StatBlock label="Total route time" value={formatDuration(duration)} />
        <StatBlock label="Route changes" value={`${changesSoFar} of ${replans.length}`} />
      </div>
    </section>
  );
}
