import { Hospital as HospitalIcon } from 'lucide-react';
import { Panel, Tooltip } from '../ui';
import { shortName } from '../../lib/format';
import type { Hospital, RunData } from '../../lib/types';
import { cx } from '../../lib/cx';

// The backend substitutes 45.0 s when a hospital can't be reached on the
// graph; a real 45.0 s can't be told apart, so it is flagged, not hidden.
const UNREACHABLE_PLACEHOLDER_S = 45;

export function HospitalCard({ run }: { run: RunData }) {
  const destName = run.selected_hospital?.name;
  const destTime = run.hospital_candidates.find((h) => h.name === destName)?.live_travel_time_sec;
  const rows = [...run.hospital_candidates].sort(
    (a, b) => (a.live_travel_time_sec ?? Infinity) - (b.live_travel_time_sec ?? Infinity),
  );

  if (rows.length === 0) {
    return (
      <Panel raised title="Hospitals by road time">
        <p className="text-body-sm text-text-3">No hospital candidates in this run.</p>
      </Panel>
    );
  }

  const row = (h: Hospital) => {
    const isDest = h.name === destName;
    const time = h.live_travel_time_sec;
    const tied = !isDest && time !== undefined && time === destTime;
    return (
      <li
        key={h.name}
        className={cx(
          'flex items-start gap-2 border-t border-border-subtle py-2 first:border-t-0',
          isDest && '-mx-2 rounded bg-accent-weak px-2 shadow-[inset_2px_0_0_var(--color-accent)]',
        )}
      >
        <HospitalIcon
          size={16}
          strokeWidth={1.75}
          aria-hidden
          className={cx('mt-0.5 shrink-0', isDest ? 'text-route-optimized' : 'text-hospital-unknown')}
        />
        <div className="flex min-w-0 flex-1 flex-col">
          <Tooltip content={`${h.name} (${h.level})`}>
            <span tabIndex={0} className="truncate text-body-sm text-text-1">{shortName(h.name)}</span>
          </Tooltip>
          <span className="text-caption text-text-3">
            {isDest ? 'Destination' : tied ? 'Tied with destination' : h.level}
          </span>
        </div>
        <span className="flex shrink-0 flex-col items-end">
          {time === undefined ? (
            <span className="text-caption text-text-3">No data</span>
          ) : (
            <span className="num text-body-sm text-text-1">{time.toFixed(1)} s</span>
          )}
          {time === UNREACHABLE_PLACEHOLDER_S && <span className="text-caption text-warning">may be unreachable</span>}
        </span>
      </li>
    );
  };

  return (
    <Panel raised title="Hospitals by road time">
      <ul aria-label="Hospitals ranked by road time">{rows.map(row)}</ul>
      <div className="mt-2 flex flex-col gap-1 border-t border-border-subtle pt-2 text-caption text-text-3">
        <span>Shortest-path road time from the incident to each hospital's entry, not the full trip.</span>
        <span>Load: no data from the system.</span>
      </div>
    </Panel>
  );
}
