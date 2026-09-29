import { Hospital as HospitalIcon, Info } from 'lucide-react';
import { Panel, Tooltip } from '../ui';
import { shortName } from '../../lib/format';
import { ABOUT_THIS_MAP, formatDistance } from '../../lib/trip';
import type { Hospital, RunData } from '../../lib/types';
import { cx } from '../../lib/cx';

export function HospitalCard({ run }: { run: RunData }) {
  const destName = run.selected_hospital?.name;
  const rows = [...run.hospital_candidates].sort(
    (a, b) => (a.straight_line_from_exit_m ?? Infinity) - (b.straight_line_from_exit_m ?? Infinity),
  );
  const about = (
    <Tooltip content={ABOUT_THIS_MAP}>
      <span tabIndex={0} aria-label="About this map" className="flex items-center gap-1 text-caption text-text-3">
        <Info size={14} strokeWidth={1.75} aria-hidden />
        About this map
      </span>
    </Tooltip>
  );

  if (rows.length === 0) {
    return (
      <Panel raised title="Hospitals" actions={about}>
        <p className="text-body-sm text-text-3">No hospital candidates in this run.</p>
      </Panel>
    );
  }

  const row = (h: Hospital) => {
    const isDest = h.name === destName;
    const d = h.straight_line_from_exit_m;
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
          <Tooltip content={`${h.name} (${h.level})${h.coord_source ? `. Location: ${h.coord_source}` : ''}`}>
            <span tabIndex={0} className="truncate text-body-sm text-text-1">{shortName(h.name)}</span>
          </Tooltip>
          <span className="text-caption text-text-3">
            {isDest ? 'Destination' : h.level}
            {h.in_network === false && ' · outside the simulated map'}
            {h.coord_source?.startsWith('unverified') && <span className="text-warning"> · location unverified</span>}
          </span>
        </div>
        <span className="flex shrink-0 flex-col items-end">
          {d === undefined ? (
            <span className="text-caption text-text-3">No data</span>
          ) : (
            <span className="num text-body-sm text-text-1">{formatDistance(d)}</span>
          )}
          <span className="text-caption text-text-3">estimate</span>
        </span>
      </li>
    );
  };

  return (
    <Panel raised title="Hospitals" actions={about}>
      <ul aria-label="Hospitals by distance beyond the simulated map">{rows.map(row)}</ul>
      <div className="mt-2 flex flex-col gap-1 border-t border-border-subtle pt-2 text-caption text-text-3">
        <span>
          Straight-line distance from each hospital's network exit to the hospital: the part of the trip that is not
          simulated. An estimate, never added to the simulated time. The destination is the hospital with the shortest
          unsimulated distance unless one was requested.
        </span>
        <span>Load: no data from the system.</span>
      </div>
    </Panel>
  );
}
