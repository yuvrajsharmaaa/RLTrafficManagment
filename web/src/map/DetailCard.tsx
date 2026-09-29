import { useEffect, useRef, type ReactNode } from 'react';
import { X } from 'lucide-react';
import { TIER_WORD, cleanLabel, formatClock, formatCoord, formatDuration } from '../lib/format';
import type { Hospital } from '../lib/types';
import type { Selection } from './engine';

interface DetailCardProps {
  selection: Selection;
  onClose: () => void;
  /** Live ambulance facts from the playback clock. */
  ambulance: { remaining: number; progressPct: number; arrived: boolean };
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="text-text-2">{label}</span>
      <span className="text-right text-text-1">{children}</span>
    </div>
  );
}

function roadTime(h: Hospital, candidates: Hospital[]): ReactNode {
  const t = h.live_travel_time_sec;
  if (t === undefined) return <span className="text-text-3">No data</span>;
  const tied = candidates.filter((c) => c.name !== h.name && c.live_travel_time_sec === t).map((c) => c.name.split(',')[0]);
  return (
    <span className="flex flex-col items-end">
      <span className="num">{t.toFixed(1)} s</span>
      {tied.length > 0 && <span className="text-caption text-text-3">tied with {tied.join(', ')}</span>}
      {t === 45 && <span className="text-caption text-warning">may be unreachable</span>}
    </span>
  );
}

function content(sel: Selection, amb: DetailCardProps['ambulance']): { title: string; body: ReactNode } {
  switch (sel.kind) {
    case 'ambulance':
      return {
        title: 'Ambulance',
        body: (
          <>
            <Fact label="State">{amb.arrived ? 'Arrived' : 'En route'}</Fact>
            <Fact label="Arrival in"><span className="num">{formatClock(amb.remaining)}</span></Fact>
            <Fact label="Progress"><span className="num">{Math.round(amb.progressPct)}%</span></Fact>
          </>
        ),
      };
    case 'incident':
      return {
        title: sel.stop ? 'Incident and patient pickup' : 'Incident location',
        body: (
          <>
            {sel.stop && <Fact label="Junction"><span className="num">{sel.stop.id}</span></Fact>}
            <Fact label="Position"><span className="num">{formatCoord(sel.lat, sel.lon)}</span></Fact>
            {!sel.stop && <Fact label="Status">Not dispatched</Fact>}
          </>
        ),
      };
    case 'destination':
    case 'hospital':
      return {
        title: sel.hospital.name.split(',')[0] ?? sel.hospital.name,
        body: (
          <>
            <Fact label="Role">{sel.kind === 'destination' ? 'Destination' : 'Alternate'}</Fact>
            <Fact label="Level">{sel.hospital.level}</Fact>
            <Fact label="Road time (shortest path)">{roadTime(sel.hospital, sel.candidates)}</Fact>
            <Fact label="Load"><span className="text-text-3">No data</span></Fact>
          </>
        ),
      };
    case 'checkpoint':
      return {
        title: `Corridor checkpoint ${sel.index}`,
        body: (
          <>
            <Fact label="Label">{cleanLabel(sel.stop.label)}</Fact>
            <Fact label="Junction"><span className="num">{sel.stop.id}</span></Fact>
          </>
        ),
      };
    case 'replan': {
      const e = sel.event;
      const hasEta = typeof e.eta_before === 'number' && typeof e.eta_after === 'number';
      return {
        title: `Route re-planned at T+${Math.round(e.t)} s`,
        body: (
          <>
            <p className="text-body-sm text-text-1">{e.detail}</p>
            {hasEta && (
              <Fact label="Arrival estimate">
                <span className="num">{formatClock(e.eta_before ?? 0)} to {formatClock(e.eta_after ?? 0)}</span>
              </Fact>
            )}
          </>
        ),
      };
    }
    case 'segment':
      return {
        title: `Traffic ${TIER_WORD[sel.tier].toLowerCase()}`,
        body: (
          <>
            <Fact label="Passed"><span className="num">T+{Math.round(sel.t0)} to T+{Math.round(sel.t1)} s</span></Fact>
            <Fact label="Unpredictability"><span className="num">{sel.v.toFixed(2)}</span></Fact>
            <p className="text-caption text-text-3">Network-wide level at that time, not a measurement of this road.</p>
          </>
        ),
      };
    case 'baseline':
      return {
        title: 'Fixed-schedule route',
        body: <Fact label="Total time to hospital"><span className="num">{formatDuration(sel.completion)}</span></Fact>,
      };
  }
}

export function DetailCard({ selection, onClose, ambulance }: DetailCardProps) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const returnFocus = useRef<Element | null>(null);

  useEffect(() => {
    returnFocus.current = document.activeElement;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      if (returnFocus.current instanceof HTMLElement) returnFocus.current.focus();
    };
  }, [onClose]);

  const { title, body } = content(selection, ambulance);
  return (
    <section
      aria-label={title}
      className="pointer-events-auto w-[280px] rounded-lg border border-border-strong bg-panel p-3 shadow-2"
    >
      <div className="flex items-start justify-between gap-2 pb-2">
        <h3 className="text-body font-semibold text-text-1">{title}</h3>
        <button
          ref={closeRef}
          type="button"
          aria-label="Close details"
          onClick={onClose}
          className="grid h-6 w-6 shrink-0 place-items-center rounded text-text-2 hover:bg-raised hover:text-text-1"
        >
          <X size={14} strokeWidth={1.75} aria-hidden />
        </button>
      </div>
      <div className="flex flex-col gap-1.5 text-body-sm">{body}</div>
    </section>
  );
}
