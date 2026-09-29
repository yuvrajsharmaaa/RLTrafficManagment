import { useMemo } from 'react';
import { CircleCheck, Navigation, Route, type LucideIcon } from 'lucide-react';
import { buildTimeline, etaChangeWords, type TimelineEntry } from '../../lib/explain';
import { cx } from '../../lib/cx';
import type { RunData } from '../../lib/types';

const ICON: Record<TimelineEntry['kind'], LucideIcon> = {
  dispatch: Navigation,
  replan: Route,
  arrival: CircleCheck,
};

interface MissionTimelineProps {
  run: RunData;
  t: number;
  onSeek: (t: number) => void;
}

export function MissionTimeline({ run, t, onSeek }: MissionTimelineProps) {
  const entries = useMemo(() => buildTimeline(run), [run]);
  // The current entry is the latest one at or before the playback time.
  const currentIndex = entries.reduce((acc, e, i) => (e.t <= t ? i : acc), -1);

  return (
    <ol aria-label="Mission timeline" className="flex flex-col">
      {entries.map((e, i) => {
        const Icon = ICON[e.kind];
        const future = i > currentIndex;
        const current = i === currentIndex;
        return (
          <li key={`${e.kind}-${e.t}`}>
            <button
              type="button"
              onClick={() => onSeek(e.t)}
              aria-current={current ? 'step' : undefined}
              className={cx(
                'flex min-h-11 w-full gap-3 rounded px-2 py-2 text-left hover:bg-raised',
                current && 'bg-accent-weak shadow-[inset_2px_0_0_var(--color-accent)]',
              )}
            >
              <span className={cx('num w-14 shrink-0 text-body-sm', future ? 'text-text-3' : 'text-text-2')}>
                T+{Math.round(e.t)} s
              </span>
              <Icon
                size={14}
                strokeWidth={1.75}
                aria-hidden
                className={cx('mt-0.5 shrink-0', future ? 'text-text-3' : e.kind === 'replan' ? 'text-vehicle-rerouting' : e.kind === 'arrival' ? 'text-success' : 'text-accent')}
              />
              <span className="flex min-w-0 flex-col gap-0.5">
                <span className={cx('text-body-sm', future ? 'text-text-3' : 'text-text-1')}>{e.title}</span>
                {e.eta && <span className={cx('text-caption', future ? 'text-text-3' : 'text-text-2')}>{etaChangeWords(e.eta)}</span>}
                {e.serverNote && <span className="text-caption text-text-3">Server note: {e.serverNote}</span>}
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}
