import type { ReactNode } from 'react';
import { cx } from '../../lib/cx';

type Size = 'xl' | 'lg' | 'md';

interface StatBlockProps {
  label: string;
  /** Pre-formatted value. Pass null when the data has no value: it renders "No data", never 0. */
  value: string | null;
  unit?: string;
  size?: Size;
  /** One line under the value, in words, e.g. "3.4 s sooner". */
  delta?: ReactNode;
  /** Marks the value as out of date, e.g. "as of T+55 s". */
  staleNote?: string;
  className?: string;
}

const valueSize: Record<Size, string> = {
  xl: 'text-metric-xl',
  lg: 'text-metric-lg',
  md: 'text-metric-md',
};

export function StatBlock({ label, value, unit, size = 'md', delta, staleNote, className }: StatBlockProps) {
  const stale = Boolean(staleNote);
  return (
    <div className={cx('flex flex-col gap-1', className)}>
      <span className="text-label text-text-2">{label}</span>
      {value === null ? (
        <span className="text-body text-text-3">No data</span>
      ) : (
        <span className="flex items-baseline gap-1">
          <span className={cx('num', valueSize[size], stale ? 'text-text-2' : 'text-text-1')}>{value}</span>
          {unit && <span className="text-caption text-text-3">{unit}</span>}
        </span>
      )}
      {delta && <span className="text-body-sm text-text-2">{delta}</span>}
      {stale && <span className="text-caption text-text-3">{staleNote}</span>}
    </div>
  );
}
