import { cx } from '../../lib/cx';

interface SkeletonProps {
  /** Number of stacked text lines to reserve. */
  lines?: number;
  className?: string;
}

/** Static placeholder in the layout's shape. No shimmer, per the Phase 2 rules. */
export function Skeleton({ lines = 1, className }: SkeletonProps) {
  return (
    <div className={cx('flex flex-col gap-2', className)} aria-hidden>
      {Array.from({ length: lines }, (_, i) => (
        <div key={i} className={cx('h-4 rounded-sm bg-raised', i === lines - 1 && lines > 1 && 'w-2/3')} />
      ))}
    </div>
  );
}
