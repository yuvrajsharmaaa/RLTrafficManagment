import type { ReactNode } from 'react';
import { OctagonX, RotateCcw } from 'lucide-react';
import { cx } from '../../lib/cx';
import { Button } from './Button';

interface ErrorStateProps {
  /** What failed, e.g. "Route request failed". */
  title: string;
  /** What it means and what still works, e.g. "Recorded runs still work." */
  message?: string;
  onRetry?: () => void;
  /** Optional second action, e.g. "Open recorded incident". */
  secondaryAction?: ReactNode;
  className?: string;
}

/** Inline danger alert. Replaces alert() dialogs; never auto-dismisses. */
export function ErrorState({ title, message, onRetry, secondaryAction, className }: ErrorStateProps) {
  return (
    <div
      role="alert"
      className={cx('flex gap-3 rounded border border-danger border-l-2 bg-danger-weak p-3', className)}
    >
      <OctagonX size={16} strokeWidth={1.75} className="mt-0.5 shrink-0 text-danger" aria-hidden />
      <div className="flex min-w-0 flex-col gap-1">
        <p className="text-body font-semibold text-text-1">{title}</p>
        {message && <p className="text-body-sm text-text-1">{message}</p>}
        {(onRetry || secondaryAction) && (
          <div className="flex flex-wrap gap-2 pt-1">
            {onRetry && (
              <Button size="compact" icon={RotateCcw} onClick={onRetry}>
                Retry
              </Button>
            )}
            {secondaryAction}
          </div>
        )}
      </div>
    </div>
  );
}
