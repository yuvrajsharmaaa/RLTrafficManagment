import { useId, useState, type ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';
import { cx } from '../../lib/cx';

/** Every chart has a data table alternative behind this toggle (Phase 6 accessibility rule). */
export function DataTableToggle({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((v) => !v)}
        className="flex w-fit items-center gap-1 rounded text-body-sm text-text-2 hover:text-text-1"
      >
        <ChevronDown size={14} strokeWidth={1.75} aria-hidden className={cx('transition-transform duration-base', !open && '-rotate-90')} />
        {open ? 'Hide data table' : 'Show data table'}
      </button>
      {open && (
        <div id={id} className="max-h-64 overflow-y-auto">
          {children}
        </div>
      )}
    </>
  );
}
