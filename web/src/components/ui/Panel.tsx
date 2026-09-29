import { useId, useState, type ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';
import { cx } from '../../lib/cx';

interface PanelProps {
  title?: string;
  /** Right-aligned control in the title row. */
  actions?: ReactNode;
  /** Makes the body collapsible behind the title ("Details" pattern). */
  collapsible?: boolean;
  defaultOpen?: boolean;
  /** Raised surface for a card inside a panel. */
  raised?: boolean;
  className?: string;
  children: ReactNode;
}

export function Panel({ title, actions, collapsible = false, defaultOpen = true, raised = false, className, children }: PanelProps) {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = useId();
  const showBody = !collapsible || open;

  return (
    <section className={cx('rounded border border-border', raised ? 'bg-raised' : 'bg-panel', className)}>
      {title && (
        <header className="flex min-h-10 items-center justify-between gap-2 px-4 pt-3">
          {collapsible ? (
            <button
              type="button"
              aria-expanded={open}
              aria-controls={bodyId}
              onClick={() => setOpen((o) => !o)}
              className="-mx-1 flex items-center gap-1 rounded px-1 text-title text-text-1"
            >
              <ChevronDown
                size={16}
                strokeWidth={1.75}
                aria-hidden
                className={cx('transition-transform duration-base', !open && '-rotate-90')}
              />
              {title}
            </button>
          ) : (
            <h2 className="text-title text-text-1">{title}</h2>
          )}
          {actions}
        </header>
      )}
      {showBody && (
        <div id={bodyId} className={cx('px-4 pb-4', title ? 'pt-2' : 'pt-4')}>
          {children}
        </div>
      )}
    </section>
  );
}
