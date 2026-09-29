import type { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';
import { cx } from '../../lib/cx';

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  message?: string;
  /** One action, usually a Button. */
  action?: ReactNode;
  className?: string;
}

export function EmptyState({ icon: Icon, title, message, action, className }: EmptyStateProps) {
  return (
    <div className={cx('flex flex-col items-start gap-2', className)}>
      <Icon size={24} strokeWidth={1.75} className="text-text-3" aria-hidden />
      <p className="text-title text-text-1">{title}</p>
      {message && <p className="text-body text-text-2">{message}</p>}
      {action && <div className="pt-1">{action}</div>}
    </div>
  );
}
