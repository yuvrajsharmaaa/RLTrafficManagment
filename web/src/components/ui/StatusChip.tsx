import { CircleCheck, Info, OctagonX, TriangleAlert, type LucideIcon } from 'lucide-react';
import { cx } from '../../lib/cx';

export type Tone = 'success' | 'warning' | 'danger' | 'info' | 'neutral';

const toneClass: Record<Tone, string> = {
  success: 'border-success bg-success-weak text-success',
  warning: 'border-warning bg-warning-weak text-warning',
  danger: 'border-danger bg-danger-weak text-danger',
  info: 'border-info bg-info-weak text-info',
  neutral: 'border-border-strong bg-raised text-text-2',
};

const toneIcon: Record<Tone, LucideIcon> = {
  success: CircleCheck,
  warning: TriangleAlert,
  danger: OctagonX,
  info: Info,
  neutral: Info,
};

interface StatusChipProps {
  tone: Tone;
  label: string;
  /** Overrides the tone's default icon. Every chip has an icon and a word. */
  icon?: LucideIcon;
  className?: string;
}

export function StatusChip({ tone, label, icon, className }: StatusChipProps) {
  const Icon = icon ?? toneIcon[tone];
  return (
    <span
      className={cx(
        'inline-flex h-5 items-center gap-1 whitespace-nowrap rounded-sm border px-1.5 text-caption font-medium',
        toneClass[tone],
        className,
      )}
    >
      <Icon size={12} strokeWidth={1.75} aria-hidden />
      {label}
    </span>
  );
}
