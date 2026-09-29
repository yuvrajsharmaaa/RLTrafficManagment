import { forwardRef, type ButtonHTMLAttributes } from 'react';
import { LoaderCircle, type LucideIcon } from 'lucide-react';
import { cx } from '../../lib/cx';

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger';
type Size = 'compact' | 'default' | 'large';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  icon?: LucideIcon;
  /** Shows a spinner in place of the icon and swaps the label, e.g. "Finding route". */
  loading?: boolean;
  loadingLabel?: string;
}

const variants: Record<Variant, string> = {
  primary: 'bg-accent text-accent-on border-accent hover:brightness-110',
  secondary: 'bg-raised text-text-1 border-border hover:border-border-strong',
  ghost: 'bg-transparent text-text-2 border-transparent hover:bg-raised hover:text-text-1',
  danger: 'bg-transparent text-danger border-danger hover:bg-danger-weak',
};

const sizes: Record<Size, string> = {
  compact: 'h-7 px-3 text-body-sm',
  default: 'h-8 px-3 text-body',
  large: 'h-10 px-4 text-body font-semibold',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'secondary', size = 'default', icon: Icon, loading = false, loadingLabel, className, children, disabled, type = 'button', onClick, ...rest },
  ref,
) {
  // While loading the button keeps its variant styling but ignores clicks;
  // only a truly disabled button gets the disabled look.
  return (
    <button
      ref={ref}
      type={type}
      disabled={disabled}
      aria-disabled={loading || undefined}
      aria-busy={loading || undefined}
      onClick={loading ? undefined : onClick}
      className={cx(
        'inline-flex select-none items-center justify-center gap-2 whitespace-nowrap rounded border font-sans font-medium',
        'transition-[background-color,border-color,filter] duration-fast ease-standard',
        'disabled:cursor-not-allowed disabled:border-border disabled:bg-transparent disabled:text-text-disabled disabled:hover:brightness-100',
        'aria-busy:cursor-progress',
        variants[variant],
        sizes[size],
        className,
      )}
      {...rest}
    >
      {loading ? (
        <LoaderCircle size={16} strokeWidth={1.75} className="animate-spin" aria-hidden />
      ) : (
        Icon && <Icon size={16} strokeWidth={1.75} aria-hidden />
      )}
      {loading && loadingLabel ? loadingLabel : children}
    </button>
  );
});
