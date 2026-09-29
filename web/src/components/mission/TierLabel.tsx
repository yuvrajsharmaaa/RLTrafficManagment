import { TIER_WORD } from '../../lib/format';
import type { TierKey } from '../../lib/types';
import { cx } from '../../lib/cx';

const TIER_CLASS: Record<TierKey, string> = {
  calm: 'text-traffic-free',
  moderate: 'text-traffic-moderate',
  turbulent: 'text-traffic-heavy',
};

// Bars 1 to 3 give the level a shape as well as a colour (colour-blind rule).
const BARS: Record<TierKey, number> = { calm: 1, moderate: 2, turbulent: 3 };

export function TierLabel({ tier, className }: { tier: TierKey; className?: string }) {
  const n = BARS[tier];
  return (
    <span className={cx('inline-flex items-center gap-1.5 font-medium', TIER_CLASS[tier], className)}>
      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
        {[0, 1, 2].map((i) => (
          <rect key={i} x={i * 4} y={8 - i * 3} width="3" height={4 + i * 3} fill="currentColor" opacity={i < n ? 1 : 0.25} />
        ))}
      </svg>
      {TIER_WORD[tier]}
    </span>
  );
}
