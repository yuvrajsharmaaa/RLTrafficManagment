import * as RadixTooltip from '@radix-ui/react-tooltip';
import type { ReactNode } from 'react';

// Opens on hover after 300 ms and immediately on keyboard focus; Esc closes.
export const TooltipProvider = ({ children }: { children: ReactNode }) => (
  <RadixTooltip.Provider delayDuration={300}>{children}</RadixTooltip.Provider>
);

interface TooltipProps {
  content: ReactNode;
  /** A single focusable element. */
  children: ReactNode;
  side?: 'top' | 'right' | 'bottom' | 'left';
}

export function Tooltip({ content, children, side = 'top' }: TooltipProps) {
  return (
    <RadixTooltip.Root>
      <RadixTooltip.Trigger asChild>{children}</RadixTooltip.Trigger>
      <RadixTooltip.Portal>
        <RadixTooltip.Content
          side={side}
          sideOffset={6}
          collisionPadding={8}
          className="z-popover max-w-[280px] rounded border border-border-strong bg-raised px-2 py-1.5 text-caption text-text-1 shadow-2"
        >
          {content}
        </RadixTooltip.Content>
      </RadixTooltip.Portal>
    </RadixTooltip.Root>
  );
}
