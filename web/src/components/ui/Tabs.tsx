import * as RadixTabs from '@radix-ui/react-tabs';
import type { ComponentPropsWithoutRef } from 'react';
import { cx } from '../../lib/cx';

// Radix supplies roving focus, arrow-key navigation and the tablist roles.

export const Tabs = RadixTabs.Root;

export function TabsList({ className, ...props }: ComponentPropsWithoutRef<typeof RadixTabs.List>) {
  return <RadixTabs.List className={cx('flex h-8 gap-4 border-b border-border-subtle', className)} {...props} />;
}

export function TabsTrigger({ className, ...props }: ComponentPropsWithoutRef<typeof RadixTabs.Trigger>) {
  return (
    <RadixTabs.Trigger
      className={cx(
        '-mb-px border-b-2 border-transparent text-body text-text-2 transition-colors duration-fast',
        'hover:text-text-1 data-[state=active]:border-accent data-[state=active]:text-text-1',
        className,
      )}
      {...props}
    />
  );
}

export function TabsContent({ className, ...props }: ComponentPropsWithoutRef<typeof RadixTabs.Content>) {
  return <RadixTabs.Content className={cx('pt-3', className)} {...props} />;
}
