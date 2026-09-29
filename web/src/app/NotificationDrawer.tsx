import * as Dialog from '@radix-ui/react-dialog';
import { Bell, CircleCheck, Info, OctagonX, TriangleAlert, X, type LucideIcon } from 'lucide-react';
import type { Tone } from '../components/ui';
import { EmptyState } from '../components/ui';
import { cx } from '../lib/cx';
import { useNotifications } from './notifications';

const toneView: Record<Tone, { icon: LucideIcon; className: string }> = {
  success: { icon: CircleCheck, className: 'text-success' },
  warning: { icon: TriangleAlert, className: 'text-warning' },
  danger: { icon: OctagonX, className: 'text-danger' },
  info: { icon: Info, className: 'text-info' },
  neutral: { icon: Info, className: 'text-text-2' },
};

const timeFmt = new Intl.DateTimeFormat('en-IN', {
  timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
});

interface NotificationDrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

// Non-modal: the map and panels stay usable while the drawer is open. Esc closes
// it and focus returns to the bell.
export function NotificationDrawer({ open, onOpenChange }: NotificationDrawerProps) {
  const { items } = useNotifications();

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange} modal={false}>
      <Dialog.Content
        id="notification-drawer"
        aria-describedby={undefined}
        className="absolute bottom-0 right-0 top-0 z-popover flex w-[360px] max-w-full flex-col border-l border-border-strong bg-panel shadow-2"
      >
        <div className="flex h-12 items-center justify-between border-b border-border-subtle px-4">
          <Dialog.Title className="text-title text-text-1">Notifications</Dialog.Title>
          <Dialog.Close
            aria-label="Close notifications"
            className="grid h-8 w-8 place-items-center rounded text-text-2 hover:bg-raised hover:text-text-1"
          >
            <X size={16} strokeWidth={1.75} aria-hidden />
          </Dialog.Close>
        </div>
        {items.length === 0 ? (
          <EmptyState
            className="p-4"
            icon={Bell}
            title="No notifications"
            message="Route changes, data-source warnings and connection changes will appear here."
          />
        ) : (
          <ol className="flex-1 overflow-y-auto">
            {items.map((n) => {
              const view = toneView[n.tone];
              const Icon = view.icon;
              return (
                <li key={n.id} className="flex gap-3 border-b border-border-subtle px-4 py-3">
                  <Icon size={16} strokeWidth={1.75} className={cx('mt-0.5 shrink-0', view.className)} aria-hidden />
                  <div className="flex min-w-0 flex-col gap-0.5">
                    <p className={cx('text-body-sm', n.read ? 'text-text-2' : 'text-text-1')}>{n.message}</p>
                    <time className="num text-caption text-text-3" dateTime={new Date(n.at).toISOString()}>
                      {timeFmt.format(n.at)} IST
                    </time>
                  </div>
                </li>
              );
            })}
          </ol>
        )}
      </Dialog.Content>
    </Dialog.Root>
  );
}
