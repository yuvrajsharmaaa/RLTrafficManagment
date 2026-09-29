import { useCallback, useMemo, useRef, useState, type ReactNode } from 'react';
import type { Tone } from '../components/ui';
import { NotificationsContext, type AppNotification, type NotificationsApi } from './notifications';

const MAX_ITEMS = 50;

// Notifications are raised by the frontend from things it observes (re-plan
// events, estimated traffic, failed requests, connection changes). There is no
// server push.
export function NotificationsProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<AppNotification[]>([]);
  const nextId = useRef(1);

  const push = useCallback((tone: Tone, message: string) => {
    const item: AppNotification = { id: nextId.current++, at: Date.now(), tone, message, read: false };
    setItems((prev) => [item, ...prev].slice(0, MAX_ITEMS));
  }, []);

  const markAllRead = useCallback(() => {
    setItems((prev) => (prev.some((n) => !n.read) ? prev.map((n) => ({ ...n, read: true })) : prev));
  }, []);

  const value = useMemo<NotificationsApi>(
    () => ({ items, unread: items.filter((n) => !n.read).length, push, markAllRead }),
    [items, push, markAllRead],
  );

  return <NotificationsContext.Provider value={value}>{children}</NotificationsContext.Provider>;
}
