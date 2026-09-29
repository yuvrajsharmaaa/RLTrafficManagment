import { createContext, useContext } from 'react';
import type { Tone } from '../components/ui';

export interface AppNotification {
  id: number;
  /** Date.now() when raised. */
  at: number;
  tone: Tone;
  message: string;
  read: boolean;
}

export interface NotificationsApi {
  items: AppNotification[];
  unread: number;
  push: (tone: Tone, message: string) => void;
  markAllRead: () => void;
}

export const NotificationsContext = createContext<NotificationsApi | null>(null);

export function useNotifications(): NotificationsApi {
  const ctx = useContext(NotificationsContext);
  if (!ctx) throw new Error('useNotifications must be used inside NotificationsProvider');
  return ctx;
}
