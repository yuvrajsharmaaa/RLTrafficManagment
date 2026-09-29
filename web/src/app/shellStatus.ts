import { createContext, useContext } from 'react';
import type { PlaybackClock, SystemState } from './StatusBar';

/** What the active screen reports to the global status bar. */
export interface ShellStatus {
  systemState: SystemState;
  dataSource?: string;
  traffic?: { word: string; v: number };
  clock?: PlaybackClock;
}

export const IDLE_STATUS: ShellStatus = { systemState: 'Ready' };

export const ShellStatusContext = createContext<(status: ShellStatus) => void>(() => undefined);

export function useReportStatus(): (status: ShellStatus) => void {
  return useContext(ShellStatusContext);
}
