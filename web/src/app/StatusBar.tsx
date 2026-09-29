import type { ReactNode } from 'react';
import { CircleCheck, Clock, LoaderCircle, OctagonX, WifiOff, type LucideIcon } from 'lucide-react';
import { cx } from '../lib/cx';
import type { Connection } from './useConnection';
import { useIstClock } from './useClock';

export type SystemState = 'Ready' | 'Computing route' | 'Playing' | 'Paused' | 'Error';

export interface PlaybackClock {
  t: number;
  total: number;
  playing: boolean;
  speed: number;
}

interface StatusBarProps {
  connection: Connection;
  systemState: SystemState;
  /** "Recorded run", "Live simulation snapshot" or "Estimated traffic". Hidden when absent. */
  dataSource?: string;
  traffic?: { word: string; v: number };
  clock?: PlaybackClock;
}

const connectionView: Record<Connection['state'], { icon: LucideIcon; label: string; className: string }> = {
  checking: { icon: LoaderCircle, label: 'Checking server', className: 'text-text-2' },
  connected: { icon: CircleCheck, label: 'Connected', className: 'text-success' },
  'server-unreachable': { icon: OctagonX, label: 'Server not reachable', className: 'text-danger' },
  offline: { icon: WifiOff, label: 'Offline', className: 'text-danger' },
};

function Item({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span className={cx('flex h-8 items-center gap-1.5 whitespace-nowrap border-r border-border-subtle pr-3', className)}>
      {children}
    </span>
  );
}

// Items with nothing to report are hidden, never shown as a dash; the order is fixed.
export function StatusBar({ connection, systemState, dataSource, traffic, clock }: StatusBarProps) {
  const ist = useIstClock();
  const conn = connectionView[connection.state];
  const ConnIcon = conn.icon;

  return (
    <footer
      aria-label="System status"
      className="flex min-h-statusbar flex-wrap items-center gap-x-3 border-t border-border bg-panel px-4 text-caption text-text-2 [grid-area:status]"
    >
      <Item className={conn.className}>
        <ConnIcon size={12} strokeWidth={1.75} aria-hidden />
        <span>{conn.label}</span>
      </Item>
      <Item>
        <span className="text-text-3">System</span>
        <span className="text-text-1">{systemState}</span>
      </Item>
      {dataSource && (
        <Item>
          <span className="text-text-3">Data</span>
          <span className="text-text-1">{dataSource}</span>
        </Item>
      )}
      {traffic && (
        <Item>
          <span className="text-text-3">Traffic</span>
          <span className="text-text-1">{traffic.word}</span>
          <span className="num">{traffic.v.toFixed(2)}</span>
        </Item>
      )}
      {clock && (
        <Item>
          <span className="num text-text-1">
            T+{Math.floor(clock.t)} s / {Math.round(clock.total)} s
          </span>
          <span>{clock.playing ? 'Playing' : 'Paused'}</span>
          <span className="num">{clock.speed}x</span>
        </Item>
      )}
      <Item className="desktop:hidden">Simulated area: Connaught Place</Item>
      <span className="ml-auto flex h-8 items-center gap-1.5">
        <Clock size={12} strokeWidth={1.75} aria-hidden />
        <span className="num">{ist} IST</span>
      </span>
    </footer>
  );
}
