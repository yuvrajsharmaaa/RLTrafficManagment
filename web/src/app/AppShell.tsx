import { useEffect, useRef, useState, type ReactNode } from 'react';
import { ConnectionContext } from './connection';
import { Header } from './Header';
import { HelpDialog } from './HelpDialog';
import { NotificationDrawer } from './NotificationDrawer';
import { useNotifications } from './notifications';
import { screenDef, type ScreenId } from './screens';
import { Sidebar } from './Sidebar';
import { StatusBar, type PlaybackClock, type SystemState } from './StatusBar';
import { useConnection, type ConnectionState } from './useConnection';

const HELP_SEEN_KEY = 'ems.helpSeen';

function helpSeen(): boolean {
  try {
    return window.localStorage.getItem(HELP_SEEN_KEY) === '1';
  } catch {
    return false;
  }
}

function rememberHelpSeen() {
  try {
    window.localStorage.setItem(HELP_SEEN_KEY, '1');
  } catch {
    // Storage unavailable (private window): the guide shows again next visit.
  }
}

interface AppShellProps {
  screen: ScreenId;
  onNavigate: (id: ScreenId) => void;
  systemState: SystemState;
  dataSource?: string;
  traffic?: { word: string; v: number };
  clock?: PlaybackClock;
  children: ReactNode;
}

export function AppShell({ screen, onNavigate, systemState, dataSource, traffic, clock, children }: AppShellProps) {
  const connection = useConnection();
  const { unread, push, markAllRead } = useNotifications();
  const [drawerOpen, setDrawerOpen] = useState(false);
  // First visit only; it never starts playback.
  const [helpOpen, setHelpOpen] = useState(() => !helpSeen());
  const bellRef = useRef<HTMLButtonElement>(null);

  // Raise a notification when the connection changes after the first check.
  const lastState = useRef<ConnectionState>('checking');
  useEffect(() => {
    const prev = lastState.current;
    const next = connection.state;
    lastState.current = next;
    if (prev === next || next === 'checking') return;
    if (next === 'offline') push('danger', 'Offline. Recorded runs remain available from the cache.');
    else if (next === 'server-unreachable') push('danger', 'Server not reachable. Live dispatch is unavailable.');
    else if (prev !== 'checking') push('success', 'Connection restored.');
  }, [connection.state, push]);

  const toggleDrawer = (open: boolean) => {
    setDrawerOpen(open);
    if (open) markAllRead();
    else bellRef.current?.focus();
  };

  const onHelpChange = (open: boolean) => {
    setHelpOpen(open);
    if (!open) rememberHelpSeen();
  };

  return (
    <div className="grid h-full grid-cols-[var(--shell-nav-w)_1fr] grid-rows-[var(--shell-header-h)_1fr_auto] [grid-template-areas:'header_header'_'nav_main'_'status_status']">
      <Header
        ref={bellRef}
        screen={screenDef(screen)}
        unread={unread}
        drawerOpen={drawerOpen}
        onToggleDrawer={() => toggleDrawer(!drawerOpen)}
        onOpenHelp={() => setHelpOpen(true)}
      />
      <Sidebar current={screen} onNavigate={onNavigate} />
      <main className="relative min-h-0 min-w-0 overflow-auto bg-bg [grid-area:main]">
        <ConnectionContext.Provider value={connection}>{children}</ConnectionContext.Provider>
        <NotificationDrawer open={drawerOpen} onOpenChange={toggleDrawer} />
      </main>
      <StatusBar
        connection={connection}
        systemState={systemState}
        dataSource={dataSource}
        traffic={traffic}
        clock={clock}
      />
      <HelpDialog open={helpOpen} onOpenChange={onHelpChange} />
    </div>
  );
}
