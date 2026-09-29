import { useCallback, useMemo, useState } from 'react';
import { AppShell } from './app/AppShell';
import { NotificationsProvider } from './app/NotificationsProvider';
import { screenDef } from './app/screens';
import { SessionContext, type SessionApi, type SessionRun } from './app/sessionRun';
import { IDLE_STATUS, ShellStatusContext, type ShellStatus } from './app/shellStatus';
import { useScreen } from './app/useScreen';
import { TooltipProvider } from './components/ui';
import { Gallery } from './dev/Gallery';
import { MissionControl } from './screens/MissionControl';
import { NotBuiltYet } from './screens/NotBuiltYet';
import { Analytics } from './screens/Analytics';
import { Benchmark } from './screens/Benchmark';
import { Optimization } from './screens/Optimization';

const showGallery = import.meta.env.DEV && new URLSearchParams(window.location.search).has('gallery');

function sameStatus(a: ShellStatus, b: ShellStatus): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

export function App() {
  const [screen, setScreen] = useScreen();
  // Status is stored with the screen that reported it, so switching screens
  // shows an idle bar until the new screen reports.
  const [entry, setEntry] = useState<{ screen: string; status: ShellStatus }>({ screen, status: IDLE_STATUS });
  const report = useCallback(
    (next: ShellStatus) =>
      setEntry((prev) => (prev.screen === screen && sameStatus(prev.status, next) ? prev : { screen, status: next })),
    [screen],
  );
  const status = entry.screen === screen ? entry.status : IDLE_STATUS;

  const [current, setCurrent] = useState<SessionRun | null>(null);
  const session = useMemo<SessionApi>(() => ({ current, setCurrent, navigate: setScreen }), [current, setScreen]);

  if (showGallery) {
    return (
      <TooltipProvider>
        <Gallery />
      </TooltipProvider>
    );
  }

  return (
    <TooltipProvider>
      <NotificationsProvider>
        <SessionContext.Provider value={session}>
          <ShellStatusContext.Provider value={report}>
            <AppShell screen={screen} onNavigate={setScreen} {...status}>
              {screen === 'mission' ? (
                <MissionControl />
              ) : screen === 'optimization' ? (
                <Optimization />
              ) : screen === 'analytics' ? (
                <Analytics />
              ) : screen === 'benchmark' ? (
                <Benchmark />
              ) : (
                <NotBuiltYet screen={screenDef(screen)} />
              )}
            </AppShell>
          </ShellStatusContext.Provider>
        </SessionContext.Provider>
      </NotificationsProvider>
    </TooltipProvider>
  );
}
