import { forwardRef } from 'react';
import { Bell, CircleHelp } from 'lucide-react';
import { Tooltip } from '../components/ui';
import type { ScreenDef } from './screens';

interface HeaderProps {
  screen: ScreenDef;
  unread: number;
  drawerOpen: boolean;
  onToggleDrawer: () => void;
  onOpenHelp: () => void;
}

const iconButton =
  'relative grid h-8 w-8 place-items-center rounded text-text-2 hover:bg-raised hover:text-text-1 aria-expanded:bg-raised aria-expanded:text-text-1';

export const Header = forwardRef<HTMLButtonElement, HeaderProps>(function Header(
  { screen, unread, drawerOpen, onToggleDrawer, onOpenHelp },
  bellRef,
) {
  const bellLabel = unread > 0 ? `Notifications, ${unread} unread` : 'Notifications';
  return (
    <header className="flex h-header items-center gap-4 border-b border-border bg-panel px-4 [grid-area:header]">
      <span className="text-body font-semibold text-text-1">Delhi EMS Routing</span>
      <span className="h-5 w-px bg-border" aria-hidden />
      <div className="flex min-w-0 items-baseline gap-2">
        <h1 className="truncate text-title text-text-1">{screen.title}</h1>
        {screen.secondary && (
          <span className="hidden truncate text-body-sm text-text-3 desktop:inline">{screen.secondary}</span>
        )}
      </div>
      <div className="ml-auto flex items-center gap-2">
        <span className="hidden text-caption text-text-3 desktop:inline">Simulated area: Connaught Place, New Delhi</span>
        <Tooltip content={bellLabel} side="bottom">
          <button
            ref={bellRef}
            type="button"
            aria-label={bellLabel}
            aria-expanded={drawerOpen}
            aria-controls="notification-drawer"
            onClick={onToggleDrawer}
            className={iconButton}
          >
            <Bell size={20} strokeWidth={1.75} aria-hidden />
            {unread > 0 && (
              <span className="num absolute -right-0.5 -top-0.5 min-w-4 rounded-sm bg-accent px-1 text-center text-caption font-semibold text-accent-on">
                {unread > 9 ? '9+' : unread}
              </span>
            )}
          </button>
        </Tooltip>
        <Tooltip content="Help and glossary" side="bottom">
          <button type="button" aria-label="Help and glossary" onClick={onOpenHelp} className={iconButton}>
            <CircleHelp size={20} strokeWidth={1.75} aria-hidden />
          </button>
        </Tooltip>
      </div>
    </header>
  );
});
