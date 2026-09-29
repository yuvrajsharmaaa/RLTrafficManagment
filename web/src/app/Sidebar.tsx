import { useRef, type KeyboardEvent } from 'react';
import { cx } from '../lib/cx';
import { SCREENS, type ScreenDef, type ScreenId } from './screens';

interface SidebarProps {
  current: ScreenId;
  onNavigate: (id: ScreenId) => void;
}

const MAIN = SCREENS.filter((s) => s.id !== 'settings');
const PINNED = SCREENS.filter((s) => s.id === 'settings');

export function Sidebar({ current, onNavigate }: SidebarProps) {
  const listRef = useRef<HTMLElement>(null);

  // Up and Down arrows move focus between items; Enter or Space opens.
  const onKeyDown = (e: KeyboardEvent<HTMLElement>) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    const items = Array.from(listRef.current?.querySelectorAll<HTMLButtonElement>('button[data-nav]') ?? []);
    const index = items.indexOf(document.activeElement as HTMLButtonElement);
    if (index === -1) return;
    e.preventDefault();
    const next = items.at((index + (e.key === 'ArrowDown' ? 1 : -1)) % items.length);
    next?.focus();
  };

  const item = (s: ScreenDef) => {
    const selected = s.id === current;
    const Icon = s.icon;
    return (
      <li key={s.id}>
        <button
          type="button"
          data-nav
          aria-current={selected ? 'page' : undefined}
          onClick={() => onNavigate(s.id)}
          className={cx(
            'flex w-full flex-col items-center gap-1 py-2 text-caption transition-colors duration-fast',
            selected
              ? 'bg-accent-weak text-text-1 shadow-[inset_2px_0_0_var(--color-accent)]'
              : 'text-text-2 hover:bg-raised hover:text-text-1',
          )}
        >
          <Icon size={20} strokeWidth={1.75} aria-hidden />
          {s.nav}
        </button>
      </li>
    );
  };

  return (
    <nav
      ref={listRef}
      aria-label="Screens"
      onKeyDown={onKeyDown}
      className="flex w-nav flex-col justify-between overflow-y-auto border-r border-border bg-panel [grid-area:nav]"
    >
      <ul className="flex flex-col">{MAIN.map(item)}</ul>
      <ul className="flex flex-col border-t border-border-subtle">{PINNED.map(item)}</ul>
    </nav>
  );
}
