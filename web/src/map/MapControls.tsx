import { LocateFixed, Minus, Moon, Plus, Scan, Sun } from 'lucide-react';
import { Tooltip } from '../components/ui';

interface MapControlsProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFit: () => void;
  dark: boolean;
  onToggleBase: () => void;
  /** False once the user has moved the map; shows "Recenter and follow". */
  following: boolean;
  onRecenter: () => void;
}

const btn =
  'grid h-8 w-8 place-items-center border-b border-border-subtle text-text-2 last:border-b-0 hover:bg-raised hover:text-text-1';

/** Top-right map controls (Phase 2): 32 px buttons, lucide icons, tooltips, all keyboard reachable. */
export function MapControls({ onZoomIn, onZoomOut, onFit, dark, onToggleBase, following, onRecenter }: MapControlsProps) {
  return (
    <div className="pointer-events-none absolute right-3 top-3 z-map-overlay flex flex-col items-end gap-2">
      <div className="pointer-events-auto flex flex-col overflow-hidden rounded border border-border bg-panel shadow-1">
        <Tooltip content="Zoom in" side="left">
          <button type="button" aria-label="Zoom in" className={btn} onClick={onZoomIn}>
            <Plus size={20} strokeWidth={1.75} aria-hidden />
          </button>
        </Tooltip>
        <Tooltip content="Zoom out" side="left">
          <button type="button" aria-label="Zoom out" className={btn} onClick={onZoomOut}>
            <Minus size={20} strokeWidth={1.75} aria-hidden />
          </button>
        </Tooltip>
        <Tooltip content="Fit route" side="left">
          <button type="button" aria-label="Fit route" className={btn} onClick={onFit}>
            <Scan size={20} strokeWidth={1.75} aria-hidden />
          </button>
        </Tooltip>
        <Tooltip content={dark ? 'Base map: light' : 'Base map: dark'} side="left">
          <button type="button" aria-label={dark ? 'Switch to light base map' : 'Switch to dark base map'} className={btn} onClick={onToggleBase}>
            {dark ? <Sun size={20} strokeWidth={1.75} aria-hidden /> : <Moon size={20} strokeWidth={1.75} aria-hidden />}
          </button>
        </Tooltip>
      </div>
      {!following && (
        <button
          type="button"
          onClick={onRecenter}
          className="pointer-events-auto flex h-8 items-center gap-2 rounded border border-accent bg-panel px-3 text-body-sm text-text-1 shadow-1 hover:bg-raised"
        >
          <LocateFixed size={16} strokeWidth={1.75} aria-hidden />
          Recenter and follow
        </button>
      )}
    </div>
  );
}
