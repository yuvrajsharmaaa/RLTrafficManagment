import { useEffect, useId } from 'react';
import { Pause, Play, RotateCcw } from 'lucide-react';
import { Tooltip } from '../ui';
import type { Playback } from '../../hooks/usePlayback';

const SPEEDS = [0.5, 1, 2, 4, 5];
const SEEK_STEP_S = 5;

interface PlaybackDockProps {
  playback: Playback;
  /** Times of re-plan events, drawn as markers on the scrubber. */
  markers: number[];
}

function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  // Map arrow keys pan the map; form fields and buttons keep their own keys.
  return Boolean(target.closest('input, select, textarea, button, [contenteditable], .leaflet-container, [role="tab"]'));
}

export function PlaybackDock({ playback, markers }: PlaybackDockProps) {
  const speedId = useId();
  const { t, duration, playing, speed, toggle, seek, restart, setSpeed, getTime } = playback;

  // Space play/pause, Left/Right seek 5 s, Home restart.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || isTyping(e.target)) return;
      if (e.key === ' ') toggle();
      else if (e.key === 'ArrowLeft') seek(getTime() - SEEK_STEP_S);
      else if (e.key === 'ArrowRight') seek(getTime() + SEEK_STEP_S);
      else if (e.key === 'Home') restart();
      else return;
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [toggle, seek, restart, getTime]);

  const iconBtn = 'grid h-8 w-8 place-items-center rounded border border-border bg-raised text-text-1 hover:border-border-strong';

  return (
    <div className="flex h-dock items-center gap-3 border-t border-border bg-panel px-4 [grid-area:dock]">
      <Tooltip content={playing ? 'Pause (Space)' : 'Play (Space)'}>
        <button type="button" aria-label={playing ? 'Pause' : 'Play'} onClick={toggle} className={iconBtn}>
          {playing ? <Pause size={16} strokeWidth={1.75} aria-hidden /> : <Play size={16} strokeWidth={1.75} aria-hidden />}
        </button>
      </Tooltip>
      <Tooltip content="Restart (Home)">
        <button type="button" aria-label="Restart" onClick={restart} className={iconBtn}>
          <RotateCcw size={16} strokeWidth={1.75} aria-hidden />
        </button>
      </Tooltip>

      <div className="relative flex-1">
        <input
          type="range"
          min={0}
          max={duration}
          step="any"
          value={t}
          onChange={(e) => seek(Number(e.target.value))}
          aria-label="Playback position"
          aria-valuetext={`T+${Math.floor(t)} s of ${Math.round(duration)} s`}
          className="w-full accent-[var(--color-accent)]"
        />
        <div className="pointer-events-none absolute inset-x-0 -top-1.5 h-2" aria-hidden>
          {markers.map((m) => (
            <span
              key={m}
              className="absolute h-2 w-0.5 -translate-x-1/2 bg-vehicle-rerouting"
              style={{ left: `${duration > 0 ? (m / duration) * 100 : 0}%` }}
            />
          ))}
        </div>
      </div>

      <span className="num w-[120px] text-right text-body-sm text-text-1">
        T+{Math.floor(t)} s / {Math.round(duration)} s
      </span>
      <label htmlFor={speedId} className="sr-only">Playback speed</label>
      <select
        id={speedId}
        value={speed}
        onChange={(e) => setSpeed(Number(e.target.value))}
        className="num h-8 rounded border border-border-control bg-raised px-2 text-body-sm text-text-1"
      >
        {SPEEDS.map((s) => (
          <option key={s} value={s}>{s}x</option>
        ))}
      </select>
    </div>
  );
}
