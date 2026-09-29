import { useId, useState, type ReactNode } from 'react';
import { ChevronDown, Layers } from 'lucide-react';
import { cx } from '../lib/cx';
import type { LayerKey } from './engine';
import type { LayerState } from './layers';

// Swatches are drawn exactly as the map draws each layer: colour + width + dash, or the marker shape.
function Line({ color, width, dash, casing }: { color: string; width: number; dash?: string; casing?: boolean }) {
  return (
    <svg width="28" height="12" aria-hidden className="shrink-0">
      {casing && <line x1="2" y1="6" x2="26" y2="6" stroke="var(--color-bg)" strokeWidth={width + 3} />}
      <line x1="2" y1="6" x2="26" y2="6" stroke={color} strokeWidth={width} strokeDasharray={dash} />
    </svg>
  );
}

function Shape({ kind }: { kind: 'square' | 'diamond' | 'circle' | 'double' | 'small' | 'ring' | 'triangle' | 'dashbox' }) {
  const common = { fill: 'var(--color-panel)', strokeWidth: 1.5 };
  return (
    <svg width="28" height="16" viewBox="0 0 28 16" aria-hidden className="shrink-0">
      {kind === 'square' && <rect x="8" y="2" width="12" height="12" rx="2" stroke="var(--vehicle-en-route)" {...common} />}
      {kind === 'diamond' && <rect x="9" y="3" width="10" height="10" transform="rotate(45 14 8)" stroke="var(--color-warning)" {...common} />}
      {kind === 'double' && (
        <>
          <circle cx="14" cy="8" r="7" stroke="var(--route-optimized)" {...common} />
          <circle cx="14" cy="8" r="4" stroke="var(--route-optimized)" {...common} />
        </>
      )}
      {kind === 'circle' && <circle cx="14" cy="8" r="5.5" stroke="var(--hospital-unknown)" {...common} />}
      {kind === 'small' && <rect x="10" y="4" width="8" height="8" rx="1" stroke="var(--color-border-control)" {...common} />}
      {kind === 'ring' && <circle cx="14" cy="8" r="5.5" stroke="var(--vehicle-rerouting)" {...common} strokeWidth={2} />}
      {kind === 'triangle' && <path d="M14 2 L21 14 L7 14 Z" fill="var(--color-text-3)" />}
      {kind === 'dashbox' && <rect x="4" y="3" width="20" height="10" fill="none" stroke="var(--color-border-strong)" strokeDasharray="3 2" />}
    </svg>
  );
}

function Heat() {
  return (
    <svg width="28" height="12" aria-hidden className="shrink-0">
      {[1, 2, 3, 4, 5].map((i) => (
        <rect key={i} x={2 + (i - 1) * 5} y="2" width="5" height="8" fill={`var(--heat-${i})`} />
      ))}
    </svg>
  );
}

interface RowProps {
  swatch: ReactNode;
  label: string;
  note?: string;
  layer?: LayerKey;
  layers: LayerState;
  onToggle: (key: LayerKey, on: boolean) => void;
  /** Explains why a row has nothing to show. */
  unavailable?: string;
}

function Row({ swatch, label, note, layer, layers, onToggle, unavailable }: RowProps) {
  const id = useId();
  const content = (
    <>
      {swatch}
      <span className="flex min-w-0 flex-col">
        <span className={unavailable ? 'text-text-3' : 'text-text-1'}>{label}</span>
        {(unavailable ?? note) && <span className="text-caption text-text-3">{unavailable ?? note}</span>}
      </span>
    </>
  );
  if (!layer || unavailable) return <li className="flex items-start gap-2 py-1">{content}</li>;
  return (
    <li className="py-1">
      <label htmlFor={id} className="flex cursor-pointer items-start gap-2">
        <input
          id={id}
          type="checkbox"
          checked={layers[layer]}
          onChange={(e) => onToggle(layer, e.target.checked)}
          className="mt-0.5 h-3.5 w-3.5 shrink-0 accent-[var(--color-accent)]"
        />
        {content}
      </label>
    </li>
  );
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="border-t border-border-subtle pt-2 first:border-t-0 first:pt-0">
      <p className="pb-1 text-label text-text-2">{title}</p>
      <ul className="text-body-sm">{children}</ul>
    </div>
  );
}

interface LegendProps {
  layers: LayerState;
  onToggle: (key: LayerKey, on: boolean) => void;
  hasBaseline: boolean;
  isLive: boolean;
}

export function Legend({ layers, onToggle, hasBaseline, isLive }: LegendProps) {
  // Collapsed by default so the map stays the hero; one click opens it.
  const [open, setOpen] = useState(false);
  const bodyId = useId();
  const p = { layers, onToggle };

  return (
    <div className="pointer-events-auto w-[272px] rounded border border-border bg-panel shadow-1">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => setOpen((o) => !o)}
        className="flex h-9 w-full items-center gap-2 px-3 text-label text-text-1"
      >
        <Layers size={16} strokeWidth={1.75} aria-hidden />
        Legend and layers
        <ChevronDown size={16} strokeWidth={1.75} aria-hidden className={cx('ml-auto transition-transform duration-base', !open && 'rotate-180')} />
      </button>
      {open && (
        <div id={bodyId} className="flex max-h-[52vh] flex-col gap-2 overflow-y-auto border-t border-border-subtle px-3 pb-3 pt-2">
          <Group title="Routes">
            <Row {...p} layer="route" label="Adaptive route" note="Solid = driven, lighter = still ahead" swatch={<Line color="var(--route-optimized)" width={4} casing />} />
            <Row
              {...p}
              layer="baseline"
              label="Fixed-schedule route"
              note="Same recorded traffic"
              unavailable={hasBaseline ? undefined : isLive ? 'Not available for live runs' : 'Not loaded'}
              swatch={<Line color="var(--route-baseline)" width={3} dash="10 6" />}
            />
            <Row {...p} label="Alternative route" unavailable="No alternative route geometry from the server" swatch={<Line color="var(--route-alternative)" width={3} dash="2 5" />} />
            <Row {...p} layer="direction" label="Direction of travel" swatch={<svg width="28" height="12" aria-hidden><path d="M11 9 L14 4 L17 9" fill="none" stroke="var(--route-optimized)" strokeWidth="1.6" /></svg>} />
          </Group>
          <Group title="Places">
            <Row {...p} label="Ambulance" note="Border: blue en route, amber re-planning, green arrived" swatch={<Shape kind="square" />} />
            <Row {...p} label="Incident and pickup" swatch={<Shape kind="diamond" />} />
            <Row {...p} label="Destination hospital" swatch={<Shape kind="double" />} />
            <Row {...p} label="Route end" note="Hospital entry junction, with the arrival time" swatch={<Shape kind="small" />} />
            <Row {...p} layer="hospitals" label="Other hospitals" note="Load: no data from the server" swatch={<Shape kind="circle" />} />
            <Row {...p} layer="checkpoints" label="Corridor checkpoints" swatch={<Shape kind="small" />} />
            <Row {...p} layer="replans" label="Re-plan points" swatch={<Shape kind="ring" />} />
            <Row {...p} label="Traffic signals" unavailable="No signal data from the server" swatch={<Shape kind="triangle" />} />
          </Group>
          <Group title="Traffic">
            <Row
              {...p}
              layer="traffic"
              label="Traffic level when passed"
              note="Network-wide level at that time, not per road. Width grows with level."
              swatch={
                <span className="flex flex-col gap-0.5">
                  <Line color="var(--traffic-free)" width={2} />
                  <Line color="var(--traffic-moderate)" width={3} />
                  <Line color="var(--traffic-heavy)" width={4} />
                </span>
              }
            />
            <Row {...p} layer="heat" label="Unpredictability heat" note="Along the route, 0 (light fill) to 1" swatch={<Heat />} />
          </Group>
          <Group title="Area">
            <Row {...p} layer="area" label="Simulated area" note="Connaught Place network" swatch={<Shape kind="dashbox" />} />
            <Row {...p} label="Affected area" unavailable="No incident extent from the server" swatch={<Shape kind="dashbox" />} />
          </Group>
        </div>
      )}
    </div>
  );
}
