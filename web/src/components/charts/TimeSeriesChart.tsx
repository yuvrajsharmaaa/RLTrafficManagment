import { useMemo } from 'react';
import { Table, type Column } from '../ui';
import { useElementWidth } from '../../hooks/useElementWidth';
import { DataTableToggle } from './DataTableToggle';

export interface SeriesPoint {
  t: number;
  v: number;
}

export interface Series {
  key: string;
  label: string;
  color: string;
  dash?: string;
  points: SeriesPoint[];
}

interface TimeSeriesChartProps {
  /** Accessible name, also the data table caption. */
  title: string;
  series: Series[];
  xMax: number;
  yMin: number;
  yMax: number;
  yTicks: number[];
  yLabel: string;
  formatY: (v: number) => string;
  /** Vertical event markers (re-plans). */
  markers?: Array<{ t: number; label: string }>;
  /** Horizontal reference lines. */
  references?: Array<{ v: number; label: string }>;
  /** Playback cursor. */
  cursor?: number;
  height?: number;
  /** Small card version: no legend row, fewer ticks. */
  compact?: boolean;
}

// Phase 2 chart style: 1 px axes, horizontal grid only, 12 px mono ticks,
// 2 px first series, 1.5 px others, dashed references, amber event markers.
export function TimeSeriesChart({
  title, series, xMax, yMin, yMax, yTicks, yLabel, formatY, markers = [], references = [], cursor, height = 180, compact = false,
}: TimeSeriesChartProps) {
  const [ref, W] = useElementWidth<HTMLDivElement>();
  const H = height;
  const PAD = compact ? { l: 32, r: 8, t: 10, b: 22 } : { l: 44, r: 12, t: 14, b: 28 };

  const geom = useMemo(() => {
    const span = Math.max(1e-9, xMax);
    const x = (t: number) => PAD.l + (Math.max(0, Math.min(t, xMax)) / span) * (W - PAD.l - PAD.r);
    const y = (v: number) => PAD.t + (1 - (Math.max(yMin, Math.min(yMax, v)) - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);
    const paths = series.map((s) => ({
      ...s,
      d: s.points
        .filter((p) => p.t <= xMax)
        .map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p.t).toFixed(1)} ${y(p.v).toFixed(1)}`)
        .join(' '),
    }));
    // Even time steps that fit the width at about one label per 56 px.
    const maxLabels = Math.max(2, Math.floor((W - PAD.l - PAD.r) / 56));
    const step = [5, 10, 20, 30, 60, 120].find((s) => xMax / s <= maxLabels) ?? 300;
    const xTicks = Array.from({ length: Math.floor(xMax / step) + 1 }, (_, i) => i * step);
    return { x, y, paths, xTicks };
  }, [series, xMax, yMin, yMax, H, W, PAD.l, PAD.r, PAD.t, PAD.b]);

  const rows = useMemo(() => {
    const first = series[0];
    if (!first) return [];
    return first.points
      .filter((p) => p.t <= xMax)
      .map((p) => ({ t: p.t, values: series.map((s) => s.points.find((q) => q.t === p.t)?.v) }));
  }, [series, xMax]);
  type Row = (typeof rows)[number];
  const columns: Column<Row>[] = [
    { key: 't', header: 'Time (s)', numeric: true, render: (r) => r.t.toFixed(0) },
    ...series.map((s, i) => ({
      key: s.key,
      header: s.label,
      numeric: true,
      render: (r: Row) => {
        const v = r.values[i];
        return v === undefined ? 'No data' : formatY(v);
      },
    })),
  ];

  const { x, y } = geom;
  return (
    <figure className="flex flex-col gap-2">
      {!compact && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-caption text-text-2">
          {series.map((s) => (
            <span key={s.key} className="flex items-center gap-1.5">
              <svg width="20" height="8" aria-hidden>
                <line x1="0" y1="4" x2="20" y2="4" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash} />
              </svg>
              {s.label}
            </span>
          ))}
          {markers.length > 0 && (
            <span className="flex items-center gap-1.5">
              <svg width="8" height="10" aria-hidden><line x1="4" y1="0" x2="4" y2="10" stroke="var(--vehicle-rerouting)" strokeWidth="2" /></svg>
              Re-plan
            </span>
          )}
        </div>
      )}
      <div ref={ref} className="w-full">
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}. A data table is available below.`} className="block">
          {yTicks.map((v) => (
            <g key={v}>
              <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} stroke="var(--color-border-subtle)" />
              <text x={PAD.l - 6} y={y(v) + 4} textAnchor="end" fontSize={12} className="num fill-text-3">{formatY(v)}</text>
            </g>
          ))}
          {references.map((r) => (
            <g key={r.label}>
              <line x1={PAD.l} x2={W - PAD.r} y1={y(r.v)} y2={y(r.v)} stroke="var(--color-border-strong)" strokeDasharray="4 4" />
              <text x={W - PAD.r} y={y(r.v) - 4} textAnchor="end" fontSize={12} className="fill-text-3">{r.label}</text>
            </g>
          ))}
          <line x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={H - PAD.b} stroke="var(--color-border)" />
          <line x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} stroke="var(--color-border)" />
          {geom.xTicks.map((t) => (
            <text key={t} x={x(t)} y={H - PAD.b + 16} textAnchor="middle" fontSize={12} className="num fill-text-3">{t}</text>
          ))}
          {!compact && <text x={W - PAD.r} y={H - 2} textAnchor="end" fontSize={12} className="fill-text-3">Time (s)</text>}
          {!compact && <text x={4} y={PAD.t - 2} fontSize={12} className="fill-text-3">{yLabel}</text>}
          {markers.map((m) => (
            <line key={m.t} x1={x(m.t)} x2={x(m.t)} y1={PAD.t} y2={H - PAD.b} stroke="var(--vehicle-rerouting)" strokeWidth="1.5">
              <title>{m.label}</title>
            </line>
          ))}
          {geom.paths.map((p) => (
            <path key={p.key} d={p.d} fill="none" stroke={p.color} strokeWidth={p.key === series[0]?.key ? 2 : 1.5} strokeDasharray={p.dash} />
          ))}
          {cursor !== undefined && (
            <line x1={x(cursor)} x2={x(cursor)} y1={PAD.t} y2={H - PAD.b} stroke="var(--color-accent)" strokeWidth="1.5" />
          )}
        </svg>
      </div>
      <DataTableToggle>
        <Table caption={title} columns={columns} rows={rows} rowKey={(r) => String(r.t)} dense />
      </DataTableToggle>
    </figure>
  );
}
