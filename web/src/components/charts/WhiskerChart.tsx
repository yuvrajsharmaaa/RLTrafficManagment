import { Table, type Column } from '../ui';
import { useElementWidth } from '../../hooks/useElementWidth';
import { DataTableToggle } from './DataTableToggle';

export interface WhiskerRow {
  key: string;
  label: string;
  /** null = not applicable (e.g. a method that doesn't iterate); drawn as text, never as 0. */
  mean: number | null;
  std: number | null;
  highlight?: boolean;
  note?: string;
}

interface WhiskerChartProps {
  title: string;
  rows: WhiskerRow[];
  xMax: number;
  unit: string;
  /** Label for rows with no value. */
  emptyLabel: string;
}

const ROW_H = 28;

/**
 * Mean as a point, ± one standard deviation as a whisker (Phase 2: never
 * floating bars). Whiskers below 0 are cut at the axis with a break mark.
 */
export function WhiskerChart({ title, rows, xMax, unit, emptyLabel }: WhiskerChartProps) {
  const [ref, W] = useElementWidth<HTMLDivElement>(560);
  const LABEL_W = Math.min(220, Math.round(W * 0.36));
  const PAD = { l: LABEL_W, r: 56, t: 4, b: 26 };
  const H = PAD.t + rows.length * ROW_H + PAD.b;
  const x = (v: number) => PAD.l + (Math.max(0, Math.min(v, xMax)) / xMax) * (W - PAD.l - PAD.r);
  const step = xMax <= 50 ? 10 : xMax <= 150 ? 25 : 50;
  const ticks = Array.from({ length: Math.floor(xMax / step) + 1 }, (_, i) => i * step);

  const columns: Column<WhiskerRow>[] = [
    { key: 'label', header: 'Method', render: (r) => r.label },
    { key: 'mean', header: `Mean (${unit})`, numeric: true, render: (r) => (r.mean === null ? emptyLabel : r.mean.toFixed(1)) },
    { key: 'std', header: `Std (${unit})`, numeric: true, render: (r) => (r.std === null ? emptyLabel : r.std.toFixed(1)) },
  ];

  return (
    <figure className="flex flex-col gap-2">
      <div ref={ref} className="w-full">
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}. A data table is available below.`} className="block">
          {ticks.map((v) => (
            <g key={v}>
              <line x1={x(v)} x2={x(v)} y1={PAD.t} y2={H - PAD.b} stroke="var(--color-border-subtle)" />
              <text x={x(v)} y={H - PAD.b + 16} textAnchor="middle" fontSize={12} className="num fill-text-3">{v}</text>
            </g>
          ))}
          <text x={W - PAD.r + 4} y={H - PAD.b + 16} fontSize={12} className="fill-text-3">{unit}</text>
          <line x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={H - PAD.b} stroke="var(--color-border)" />
          {rows.map((r, i) => {
            const cy = PAD.t + i * ROW_H + ROW_H / 2;
            const color = r.highlight ? 'var(--series-va-qpso)' : 'var(--color-text-2)';
            return (
              <g key={r.key}>
                <text x={PAD.l - 8} y={cy + 4} textAnchor="end" fontSize={12} className={r.highlight ? 'fill-text-1' : 'fill-text-2'} fontWeight={r.highlight ? 600 : 400}>
                  {r.label}
                </text>
                {r.mean === null || r.std === null ? (
                  <text x={PAD.l + 8} y={cy + 4} fontSize={12} className="fill-text-3">{emptyLabel}</text>
                ) : (
                  <>
                    <line x1={x(r.mean - r.std)} x2={x(r.mean + r.std)} y1={cy} y2={cy} stroke={color} strokeWidth="1.5" />
                    <line x1={x(r.mean + r.std)} x2={x(r.mean + r.std)} y1={cy - 5} y2={cy + 5} stroke={color} strokeWidth="1.5" />
                    {r.mean - r.std < 0 ? (
                      // Cut at the axis: two short slashes instead of an end cap.
                      <path d={`M${PAD.l + 2} ${cy + 5} l4 -10 M${PAD.l + 6} ${cy + 5} l4 -10`} stroke={color} strokeWidth="1.25" />
                    ) : (
                      <line x1={x(r.mean - r.std)} x2={x(r.mean - r.std)} y1={cy - 5} y2={cy + 5} stroke={color} strokeWidth="1.5" />
                    )}
                    <circle cx={x(r.mean)} cy={cy} r={r.highlight ? 5 : 4} fill={color} />
                    <text x={x(r.mean + r.std) + 8} y={cy + 4} fontSize={12} className="num fill-text-1">{r.mean.toFixed(1)}</text>
                  </>
                )}
              </g>
            );
          })}
        </svg>
      </div>
      <DataTableToggle>
        <Table caption={title} columns={columns} rows={rows} rowKey={(r) => r.key} dense />
      </DataTableToggle>
    </figure>
  );
}
