import { memo, useEffect, useMemo, useState } from 'react';
import { BookOpen, CircleCheck, Columns2, History } from 'lucide-react';
import { useReportStatus } from '../app/shellStatus';
import { TimeSeriesChart } from '../components/charts/TimeSeriesChart';
import { PlaybackDock } from '../components/mission/PlaybackDock';
import { Button, EmptyState, ErrorState, Panel, Skeleton, StatusChip, Table, type Column } from '../components/ui';
import { usePlayback, type Playback } from '../hooks/usePlayback';
import { fetchRunData, recordedRunId, type LoadedRun } from '../lib/api';
import { PAIRED, PAIRED_SEEDS_IDENTICAL, README_SOURCE, type PairedRow } from '../lib/benchmark';
import { cx } from '../lib/cx';
import { buildTimeline, etaChangeWords } from '../lib/explain';
import { SCENARIO_WORD, TIER_WORD, formatClock, formatDuration } from '../lib/format';
import { replanEvents, trafficAt } from '../lib/timeline';
import { tripOf } from '../lib/trip';
import type { RunData, ScenarioTier } from '../lib/types';
import { MissionMap } from '../map/MissionMap';

const TIERS: ScenarioTier[] = ['low', 'medium', 'high'];

/** Honours the old deep link ?run=hero_{tier}_... for the starting traffic level. */
function initialTier(): ScenarioTier {
  const run = new URLSearchParams(window.location.search).get('run') ?? '';
  return TIERS.find((t) => run.includes(`_${t}_`)) ?? 'medium';
}

type Side = { state: 'ok'; run: RunData } | { state: 'missing' };
type Load = { state: 'loading' } | { state: 'error'; message: string } | { state: 'ready'; adaptive: Side; fixed: Side };

const pairedColumns: Column<PairedRow>[] = [
  { key: 'tier', header: 'Traffic', render: (r) => SCENARIO_WORD[r.tier] },
  { key: 'a', header: 'Adaptive mean (s)', numeric: true, render: (r) => r.adaptiveMean.toFixed(2) },
  { key: 'f', header: 'Fixed mean (s)', numeric: true, render: (r) => r.fixedMean.toFixed(2) },
  { key: 'd', header: 'Difference', numeric: true, render: (r) => `${Math.abs(r.diff).toFixed(2)} s ${r.diff <= 0 ? 'sooner' : 'later'}` },
  { key: 'p', header: 'Chance it is luck (p)', numeric: true, render: (r) => r.pValue.toFixed(4) },
  { key: 'a12', header: 'Adaptive wins (A12)', numeric: true, render: (r) => `${Math.round(r.a12 * 100)}%` },
  { key: 'c', header: 'Congestion score, adaptive − fixed', numeric: true, render: (r) => `${r.congestionDiff > 0 ? '+' : ''}${r.congestionDiff.toFixed(4)}` },
];

function SidePanel({ title, side, playback, style }: { title: string; side: Side; playback: Playback; style: 'optimized' | 'baseline' }) {
  const { t } = playback;
  const run = side.state === 'ok' ? side.run : null;
  const latest = useMemo(() => (run ? buildTimeline(run) : []), [run]);
  const series = useMemo(
    () =>
      run
        ? [{ key: 'v', label: 'Traffic unpredictability', color: 'var(--color-text-2)',
            points: run.metrics_over_time.filter((m) => m.t <= run.completion_time).map((m) => ({ t: m.t, v: m.volatility_index })) }]
        : [],
    [run],
  );
  if (!run) {
    return (
      <div className="flex flex-col gap-2 p-4">
        <p className="text-label text-text-2">{title}</p>
        <ErrorState title={`${title} run could not be loaded`} message="The other side still plays. No difference is shown." />
      </div>
    );
  }
  const remaining = Math.max(0, run.completion_time - t);
  const ended = t >= run.completion_time;
  const trip = tripOf(run);
  const current = [...latest].reverse().find((e) => e.t <= t);
  const m = trafficAt(run, t);
  return (
    <div className="flex min-h-0 flex-col">
      <div className="flex items-baseline justify-between gap-2 px-4 pb-2 pt-3">
        <span className="flex items-center gap-2 text-label text-text-2">
          <svg width="24" height="8" aria-hidden>
            <line x1="0" y1="4" x2="24" y2="4" stroke={style === 'optimized' ? 'var(--route-optimized)' : 'var(--route-baseline)'} strokeWidth="3" strokeDasharray={style === 'baseline' ? '6 4' : undefined} />
          </svg>
          {title}
        </span>
        <span className="num text-metric-md text-text-1">
          {ended
            ? trip.arrived
              ? `At exit ${formatClock(run.completion_time)}`
              : 'No arrival'
            : trip.arrived
              ? formatClock(remaining)
              : `Simulated ${formatClock(t)}`}
        </span>
      </div>
      <div className="relative min-h-[240px] flex-1">
        <MissionMap run={run} baseline={null} isLive={false} playback={playback} incidentPreview={null}
          routeStyle={style} showLegend={false} label={`${title} map`} />
      </div>
      <div className="flex flex-col gap-1 px-4 py-3">
        <p className="text-body-sm text-text-1">
          {current ? `T+${Math.round(current.t)} s: ${current.title}${current.eta ? `, ${etaChangeWords(current.eta)}` : ''}` : 'Waiting to dispatch'}
        </p>
        {m && <p className="text-caption text-text-3">Speeds {TIER_WORD[m.tier].toLowerCase()} {m.volatility_index.toFixed(2)}</p>}
        <TimeSeriesChart compact title={`${title}: traffic unpredictability`} series={series} xMax={run.completion_time}
          yMin={0} yMax={1} yTicks={[0, 1]} yLabel="V" formatY={(v) => v.toFixed(0)}
          markers={replanEvents(run.events).map((e) => ({ t: e.t, label: `Re-plan at T+${Math.round(e.t)} s` }))} cursor={t} height={64} />
      </div>
    </div>
  );
}

export const Analytics = memo(function Analytics() {
  const report = useReportStatus();
  const [tier, setTier] = useState<ScenarioTier>(initialTier);
  const [attempt, setAttempt] = useState(0);
  const key = `${tier}-${attempt}`;
  const [result, setResult] = useState<{ key: string; load: Load } | null>(null);

  // The same two requests as the old compare view.
  useEffect(() => {
    let cancelled = false;
    const side = (r: LoadedRun | null): Side => (r && !r.isDefaultFallback ? { state: 'ok', run: r.data } : { state: 'missing' });
    Promise.all([
      fetchRunData(recordedRunId(tier, 'va_qpso')).catch(() => null),
      fetchRunData(recordedRunId(tier, 'fixed_beta_qpso')).catch(() => null),
    ])
      .then(([a, f]) => {
        if (!cancelled) setResult({ key, load: { state: 'ready', adaptive: side(a), fixed: side(f) } });
      })
      .catch((err: unknown) => {
        if (!cancelled) setResult({ key, load: { state: 'error', message: err instanceof Error ? err.message : 'Unknown error' } });
      });
    return () => {
      cancelled = true;
    };
  }, [tier, key]);

  const load: Load = result?.key === key ? result.load : { state: 'loading' };
  const adaptive = load.state === 'ready' && load.adaptive.state === 'ok' ? load.adaptive.run : null;
  const fixed = load.state === 'ready' && load.fixed.state === 'ok' ? load.fixed.run : null;
  const duration = Math.max(adaptive?.completion_time ?? 0, fixed?.completion_time ?? 0);
  const playback = usePlayback(duration, `analytics-${key}-${duration}`);

  const metric = adaptive ? trafficAt(adaptive, playback.t) : null;
  useEffect(() => {
    report({
      systemState: load.state === 'error' ? 'Error' : duration > 0 ? (playback.playing ? 'Playing' : 'Paused') : 'Ready',
      dataSource: 'Recorded runs',
      traffic: metric ? { word: TIER_WORD[metric.tier], v: metric.volatility_index } : undefined,
      clock: duration > 0 ? { t: playback.t, total: duration, playing: playback.playing, speed: playback.speed } : undefined,
    });
  }, [report, load.state, duration, metric, playback.t, playback.playing, playback.speed]);

  const paired = PAIRED.find((p) => p.tier === tier);
  const aTrip = adaptive ? tripOf(adaptive) : null;
  const fTrip = fixed ? tripOf(fixed) : null;
  // A time difference exists only when both simulated ambulances reached the exit.
  const diff = aTrip && fTrip && aTrip.arrived && fTrip.arrived ? fTrip.duration - aTrip.duration : null;
  const capWords = formatDuration(aTrip?.capS ?? fTrip?.capS ?? duration);
  const headline =
    !aTrip || !fTrip
      ? 'Comparison incomplete: one run is missing'
      : diff !== null
        ? Math.abs(diff) < 0.05
          ? 'Both methods reached the network exit at the same time'
          : diff > 0
            ? `Adaptive routing reached the exit ${formatDuration(diff)} sooner`
            : `Fixed schedule reached the exit ${formatDuration(-diff)} sooner`
        : aTrip.arrived
          ? `Only adaptive routing reached the exit within ${capWords}`
          : fTrip.arrived
            ? `Only the fixed schedule reached the exit within ${capWords}`
            : `Neither method reached the network exit within ${capWords}`;
  const winnerArrived =
    diff !== null && adaptive && fixed && playback.t >= Math.min(adaptive.completion_time, fixed.completion_time);
  const timeWords = (t: typeof aTrip) => (t && t.arrived ? formatDuration(t.duration) : `no arrival in ${capWords}`);

  const tierPicker = (
    <div className="flex overflow-hidden rounded border border-border-control" role="group" aria-label="Traffic level">
      {TIERS.map((tt) => (
        <button key={tt} type="button" aria-pressed={tier === tt} onClick={() => setTier(tt)}
          className={cx('h-8 border-r border-border-subtle px-3 text-body-sm last:border-r-0',
            tier === tt ? 'bg-accent-weak text-text-1' : 'text-text-2 hover:bg-raised hover:text-text-1')}>
          {SCENARIO_WORD[tt]}
        </button>
      ))}
    </div>
  );

  if (load.state === 'loading') return <div className="flex flex-col gap-4 p-6">{tierPicker}<Skeleton lines={4} /></div>;
  if (load.state === 'error') {
    return (
      <div className="flex max-w-[640px] flex-col gap-4 p-6">
        {tierPicker}
        <ErrorState title="Recorded runs could not be loaded" message={`${load.message}.`} onRetry={() => setAttempt((a) => a + 1)} />
      </div>
    );
  }
  if (!adaptive && !fixed) {
    return (
      <div className="flex flex-col gap-4 p-6">
        {tierPicker}
        <EmptyState icon={Columns2} title={`No recorded runs for ${SCENARIO_WORD[tier]} traffic`} message="Choose another traffic level."
          action={<Button size="compact" onClick={() => setAttempt((a) => a + 1)}>Retry</Button>} />
      </div>
    );
  }

  const markers = [...(adaptive ? replanEvents(adaptive.events) : []), ...(fixed ? replanEvents(fixed.events) : [])]
    .map((e) => e.t)
    .filter((t, i, all) => all.indexOf(t) === i);

  return (
    <div className="grid h-full min-h-0 grid-rows-[auto_1fr_var(--shell-dock-h)] [grid-template-areas:'answer'_'maps'_'dock']">
      <div className="flex flex-col gap-2 border-b border-border bg-panel px-6 py-4 [grid-area:answer]">
        <div className="flex flex-wrap items-center gap-4">
          <p className="text-title-lg text-text-1">{headline}</p>
          {winnerArrived && diff !== null && Math.abs(diff) >= 0.05 && (
            <StatusChip tone="success" icon={CircleCheck} label={`${diff > 0 ? 'Adaptive routing' : 'Fixed schedule'} reached the hospital first`} />
          )}
          <div className="ml-auto flex items-center gap-3">
            <StatusChip tone="neutral" icon={History} label="Recorded runs" />
            {tierPicker}
          </div>
        </div>
        {adaptive && fixed && (
          <p className="text-body-sm text-text-2">
            {timeWords(aTrip)} adaptive vs {timeWords(fTrip)} fixed at {SCENARIO_WORD[tier]} traffic (simulated drives to the network exit).
            {paired?.congestionNote && ` The adaptive route had ${paired.congestionNote} (${README_SOURCE}).`}
          </p>
        )}
        <Panel title="Across repeated trials" collapsible defaultOpen={false} className="mt-1">
          <div className="flex flex-col gap-2">
            <Table caption="Paired comparison of adaptive routing and the fixed schedule" columns={pairedColumns} rows={PAIRED}
              rowKey={(r) => r.tier} selectedKey={tier} dense source={`Source: ${README_SOURCE}, table 4 (10 seeds per traffic level). Differences are adaptive minus fixed.`} />
            {PAIRED_SEEDS_IDENTICAL && (
              <p className="flex gap-2 text-caption text-warning">
                <BookOpen size={14} strokeWidth={1.75} aria-hidden className="mt-0.5 shrink-0" />
                In results/experiments.csv every one of the 10 seeds in a traffic level gave exactly the same time and congestion
                score. The 10 trials are one outcome repeated, so the p-value reflects ten identical differences, not variation
                across conditions.
              </p>
            )}
          </div>
        </Panel>
      </div>

      <div className="grid min-h-0 grid-cols-1 overflow-y-auto [grid-area:maps] desktop:grid-cols-2 desktop:overflow-hidden">
        <div className={cx('flex min-h-[420px] flex-col border-b border-border desktop:border-b-0 desktop:border-r')}>
          <SidePanel title="Adaptive routing" side={load.adaptive} playback={playback} style="optimized" />
        </div>
        <div className="flex min-h-[420px] flex-col">
          <SidePanel title="Fixed schedule" side={load.fixed} playback={playback} style="baseline" />
        </div>
      </div>

      <PlaybackDock playback={playback} markers={markers} />
    </div>
  );
});
