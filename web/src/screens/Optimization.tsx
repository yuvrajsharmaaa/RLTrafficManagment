import { memo, useEffect, useMemo, useState, type ReactNode } from 'react';
import { CircleSlash, Cpu, Database, History, Radio } from 'lucide-react';
import { useSession, type SessionRun } from '../app/sessionRun';
import { useReportStatus } from '../app/shellStatus';
import { TimeSeriesChart } from '../components/charts/TimeSeriesChart';
import { PlaybackDock } from '../components/mission/PlaybackDock';
import { TierLabel } from '../components/mission/TierLabel';
import { Button, EmptyState, ErrorState, Panel, Skeleton, StatBlock, StatusChip, Table, type Column } from '../components/ui';
import { usePlayback } from '../hooks/usePlayback';
import { fetchRunData, recordedRunId } from '../lib/api';
import { METHODS, METHODS_SETUP, README_SOURCE, type MethodRow } from '../lib/benchmark';
import { cx } from '../lib/cx';
import { doingNow } from '../lib/explain';
import { SCENARIO_WORD, SOURCE_WORD, TIER_WORD, formatDuration, shortName } from '../lib/format';
import { BETA_MAX, BETA_MIN, FIXED_EXPORT_BETA, LIVE_SEARCH_BUDGET, VA_FLOOR_SPAN, explorationKept } from '../lib/search';
import { replanEvents, trafficAt } from '../lib/timeline';
import { tripOf } from '../lib/trip';
import type { ScenarioTier } from '../lib/types';

type Choice = { kind: 'session' } | { kind: 'recorded'; tier: ScenarioTier };

type Load =
  | { state: 'empty' }
  | { state: 'loading' }
  | { state: 'error'; message: string }
  | { state: 'ready'; data: SessionRun; note: string | null };

// Aggregate figures only (README); the per-iteration curves are not available here.
const speedColumns: Column<MethodRow>[] = [
  { key: 'm', header: 'Method', render: (r) => `${r.method} (${r.technical})` },
  { key: 'r', header: 'Rounds to 95% (mean ± std)', numeric: true, render: (r) => (r.to95 === null || r.to95Std === null ? 'Not iterative' : `${r.to95.toFixed(1)} ± ${r.to95Std.toFixed(1)}`) },
  { key: 'b', header: 'Best score (s)', numeric: true, render: (r) => r.best.toFixed(3) },
];

function Unavailable({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Panel raised title={title} collapsible defaultOpen={false}
      actions={<StatusChip tone="neutral" icon={CircleSlash} label="Not available" />}>
      <div className="flex flex-col gap-2 text-body-sm text-text-2">{children}</div>
    </Panel>
  );
}

export const Optimization = memo(function Optimization() {
  const session = useSession();
  const report = useReportStatus();
  const [choice, setChoice] = useState<Choice>(session.current ? { kind: 'session' } : { kind: 'recorded', tier: 'medium' });
  const [attempt, setAttempt] = useState(0);
  const recordedKey = choice.kind === 'recorded' ? `${choice.tier}-${attempt}` : null;
  const [recorded, setRecorded] = useState<{ key: string; result: Load } | null>(null);

  // Recorded runs load the adaptive run and its fixed-schedule pair (the same calls as the old compare view).
  useEffect(() => {
    if (choice.kind !== 'recorded' || recordedKey === null) return;
    let cancelled = false;
    const setLoad = (result: Load) => {
      if (!cancelled) setRecorded({ key: recordedKey, result });
    };
    Promise.all([
      fetchRunData(recordedRunId(choice.tier, 'va_qpso')),
      fetchRunData(recordedRunId(choice.tier, 'fixed_beta_qpso')).catch(() => null),
    ])
      .then(([adaptive, fixed]) => {
        setLoad({
          state: 'ready',
          data: {
            run: adaptive.data,
            baseline: fixed && !fixed.isDefaultFallback && !adaptive.isDefaultFallback ? fixed.data : null,
            kind: 'recorded',
            tier: adaptive.isDefaultFallback ? 'medium' : choice.tier,
            requestMs: null,
          },
          note: adaptive.isDefaultFallback
            ? `The ${SCENARIO_WORD[choice.tier]} traffic run could not be loaded. Showing the default run (Moderate traffic).`
            : null,
        });
      })
      .catch((err: unknown) => {
        setLoad({ state: 'error', message: err instanceof Error ? err.message : 'Unknown error' });
      });
    return () => {
      cancelled = true;
    };
  }, [choice, recordedKey]);

  const load: Load =
    choice.kind === 'session'
      ? session.current
        ? { state: 'ready', data: session.current, note: null }
        : { state: 'empty' }
      : recorded?.key === recordedKey
        ? recorded.result
        : { state: 'loading' };

  const data = load.state === 'ready' ? load.data : null;
  const run = data?.run ?? null;
  const duration = run?.completion_time ?? 0;
  const playback = usePlayback(duration, run ? `${data?.kind}-${run.scenario}-${run.completion_time}-${run.seed ?? ''}` : 'none');
  const t = playback.t;
  const metric = run ? trafficAt(run, t) : null;

  useEffect(() => {
    report({
      systemState: load.state === 'error' ? 'Error' : run ? (playback.playing ? 'Playing' : 'Paused') : 'Ready',
      dataSource: data ? (data.kind === 'recorded' ? 'Recorded run' : (run?.source && SOURCE_WORD[run.source]) ?? 'Live result') : undefined,
      traffic: metric ? { word: TIER_WORD[metric.tier], v: metric.volatility_index } : undefined,
      clock: run ? { t, total: run.completion_time, playing: playback.playing, speed: playback.speed } : undefined,
    });
  }, [report, load.state, data, run, metric, t, playback.playing, playback.speed]);

  const baseline = data?.baseline ?? null;
  const note = load.state === 'ready' ? load.note : null;
  const derived = useMemo(() => {
    if (!run) return null;
    // Every sample is a SUMO measurement: per second for simulated drives, one dispatch reading for estimates.
    const inTrip = run.metrics_over_time.filter((m) => m.t <= run.completion_time);
    const peak = inTrip.reduce((a, m) => (m.beta > a.beta ? m : a), inTrip[0] ?? { t: 0, beta: BETA_MIN, volatility_index: 0, tier: 'calm' as const });
    return {
      replans: replanEvents(run.events),
      vSeries: [{ key: 'v', label: 'Traffic unpredictability (V)', color: 'var(--color-text-1)', points: inTrip.map((m) => ({ t: m.t, v: m.volatility_index })) }],
      betaSeries: [
        ...(run.algorithm === 'va_qpso'
          ? [{ key: 'floor', label: 'Adaptive: search-breadth floor', color: 'var(--route-optimized)', points: inTrip.map((m) => ({ t: m.t, v: m.beta })) }]
          : []),
        ...(baseline
          ? [{ key: 'fixed', label: `Fixed schedule: floor ${FIXED_EXPORT_BETA.toFixed(2)}, ignores traffic`, color: 'var(--route-baseline)', dash: '6 4',
              points: baseline.metrics_over_time.filter((m) => m.t <= run.completion_time).map((m) => ({ t: m.t, v: m.beta })) }]
          : []),
      ],
      peak,
    };
  }, [run, baseline]);

  const chooser = (
    <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Run to explain">
      {session.current && (
        <Button size="compact" variant={choice.kind === 'session' ? 'primary' : 'secondary'} icon={session.current.kind === 'live' ? Radio : Database}
          onClick={() => setChoice({ kind: 'session' })}>
          Current mission
        </Button>
      )}
      {(['low', 'medium', 'high'] as ScenarioTier[]).map((tier) => (
        <Button key={tier} size="compact" icon={History}
          variant={choice.kind === 'recorded' && choice.tier === tier ? 'primary' : 'secondary'}
          onClick={() => setChoice({ kind: 'recorded', tier })}>
          Recorded: {SCENARIO_WORD[tier]}
        </Button>
      ))}
    </div>
  );

  if (load.state === 'loading') {
    return (
      <div className="flex flex-col gap-4 p-6">
        {chooser}
        <Skeleton lines={4} />
      </div>
    );
  }
  if (load.state === 'error') {
    return (
      <div className="flex max-w-[640px] flex-col gap-4 p-6">
        {chooser}
        <ErrorState title="Run could not be loaded" message={`${load.message}.`} onRetry={() => setAttempt((a) => a + 1)} />
      </div>
    );
  }
  if (!run || !derived || !data) {
    return (
      <div className="p-6">
        <EmptyState icon={Cpu} title="No run to explain" message="Open a recorded run above, or finish a live dispatch in Mission Control." action={chooser} />
      </div>
    );
  }

  const isAdaptive = run.algorithm === 'va_qpso';
  const trip = tripOf(run);
  const baseTrip = data.baseline ? tripOf(data.baseline) : null;
  // A difference exists only when both runs reached the exit.
  const diff = baseTrip && trip.arrived && baseTrip.arrived ? baseTrip.duration - trip.duration : null;
  const kept = metric && isAdaptive ? explorationKept(metric.beta) : null;
  const answer = !isAdaptive
    ? 'This is a fixed-schedule run: its search narrows on the same timetable whatever the traffic.'
    : run.metrics_over_time.length <= 1
      ? `Speeds were ${TIER_WORD[derived.peak.tier].toLowerCase()} when this route was requested, so each search narrowed to a floor of ${derived.peak.beta.toFixed(2)}. The fixed schedule does not react to traffic.`
      : `Search-breadth floor rose to ${derived.peak.beta.toFixed(2)} at T+${Math.round(derived.peak.t)} s, when speeds were ${TIER_WORD[derived.peak.tier].toLowerCase()}. The fixed schedule does not react to traffic.`;

  return (
    <div className="grid h-full min-h-0 grid-rows-[1fr_var(--shell-dock-h)] [grid-template-areas:'body'_'dock']">
      <div className="min-h-0 overflow-y-auto [grid-area:body]">
        <div className="mx-auto grid max-w-[1600px] grid-cols-8 gap-4 p-6 wide:grid-cols-12 wide:gap-6">
          {/* Answer + run choice */}
          <div className="col-span-8 flex flex-col gap-3 wide:col-span-12">
            <p className="text-title-lg text-text-1">{answer}</p>
            <div className="flex flex-wrap items-center gap-3">
              <StatusChip tone="neutral" icon={History} label="Replay of completed run" />
              <span className="text-caption text-text-3">
                {data.kind === 'recorded' ? `Recorded run, ${SCENARIO_WORD[data.tier]} traffic` : `Live run, ${SCENARIO_WORD[data.tier]} traffic`}
                {' · '}{isAdaptive ? 'Adaptive routing' : 'Fixed schedule'}
              </span>
              <div className="ml-auto">{chooser}</div>
            </div>
            {note && <p className="text-caption text-warning">{note}</p>}
          </div>

          {/* 1. Summary strip */}
          <section aria-label="Summary" className="col-span-8 grid grid-cols-2 gap-4 rounded border border-border bg-panel p-4 desktop:grid-cols-5 wide:col-span-12">
            <StatBlock label="Status" value={t >= duration ? 'Arrived' : playback.playing ? 'Replaying' : 'Paused'} size="md" delta="The search finished before playback" />
            <StatBlock label="Trip time" value={`T+${Math.floor(t)} / ${Math.round(duration)}`} unit="s" size="md" delta="Search iterations: not reported" />
            <StatBlock
              label={trip.state === 'estimate' ? 'Planner estimate' : 'Time to network exit'}
              value={trip.arrived || trip.state === 'estimate' ? formatDuration(duration) : 'No arrival'}
              size="md"
              delta={
                trip.arrived || trip.state === 'estimate'
                  ? trip.timeSource
                  : `Not at exit within ${formatDuration(trip.capS ?? duration)} simulated`
              }
            />
            <StatBlock
              label="Versus fixed schedule"
              value={diff === null ? null : `${Math.abs(diff).toFixed(1)} s`}
              size="md"
              delta={
                diff !== null
                  ? diff >= 0 ? 'sooner on the same traffic' : 'later on the same traffic'
                  : baseTrip
                    ? 'No difference: a run did not reach the exit'
                    : data.kind === 'live' ? 'No baseline for live runs' : 'Not loaded'
              }
            />
            <StatBlock
              label="Search compute time"
              value={data.requestMs === null ? null : `${Math.round(data.requestMs).toLocaleString('en-IN')} ms`}
              size="md"
              delta={data.requestMs === null ? 'Not measured for recorded runs' : 'Request round trip, measured in the browser'}
            />
          </section>

          {/* Main column */}
          <div className="col-span-8 flex flex-col gap-4">
            {/* 6. What the algorithm is doing now */}
            <Panel title="What the algorithm is doing now">
              <ul className="flex flex-col gap-1.5 text-body text-text-1" aria-live="off">
                {doingNow(run, t).map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </Panel>

            {/* 4. Adaptive behaviour */}
            <Panel title="How the search adapted to traffic">
              <div className="flex flex-col gap-6">
                <div className="flex flex-col gap-2">
                  <p className="flex items-baseline gap-2 text-body">
                    {metric && <TierLabel tier={metric.tier} />}
                    <span className="text-text-2">Traffic unpredictability, network-wide</span>
                  </p>
                  <TimeSeriesChart
                    title="Traffic unpredictability over the trip"
                    series={derived.vSeries}
                    xMax={duration}
                    yMin={0}
                    yMax={1}
                    yTicks={[0, 0.25, 0.5, 0.75, 1]}
                    yLabel="V (0 to 1)"
                    formatY={(v) => v.toFixed(2)}
                    references={[{ v: 0.25, label: 'Changing' }, { v: 0.5, label: 'Unstable' }]}
                    markers={derived.replans.map((e) => ({ t: e.t, label: `Re-plan at T+${Math.round(e.t)} s` }))}
                    cursor={t}
                  />
                </div>

                <div className="flex flex-col gap-2">
                  <p className="text-body text-text-2">Search-breadth floor: the lowest β each search contracts to</p>
                  <TimeSeriesChart
                    title="Search-breadth floor over the trip"
                    series={derived.betaSeries}
                    xMax={duration}
                    yMin={BETA_MIN}
                    yMax={BETA_MAX}
                    yTicks={[0.5, 0.625, 0.75, 0.875, 1]}
                    yLabel="β"
                    formatY={(v) => v.toFixed(2)}
                    markers={derived.replans.map((e) => ({ t: e.t, label: `Re-plan at T+${Math.round(e.t)} s` }))}
                    cursor={t}
                    height={160}
                  />
                  <p className="text-caption text-text-3">
                    {run.metrics_over_time.length <= 1
                      ? `Floor = ${BETA_MIN.toFixed(2)} + ${VA_FLOOR_SPAN.toFixed(2)} × V, from the one traffic reading taken in SUMO at dispatch.`
                      : `Floor = ${BETA_MIN.toFixed(2)} + ${VA_FLOOR_SPAN.toFixed(2)} × V, with V measured in SUMO every simulated second.`}
                  </p>
                  {data.baseline && (
                    <p className="text-caption text-text-3">
                      The fixed schedule narrows from {BETA_MAX.toFixed(1)} to {BETA_MIN.toFixed(2)} inside every search, whatever the traffic, so its floor is {FIXED_EXPORT_BETA.toFixed(2)} at every second.
                    </p>
                  )}
                </div>

                {kept !== null && (
                  <div className="flex flex-col gap-2">
                    <div className="flex items-baseline justify-between">
                      <span className="text-body text-text-2">Exploration kept at the end of a search</span>
                      <span className="num text-metric-md text-text-1">{Math.round(kept * 100)}%</span>
                    </div>
                    <div
                      role="meter"
                      aria-label="Exploration kept at the end of a search"
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={Math.round(kept * 100)}
                      className="relative h-2 rounded-sm bg-raised"
                    >
                      <div className="h-full rounded-sm bg-route-optimized" style={{ width: `${kept * 100}%` }} />
                    </div>
                    <div className="flex justify-between text-caption text-text-3">
                      <span>0%: narrows fully onto the best route (exploitation)</span>
                      <span>100%: never narrows (exploration)</span>
                    </div>
                    <p className="text-caption text-text-3">
                      Defined as (floor − {BETA_MIN.toFixed(2)}) ÷ ({BETA_MAX.toFixed(2)} − {BETA_MIN.toFixed(2)}), from the exported floor and the constants in qpso.py. At V = 1 it reaches {Math.round((VA_FLOOR_SPAN / (BETA_MAX - BETA_MIN)) * 100)}%. The fixed schedule is always 0%.
                    </p>
                  </div>
                )}
              </div>
            </Panel>
          </div>

          {/* Side column */}
          <div className="col-span-8 flex flex-col gap-4 wide:col-span-4">
            <Panel title="About this method">
              <div className="flex flex-col gap-3 text-body-sm text-text-1">
                <p>
                  This is quantum-inspired optimisation running on an ordinary, classical computer. No quantum computer is
                  involved. The method, Quantum-behaved Particle Swarm Optimisation (QPSO), borrows one idea from quantum
                  physics: each candidate is redrawn from a probability cloud around promising positions, instead of moving
                  with a fixed speed and direction.
                </p>
                <p>
                  Here each candidate, or "particle", is one order for visiting the corridor stops. Many candidates are scored
                  at once on travel time, distance and congestion, and every round they are redrawn around the best found so
                  far. Search breadth (β) sets how wide that cloud is: each search starts wide at {BETA_MAX.toFixed(1)} and
                  narrows, so it explores first and then settles.
                </p>
                <p>
                  Volatility adaptation ties that narrowing to traffic. When road speeds become unpredictable, the adaptive
                  search stops narrowing earlier (its floor rises above {BETA_MIN.toFixed(2)}, further the more unpredictable traffic is),
                  so it keeps trying other streets. In steady traffic it narrows fully, exactly like the fixed schedule.
                </p>
              </div>
            </Panel>

            <Panel title="What this data contains" collapsible defaultOpen={false}>
              <div className="flex flex-col gap-3 text-body-sm">
                <div>
                  <p className="pb-1 text-label text-text-2">Available</p>
                  <ul className="list-disc pl-4 text-text-1">
                    <li>Traffic unpredictability and congestion for each simulated second (simulated drives); one reading at dispatch (planner estimates)</li>
                    <li>Search-breadth floor for each second (adaptive runs)</li>
                    <li>Re-plan times, whether the waypoint order changed, and the planner's remaining-time estimate at each re-plan (simulated drives)</li>
                    <li>The final route, its total time, the hospitals considered</li>
                    <li>Run number and data source (live runs)</li>
                  </ul>
                </div>
                <div>
                  <p className="pb-1 text-label text-text-2">Not available from the server</p>
                  <ul className="list-disc pl-4 text-text-1">
                    <li>Particle positions and the global best per iteration</li>
                    <li>β per iteration, and how many iterations ran</li>
                  </ul>
                  <p className="pt-2 text-caption text-text-3">
                    Live searches are configured in server.py for {LIVE_SEARCH_BUDGET.particles} particles and {LIVE_SEARCH_BUDGET.iterations} iterations; the response does not confirm this.
                  </p>
                </div>
              </div>
            </Panel>

            {run.best_score_history && run.best_score_history.length > 1 ? (
              <Panel raised title="Convergence by iteration">
                <div className="flex flex-col gap-2 text-body-sm text-text-2">
                  <TimeSeriesChart
                    title="Best route score after each search iteration (dispatch plan)"
                    series={[{ key: 'best', label: 'Best score (s)', color: 'var(--route-optimized)',
                      points: run.best_score_history.map((v, i) => ({ t: i + 1, v })) }]}
                    xMax={run.best_score_history.length}
                    yMin={Math.floor(Math.min(...run.best_score_history) * 0.95)}
                    yMax={Math.ceil(Math.max(...run.best_score_history) * 1.02)}
                    yTicks={[...new Set([Math.min(...run.best_score_history), Math.max(...run.best_score_history)].map((v) => Math.round(v)))]}
                    xLabel="Iteration"
                    yLabel="s"
                    formatY={(v) => v.toFixed(0)}
                    height={160}
                  />
                  <p className="text-caption text-text-3">
                    Score = planned travel time to the network exit in seconds, from the edge speeds SUMO measured at dispatch. X axis: search iteration.
                  </p>
                  <p className="text-caption text-text-3">Algorithm comparison — synthetic congestion model, not measured travel time.</p>
                  <Table caption="Search speed across 30 trials (synthetic congestion model)" columns={speedColumns} rows={METHODS} rowKey={(r) => r.key} source={`Source: ${README_SOURCE}, table 5 (${METHODS_SETUP})`} dense />
                </div>
              </Panel>
            ) : (
              <Unavailable title="Convergence by iteration">
                <p>This run file has no per-iteration scores, so no convergence curve is drawn. The summary below is from repeated trials on a synthetic congestion model (not measured SUMO travel time).</p>
                <Table caption="Search speed across 30 trials (synthetic congestion model)" columns={speedColumns} rows={METHODS} rowKey={(r) => r.key} source={`Source: ${README_SOURCE}, table 5 (${METHODS_SETUP})`} dense />
              </Unavailable>
            )}

            <Unavailable title="Swarm view">
              <p>Particle positions are not sent by the server, so the swarm can't be drawn or replayed. No projection is shown rather than an invented one.</p>
            </Unavailable>

            <Panel title="Search space" collapsible defaultOpen={false}>
              <div className="flex flex-col gap-3 text-body-sm">
                <p className="text-text-2">How the choices narrowed for this trip, from the fields the server returns.</p>
                <ol className="flex flex-col gap-2">
                  <li className="flex justify-between gap-3">
                    <span className="text-text-1">Hospitals considered</span>
                    <span className="num text-text-1">{run.hospital_candidates.length}</span>
                  </li>
                  <li className={cx('flex justify-between gap-3 border-t border-border-subtle pt-2')}>
                    <span className="text-text-1">Destination, by shortest unsimulated distance</span>
                    <span className="text-right text-text-1">{run.selected_hospital ? shortName(run.selected_hospital.name) : 'No data'}</span>
                  </li>
                  <li className="flex justify-between gap-3 border-t border-border-subtle pt-2">
                    <span className="text-text-1">Stops: pickup, planner waypoints, network exit</span>
                    <span className="num text-text-1">{run.stops.length}</span>
                  </li>
                  <li className="flex justify-between gap-3 border-t border-border-subtle pt-2">
                    <span className="text-text-1">Visiting order chosen by the route search</span>
                    <span className="num text-text-1">1</span>
                  </li>
                </ol>
                <p className="text-caption text-text-3">
                  The destination is the hospital closest to the simulated map (straight-line from its network exit), picked
                  before the route search runs. The search orders the auto-selected waypoints between the fixed pickup and
                  network exit. The candidate orders it tried are not reported, so how they narrowed can't be shown.
                </p>
              </div>
            </Panel>
          </div>
        </div>
      </div>
      <PlaybackDock playback={playback} markers={derived.replans.map((e) => e.t)} />
    </div>
  );
});
