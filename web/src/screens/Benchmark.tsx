import { memo, useEffect, useState } from 'react';
import { BookOpen, Compass, ShieldAlert } from 'lucide-react';
import { useReportStatus } from '../app/shellStatus';
import { WhiskerChart, type WhiskerRow } from '../components/charts/WhiskerChart';
import { Panel, StatusChip, Table, type Column } from '../components/ui';
import { METHODS, METHODS_SETUP, REAL_BENCHMARK, README_SOURCE, type MethodRow, type RealBenchmarkRow } from '../lib/benchmark';
import { SCENARIO_WORD } from '../lib/format';
import type { ScenarioTier } from '../lib/types';

const pm = (m: number | null, s: number | null) => (m === null || s === null ? 'Not iterative' : `${m.toFixed(1)} ± ${s.toFixed(1)}`);

const columns: Column<MethodRow>[] = [
  { key: 'method', header: 'Method', render: (r) => (
    <span className="flex flex-col">
      <span>{r.method}</span>
      <span className="text-caption text-text-3">{r.technical}</span>
    </span>
  ), sortValue: (r) => r.method },
  { key: 'best', header: 'Best score (s)', numeric: true, render: (r) => r.best.toFixed(3), sortValue: (r) => r.best },
  { key: 'mean', header: 'Mean ± std (s)', numeric: true, render: (r) => `${r.mean.toFixed(3)} ± ${r.meanStd.toFixed(3)}`, sortValue: (r) => r.mean },
  { key: 'to95', header: 'Rounds to 95%', numeric: true, render: (r) => pm(r.to95, r.to95Std), sortValue: (r) => r.to95 ?? Infinity },
  { key: 'margin', header: 'Rounds to 5% margin', numeric: true, render: (r) => pm(r.toMargin, r.toMarginStd), sortValue: (r) => r.toMargin ?? Infinity },
  { key: 'hit', header: 'Found the best route (%)', numeric: true, render: (r) => r.hitRate.toFixed(1), sortValue: (r) => r.hitRate },
];

const realColumns: Column<RealBenchmarkRow>[] = [
  { key: 'tier', header: 'Traffic tier', render: (r) => SCENARIO_WORD[r.tier], sortValue: (r) => r.tier },
  { key: 'method', header: 'Method', render: (r) => (
    <span className="flex flex-col">
      <span>{r.algorithm}</span>
      <span className="text-caption text-text-3">{r.technical}</span>
    </span>
  ), sortValue: (r) => r.algorithm },
  { key: 'completed', header: 'Completed (reached exit)', numeric: true, render: (r) => `${r.completed} / ${r.total} (${Math.round((r.completed / r.total) * 100)}%)`, sortValue: (r) => r.completed },
  { key: 'arrival', header: 'Mean arrival time (s)', numeric: true, render: (r) => (r.meanArrivalS !== null ? `${r.meanArrivalS.toFixed(1)} s` : 'Did not arrive within cap'), sortValue: (r) => r.meanArrivalS ?? 9999 },
  { key: 'range', header: 'Min – max (s)', numeric: true, render: (r) => (r.minArrivalS !== null && r.maxArrivalS !== null ? `${r.minArrivalS.toFixed(1)} – ${r.maxArrivalS.toFixed(1)} s` : '—'), sortValue: (r) => r.minArrivalS ?? 9999 },
  { key: 'driven', header: 'Mean distance driven', numeric: true, render: (r) => `${Math.round(r.meanDrivenM)} m`, sortValue: (r) => r.meanDrivenM },
];

const whiskerRows: WhiskerRow[] = METHODS.map((m) => ({
  key: m.key,
  label: m.method,
  mean: m.to95,
  std: m.to95Std,
  highlight: m.key === 'va',
}));

export const Benchmark = memo(function Benchmark() {
  const report = useReportStatus();
  useEffect(() => report({ systemState: 'Ready', dataSource: 'README snapshot & SUMO measured runs' }), [report]);
  const [tierFilter, setTierFilter] = useState<ScenarioTier | 'all'>('all');

  const filteredReal = tierFilter === 'all' ? REAL_BENCHMARK : REAL_BENCHMARK.filter((r) => r.tier === tierFilter);

  const fastest = METHODS.filter((m) => m.to95 !== null).sort((a, b) => (a.to95 ?? 0) - (b.to95 ?? 0))[0];
  const bestScore = Math.min(...METHODS.map((m) => m.best));
  const tiedBest = METHODS.filter((m) => m.best === bestScore).length;
  const va = METHODS.find((m) => m.key === 'va');
  const tiedMean = va ? METHODS.filter((m) => m.mean === va.mean && m.meanStd === va.meanStd).length : 0;
  const dijkstra = METHODS.find((m) => m.key === 'dijkstra');

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto grid max-w-[1600px] grid-cols-8 gap-4 p-6 wide:grid-cols-12 wide:gap-6">
        {/* Real-World SUMO Drive-Through Performance Table */}
        <div className="col-span-8 flex flex-col gap-2 wide:col-span-12">
          <p className="text-title-lg text-text-1">
            Real-world performance: measured SUMO drive-throughs (n=5 seeds per tier)
          </p>
          <p className="text-body text-text-2">
            Actual vehicle drive-throughs in calibrated Eclipse SUMO (50 km/h arterial / 30 km/h minor limits) with 8 stops
            and destination exit. Runs that did not reach the network exit within the 900 s cap are recorded honestly as &quot;Did not arrive within cap&quot;
            along with distance covered before timeout.
          </p>
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <StatusChip tone="info" icon={Compass} label="Measured SUMO vehicle runs" />
            <span className="text-caption text-text-3">Source: results/real_travel_time_benchmark.json · 90 total simulation trials</span>
            <div className="ml-auto flex items-center gap-1 text-caption">
              <span>Filter tier:</span>
              {(['all', 'low', 'medium', 'high'] as const).map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setTierFilter(t)}
                  className={`rounded px-2 py-0.5 ${tierFilter === t ? 'bg-primary text-text-inverse font-medium' : 'bg-surface-2 text-text-2 hover:bg-surface-3'}`}
                >
                  {t === 'all' ? 'All tiers' : SCENARIO_WORD[t]}
                </button>
              ))}
            </div>
          </div>
        </div>

        <Panel title="Measured SUMO drive-through performance across algorithms" className="col-span-8 wide:col-span-12">
          <Table
            caption="Measured SUMO vehicle drive-through results across six algorithms"
            columns={realColumns}
            rows={filteredReal}
            rowKey={(r) => `${r.tier}-${r.technical}`}
            source="Source: results/real_travel_time_benchmark.json (900 s cap per trial)."
          />
        </Panel>

        {/* Synthetic Congestion Search Dynamics Table */}
        <div className="col-span-8 mt-6 flex flex-col gap-2 border-t border-border-subtle pt-6 wide:col-span-12">
          {fastest && (
            <p className="text-title-lg text-text-1">
              Search dynamics: {fastest.method} reached 95% threshold in {fastest.to95?.toFixed(1)} rounds
            </p>
          )}
          <p className="text-body text-text-2">
            {tiedBest} metaheuristics found the same optimal visiting order ({bestScore.toFixed(1)} s score) and {tiedMean} share the same mean on the
            synthetic congestion model. This benchmark separates algorithms by search speed and convergence dynamics, not physical driving time.
            {dijkstra && ` The greedy shortest-path baseline scored ${dijkstra.best.toFixed(1)} s.`}
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <StatusChip tone="warning" icon={ShieldAlert} label="Algorithm comparison — synthetic congestion model, not measured travel time" />
            <StatusChip tone="neutral" icon={BookOpen} label="README snapshot" />
            <span className="text-caption text-text-3">Source: {README_SOURCE}, table 5 · {METHODS_SETUP}</span>
          </div>
        </div>

        <Panel title="Search rounds to reach 95% of the final score (synthetic model)" className="col-span-8 wide:col-span-7">
          <div className="flex flex-col gap-2">
            <p className="text-caption text-text-3">
              Point = mean, line = ± one standard deviation across 30 trials. Lines cut at 0 with a break mark where the
              spread is wider than the mean. Fewer rounds is faster.
            </p>
            <WhiskerChart title="Search rounds to reach 95% of the final score, by method" rows={whiskerRows} xMax={250} unit="rounds" emptyLabel="Not iterative" />
          </div>
        </Panel>

        <Panel title="Found the best route (synthetic model)" className="col-span-8 wide:col-span-5">
          <ul className="flex flex-col">
            {METHODS.map((m) => (
              <li key={m.key} className="flex items-center justify-between gap-3 border-t border-border-subtle py-2 first:border-t-0">
                <span className={m.key === 'va' ? 'text-body-sm font-semibold text-text-1' : 'text-body-sm text-text-1'}>{m.method}</span>
                <span className="num text-body-sm text-text-1">{m.hitRate.toFixed(1)}%</span>
              </li>
            ))}
          </ul>
          <p className="pt-2 text-caption text-text-3">Share of the 30 trials that reached the known best score.</p>
        </Panel>

        <Panel title="Full results (synthetic congestion model)" className="col-span-8 wide:col-span-12">
          <Table caption="Benchmark results for six route-search methods on synthetic congestion model" columns={columns} rows={METHODS} rowKey={(r) => r.key}
            source={`Source: ${README_SOURCE}, table 5. ${METHODS_SETUP}.`} />
        </Panel>

        <div className="col-span-8 flex flex-col gap-4 wide:col-span-12">
          <Panel title="Why the methods tie on synthetic score" collapsible defaultOpen={false}>
            <p className="text-body-sm text-text-1">
              With 8 stops, all five metaheuristic methods reach the same global-optimum stop sequence in every trial (100% hit rate),
              producing identical best (361.158 s) and mean (366.966 s) scores. The benchmark evaluates algorithmic efficiency:
              how many iterations they require to converge. Simulated annealing converges in 23.2 rounds and VA-QPSO in 35.3 rounds,
              while greedy Dijkstra-NN never finds the global optimum and incurs an average score of {dijkstra ? dijkstra.mean.toFixed(1) : ''} s.
            </p>
          </Panel>
          <Panel title="About these figures" collapsible defaultOpen={false}>
            <div className="flex flex-col gap-2 text-body-sm text-text-1">
              <p>
                Table 5 reflects algorithmic search dynamics under the synthetic congestion model across 30 seeds.
                Table 6 reflects real-world SUMO simulation vehicle travel times under calibrated Delhi speed limits (50 km/h / 30 km/h)
                with actual vehicle dispatch, queue delays, and destination exit arrival tracking.
              </p>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
});
