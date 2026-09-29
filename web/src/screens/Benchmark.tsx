import { memo, useEffect } from 'react';
import { BookOpen } from 'lucide-react';
import { useReportStatus } from '../app/shellStatus';
import { WhiskerChart, type WhiskerRow } from '../components/charts/WhiskerChart';
import { Panel, StatusChip, Table, type Column } from '../components/ui';
import { METHODS, METHODS_SETUP, README_SOURCE, type MethodRow } from '../lib/benchmark';

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

const whiskerRows: WhiskerRow[] = METHODS.map((m) => ({
  key: m.key,
  label: m.method,
  mean: m.to95,
  std: m.to95Std,
  highlight: m.key === 'va',
}));

export const Benchmark = memo(function Benchmark() {
  const report = useReportStatus();
  useEffect(() => report({ systemState: 'Ready', dataSource: 'README snapshot' }), [report]);

  const fastest = METHODS.filter((m) => m.to95 !== null).sort((a, b) => (a.to95 ?? 0) - (b.to95 ?? 0))[0];
  const bestScore = Math.min(...METHODS.map((m) => m.best));
  const tiedBest = METHODS.filter((m) => m.best === bestScore).length;
  const va = METHODS.find((m) => m.key === 'va');
  const tiedMean = va ? METHODS.filter((m) => m.mean === va.mean && m.meanStd === va.meanStd).length : 0;
  const dijkstra = METHODS.find((m) => m.key === 'dijkstra');

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto grid max-w-[1600px] grid-cols-8 gap-4 p-6 wide:grid-cols-12 wide:gap-6">
        <div className="col-span-8 flex flex-col gap-2 wide:col-span-12">
          {fastest && (
            <p className="text-title-lg text-text-1">
              {fastest.method} settled fastest: {fastest.to95?.toFixed(1)} search rounds to reach 95% of its final score.
            </p>
          )}
          <p className="text-body text-text-2">
            {tiedBest} methods found the same best score ({bestScore.toFixed(1)} s) and {tiedMean} share the same mean, so the
            difference is search speed, not route quality.{dijkstra && ` The shortest-path baseline scored ${dijkstra.best.toFixed(1)} s.`}
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <StatusChip tone="neutral" icon={BookOpen} label="README snapshot" />
            <span className="text-caption text-text-3">Source: {README_SOURCE}, table 5 · {METHODS_SETUP}</span>
          </div>
        </div>

        <Panel title="Search rounds to reach 95% of the final score" className="col-span-8 wide:col-span-7">
          <div className="flex flex-col gap-2">
            <p className="text-caption text-text-3">
              Point = mean, line = ± one standard deviation across 30 trials. Lines cut at 0 with a break mark where the
              spread is wider than the mean. Fewer rounds is faster.
            </p>
            <WhiskerChart title="Search rounds to reach 95% of the final score, by method" rows={whiskerRows} xMax={250} unit="rounds" emptyLabel="Not iterative" />
          </div>
        </Panel>

        <Panel title="Found the best route" className="col-span-8 wide:col-span-5">
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

        <Panel title="Full results" className="col-span-8 wide:col-span-12">
          <Table caption="Benchmark results for six route-search methods" columns={columns} rows={METHODS} rowKey={(r) => r.key}
            source={`Source: ${README_SOURCE}, table 5. ${METHODS_SETUP}.`} />
        </Panel>

        <div className="col-span-8 flex flex-col gap-4 wide:col-span-12">
          <Panel title="Why the methods tie on score" collapsible defaultOpen={false}>
            <p className="text-body-sm text-text-1">
              With 8 stops, the swarm and genetic methods reach the same best visiting order in every trial (100% found the
              best route), so their best and mean scores are identical. The benchmark separates them only by how many search
              rounds they needed. Annealing search found it in 96.7% of trials; the shortest-path baseline, which visits the
              nearest stop next, never did and averages {dijkstra ? dijkstra.mean.toFixed(1) : ''} s.
            </p>
          </Panel>
          <Panel title="About these figures" collapsible defaultOpen={false}>
            <div className="flex flex-col gap-2 text-body-sm text-text-1">
              <p>
                These are the README's published numbers, copied into the app. The server does not serve the results folder,
                so they can't be recomputed here.
              </p>
              <p>
                The README also shows the convergence curves as an image (results/convergence_comparison.png). It isn't shown
                here: the folder is excluded from the repository, and the curve data isn't in a file the app can read, so it
                can't be redrawn with a data table.
              </p>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
});
