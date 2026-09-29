import type { ReactNode } from 'react';
import { ArrowRight } from 'lucide-react';
import { Panel, Tooltip } from '../ui';
import { SOURCE_WORD, formatDuration } from '../../lib/format';
import { trafficAt } from '../../lib/timeline';
import type { RunData } from '../../lib/types';

interface OptimizationDetailsProps {
  run: RunData;
  kind: 'live' | 'recorded';
  t: number;
  /** Request round-trip for live runs, measured in the browser. */
  requestMs: number | null;
  onOpenOptimization: () => void;
}

function Row({ label, tip, children }: { label: string; tip?: string; children: ReactNode }) {
  return (
    <div className="flex justify-between gap-3">
      {tip ? (
        <Tooltip content={tip}>
          <span tabIndex={0} className="text-text-2 underline decoration-dotted underline-offset-2">{label}</span>
        </Tooltip>
      ) : (
        <span className="text-text-2">{label}</span>
      )}
      <span className="text-right text-text-1">{children}</span>
    </div>
  );
}

/** "How this route was found". Rows without data are left out, never shown blank. */
export function OptimizationDetails({ run, kind, t, requestMs, onOpenOptimization }: OptimizationDetailsProps) {
  const m = trafficAt(run, t);
  return (
    <Panel raised title="How this route was found" collapsible defaultOpen={false}>
      <div className="flex flex-col gap-1.5 text-body-sm">
        <Row label="Method" tip={run.algorithm === 'va_qpso' ? 'Volatility-adaptive QPSO (va_qpso)' : 'QPSO with a fixed schedule (fixed_beta_qpso)'}>
          {run.algorithm === 'va_qpso' ? 'Adaptive routing' : 'Fixed schedule'}
        </Row>
        <Row label="Data">{kind === 'recorded' ? 'Recorded run' : (run.source && SOURCE_WORD[run.source]) ?? 'Live result'}</Row>
        {m && (
          <Row label="Search-breadth floor now" tip={kind === 'recorded' ? 'Lowest β each search contracts to. Recorded files: 0.50 + 0.50 × traffic unpredictability' : 'Lowest β each search contracts to: 0.50 + 0.25 × traffic unpredictability'}>
            <span className="num">{m.beta.toFixed(2)}</span>
          </Row>
        )}
        {run.seed !== undefined && (
          <Row label="Run number" tip="Random seed; the same number reproduces the run">
            <span className="num">{run.seed}</span>
          </Row>
        )}
        <Row label="Stops">
          <span className="num">{run.stops.length}</span>
        </Row>
        <Row label="Total time to hospital">
          <span className="num">{formatDuration(run.completion_time)}</span>
        </Row>
        {kind === 'live' && requestMs !== null && (
          <Row label="Computed in">
            <span className="num">{Math.round(requestMs).toLocaleString('en-IN')} ms</span>
          </Row>
        )}
        <button
          type="button"
          onClick={onOpenOptimization}
          className="mt-1 flex w-fit items-center gap-1 rounded text-body-sm text-accent hover:underline"
        >
          Explain in Optimization
          <ArrowRight size={14} strokeWidth={1.75} aria-hidden />
        </button>
      </div>
    </Panel>
  );
}
