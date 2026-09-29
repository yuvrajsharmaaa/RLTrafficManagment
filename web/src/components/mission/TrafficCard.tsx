import { useMemo } from 'react';
import { Panel, Tooltip } from '../ui';
import { TimeSeriesChart } from '../charts/TimeSeriesChart';
import { replanEvents, trafficAt } from '../../lib/timeline';
import type { RunData } from '../../lib/types';
import { TierLabel } from './TierLabel';

// Tier boundaries used by export_for_frontend.py tier_for(), which server.py also calls.
const CHANGING_FROM = 0.25;
const UNSTABLE_FROM = 0.5;

interface TrafficCardProps {
  run: RunData;
  t: number;
}

export function TrafficCard({ run, t }: TrafficCardProps) {
  const duration = run.completion_time;
  const metric = trafficAt(run, t);
  const single = run.metrics_over_time.length <= 1;

  const { series, markers } = useMemo(() => {
    // Every sample is a measurement. A planner estimate has one (at dispatch), drawn as a single point.
    const pts = run.metrics_over_time.filter((m) => m.t <= duration).map((m) => ({ t: m.t, v: m.volatility_index }));
    return {
      series: [{ key: 'v', label: 'Traffic unpredictability (V)', color: 'var(--color-text-2)', points: pts }],
      markers: replanEvents(run.events).map((e) => ({ t: e.t, label: `Re-plan at T+${Math.round(e.t)} s` })),
    };
  }, [run, duration]);

  if (!metric) {
    return (
      <Panel raised title="Traffic now">
        <p className="text-body-sm text-text-3">No traffic data in this run.</p>
      </Panel>
    );
  }

  return (
    <Panel raised title="Traffic now">
      <div className="flex flex-col gap-2">
        {metric.vehicles !== undefined && (
          <dl className="grid grid-cols-3 gap-2 text-body-sm" aria-label="Congestion now">
            <div>
              <dt className="text-caption text-text-3">Vehicles</dt>
              <dd className="num text-text-1">{metric.vehicles}</dd>
            </div>
            <div>
              <dt className="text-caption text-text-3">Mean speed</dt>
              <dd className="num text-text-1">{metric.mean_vehicle_speed_kmh != null ? `${metric.mean_vehicle_speed_kmh.toFixed(1)} km/h` : 'No data'}</dd>
            </div>
            <div>
              <dt className="text-caption text-text-3">Stopped</dt>
              <dd className="num text-text-1">{metric.stopped_vehicles ?? 'No data'}</dd>
            </div>
          </dl>
        )}
        <p className="flex items-baseline gap-2 text-body">
          <span className="text-body-sm text-text-2">Unpredictability</span>
          <TierLabel tier={metric.tier} />
          <Tooltip content="How much network speeds are changing (volatility index V, rolling speed variance, 0 to 1). It is not a congestion level: a fully jammed network whose speeds stay low reads as steady.">
            <span tabIndex={0} className="num text-text-2 underline decoration-dotted underline-offset-2">
              {metric.volatility_index.toFixed(2)}
            </span>
          </Tooltip>
        </p>

        <TimeSeriesChart
          compact
          title="Unpredictability over the trip"
          series={series}
          xMax={duration}
          yMin={0}
          yMax={1}
          yTicks={[0, 0.5, 1]}
          yLabel="V"
          formatY={(v) => v.toFixed(1)}
          references={[{ v: CHANGING_FROM, label: 'Changing' }, { v: UNSTABLE_FROM, label: 'Unstable' }]}
          markers={markers}
          cursor={t}
          height={104}
        />
        <p className="text-caption text-text-3">
          {single
            ? 'Measured once in SUMO at dispatch; no vehicle was driven, so there is no reading during the trip. Network-wide, not a single road.'
            : 'Measured in SUMO every simulated second. Amber lines mark route re-plans. Network-wide, not a single road.'}
        </p>

        <Panel title="Details" collapsible defaultOpen={false} className="-mx-4 -mb-4 border-x-0 border-b-0 bg-transparent">
          <div className="flex flex-col gap-1 text-body-sm">
            <div className="flex justify-between">
              <Tooltip content="Search breadth (β): lowest value each search contracts to. Higher means the route search keeps exploring more alternative streets.">
                <span tabIndex={0} className="text-text-2 underline decoration-dotted underline-offset-2">Search-breadth floor now</span>
              </Tooltip>
              <span className="num text-text-1">{metric.beta.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-2">Level boundaries</span>
              <span className="num text-text-1">Changing {CHANGING_FROM.toFixed(2)} · Unstable {UNSTABLE_FROM.toFixed(2)}</span>
            </div>
          </div>
        </Panel>
      </div>
    </Panel>
  );
}
