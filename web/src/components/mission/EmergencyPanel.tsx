import { useId, useState } from 'react';
import { History, MapPin, Navigation } from 'lucide-react';
import { Button, EmptyState, ErrorState } from '../ui';
import { insideSimulatedArea, type LatLon } from '../../lib/geo';
import { SCENARIO_WORD, formatCoord } from '../../lib/format';
import type { ScenarioTier } from '../../lib/types';
import { cx } from '../../lib/cx';

// Two real places, taken from the old app (its default incident point) and
// from the recorded runs (their pickup junction).
const PRESETS: Array<{ label: string; at: LatLon }> = [
  { label: 'Connaught Place centre', at: { lat: 28.6325, lon: 77.2215 } },
  { label: 'Recorded pickup junction', at: { lat: 28.632587, lon: 77.222985 } },
];

const TIERS: ScenarioTier[] = ['low', 'medium', 'high'];

function parseCoords(text: string): LatLon | null {
  const m = /^\s*(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)\s*$/.exec(text);
  if (!m) return null;
  const lat = Number(m[1]);
  const lon = Number(m[2]);
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { lat, lon };
}

interface EmergencyPanelProps {
  incident: LatLon | null;
  onIncidentChange: (p: LatLon | null) => void;
  tier: ScenarioTier;
  onTierChange: (t: ScenarioTier) => void;
  onFindRoute: () => void;
  /** Drive an ambulance through SUMO (exact, slower) instead of returning the planner estimate. */
  driveThrough: boolean;
  onDriveThroughChange: (on: boolean) => void;
  onOpenRecorded: () => void;
  computing: boolean;
  elapsedMs: number;
  error: string | null;
  /** Why live dispatch can't run right now (offline, server down), or null. */
  blockedReason: string | null;
}

export function EmergencyPanel({
  incident, onIncidentChange, tier, onTierChange, onFindRoute, driveThrough, onDriveThroughChange, onOpenRecorded, computing,
  elapsedMs, error, blockedReason,
}: EmergencyPanelProps) {
  const inputId = useId();
  const helpId = useId();
  const [text, setText] = useState(incident ? formatCoord(incident.lat, incident.lon) : '');
  const [invalid, setInvalid] = useState(false);

  // Map clicks and presets update the field (adjusting state when a prop changes, during render).
  const [shownIncident, setShownIncident] = useState(incident);
  if (incident !== shownIncident) {
    setShownIncident(incident);
    setText(incident ? formatCoord(incident.lat, incident.lon) : '');
    setInvalid(false);
  }

  const commit = () => {
    if (text.trim() === '') {
      onIncidentChange(null);
      return;
    }
    const p = parseCoords(text);
    setInvalid(!p);
    if (p) onIncidentChange(p);
  };

  const outside = incident !== null && !insideSimulatedArea(incident);
  const canFind = incident !== null && !computing && blockedReason === null;

  return (
    <div className="flex flex-col gap-4">
      {!incident && (
        <EmptyState icon={MapPin} title="No active incident" message="Click the map inside the simulated area, or type coordinates, to begin." />
      )}

      <div className="flex flex-col gap-1">
        <label htmlFor={inputId} className="text-label text-text-2">Location (latitude, longitude)</label>
        <input
          id={inputId}
          value={text}
          disabled={computing}
          onChange={(e) => setText(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => e.key === 'Enter' && commit()}
          placeholder="28.63250, 77.22150"
          aria-invalid={invalid || undefined}
          aria-describedby={helpId}
          className={cx(
            'num h-8 rounded border bg-raised px-2 text-body text-text-1 placeholder:text-text-3',
            invalid ? 'border-danger' : 'border-border-control',
          )}
        />
        <p id={helpId} className={cx('text-caption', invalid ? 'text-danger' : outside ? 'text-warning' : 'text-text-3')}>
          {invalid
            ? 'Enter two numbers separated by a comma.'
            : outside
              ? 'Outside the simulated area. The route will start from the nearest network junction.'
              : 'Or choose a known place:'}
        </p>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <Button key={p.label} size="compact" variant="ghost" disabled={computing} onClick={() => onIncidentChange(p.at)}>
              {p.label}
            </Button>
          ))}
        </div>
      </div>

      <fieldset className="flex flex-col gap-1">
        <legend className="pb-1 text-label text-text-2">Traffic level</legend>
        <div className="flex overflow-hidden rounded border border-border-control" role="group">
          {TIERS.map((t) => (
            <button
              key={t}
              type="button"
              aria-pressed={tier === t}
              disabled={computing}
              onClick={() => onTierChange(t)}
              className={cx(
                'h-8 flex-1 border-r border-border-subtle text-body-sm last:border-r-0',
                tier === t ? 'bg-accent-weak text-text-1' : 'text-text-2 hover:bg-raised hover:text-text-1',
              )}
            >
              {SCENARIO_WORD[t]}
            </button>
          ))}
        </div>
      </fieldset>

      <label className="flex items-start gap-2 text-body-sm text-text-1">
        <input
          type="checkbox"
          className="mt-0.5"
          checked={driveThrough}
          disabled={computing}
          onChange={(e) => onDriveThroughChange(e.target.checked)}
        />
        <span className="flex flex-col">
          Simulate the drive in SUMO
          <span className="text-caption text-text-3">
            Exact simulated time, up to a 15-minute limit; takes seconds to a minute. Off: planner estimate from the
            traffic measured at dispatch.
          </span>
        </span>
      </label>

      <Button
        variant="primary"
        size="large"
        icon={Navigation}
        loading={computing}
        loadingLabel={`${driveThrough ? 'Simulating drive' : 'Finding route'} · ${(elapsedMs / 1000).toFixed(1)} s`}
        disabled={!canFind && !computing}
        onClick={onFindRoute}
      >
        Find route
      </Button>
      {blockedReason && <p className="text-caption text-text-3">{blockedReason}</p>}

      {error && (
        <ErrorState
          title="Route request failed"
          message={`${error}. Recorded runs still work.`}
          onRetry={canFind ? onFindRoute : undefined}
          secondaryAction={
            <Button size="compact" icon={History} onClick={onOpenRecorded}>
              Open recorded incident
            </Button>
          }
        />
      )}

      {!error && (
        <div className="border-t border-border-subtle pt-3">
          <Button variant="ghost" size="compact" icon={History} disabled={computing} onClick={onOpenRecorded}>
            Open recorded incident ({SCENARIO_WORD[tier]} traffic)
          </Button>
        </div>
      )}
    </div>
  );
}
