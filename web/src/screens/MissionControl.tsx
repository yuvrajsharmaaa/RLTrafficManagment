import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Database, PanelLeft, Radio, RotateCcw, TriangleAlert } from 'lucide-react';
import { useConnectionState } from '../app/connection';
import { useNotifications } from '../app/notifications';
import { useSession } from '../app/sessionRun';
import { useReportStatus } from '../app/shellStatus';
import { EmergencyPanel } from '../components/mission/EmergencyPanel';
import { OptimizationDetails } from '../components/mission/OptimizationDetails';
import { RouteDetailsTabs } from '../components/mission/RouteDetailsTabs';
import { TrafficCard } from '../components/mission/TrafficCard';
import { insideSimulatedArea, type LatLon } from '../lib/geo';
import { PlaybackDock } from '../components/mission/PlaybackDock';
import { VehicleCard } from '../components/mission/VehicleCard';
import { Button, Panel, StatusChip } from '../components/ui';
import { usePlayback } from '../hooks/usePlayback';
import { fetchRunData, planRoute, randomSeed, recordedRunId } from '../lib/api';
import { cx } from '../lib/cx';
import { SCENARIO_WORD, SOURCE_WORD, TIER_WORD, formatClock, formatCoord } from '../lib/format';
import { replanEvents, trafficAt } from '../lib/timeline';
import type { RunData, ScenarioTier } from '../lib/types';
import { MissionMap } from '../map/MissionMap';

interface Mission {
  key: string;
  run: RunData;
  baseline: RunData | null;
  kind: 'live' | 'recorded';
  /** Plain-language data source, always shown. */
  source: string;
  incident: LatLon | null;
  tier: ScenarioTier;
  note: string | null;
  requestMs: number | null;
}

export const MissionControl = memo(function MissionControl() {
  const report = useReportStatus();
  const { push } = useNotifications();
  const connection = useConnectionState();
  const session = useSession();
  const restored = session.current;

  const [incident, setIncident] = useState<LatLon | null>(null);
  const [tier, setTier] = useState<ScenarioTier>(restored?.tier ?? 'medium');
  // Coming back from another screen restores the mission instead of losing it.
  const [mission, setMission] = useState<Mission | null>(() =>
    restored
      ? {
          key: `restored-${restored.kind}-${restored.run.completion_time}`,
          run: restored.run,
          baseline: restored.baseline,
          kind: restored.kind,
          source: restored.kind === 'recorded' ? 'Recorded run' : ((restored.run.source && SOURCE_WORD[restored.run.source]) ?? 'Live result'),
          incident: null,
          tier: restored.tier,
          note: null,
          requestMs: restored.requestMs,
        }
      : null,
  );
  const [computing, setComputing] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [panelOpen, setPanelOpen] = useState(true);

  const run = mission?.run ?? null;
  const playback = usePlayback(run?.completion_time ?? 0, mission?.key ?? 'none');
  const { play } = playback;

  // Start playback once the new run's clock exists (a new mission resets it first).
  const pendingPlay = useRef(false);
  useEffect(() => {
    if (pendingPlay.current && mission) {
      pendingPlay.current = false;
      play();
    }
  }, [mission, play]);

  const blockedReason =
    connection.state === 'offline'
      ? 'Offline: live dispatch needs the server. Recorded incidents still work.'
      : connection.state === 'server-unreachable'
        ? 'Server not reachable: live dispatch is unavailable. Recorded incidents still work.'
        : null;

  // Elapsed time while a route request is in flight.
  useEffect(() => {
    if (!computing) return;
    const started = performance.now();
    const id = window.setInterval(() => setElapsedMs(performance.now() - started), 100);
    return () => window.clearInterval(id);
  }, [computing]);

  const placeIncident = useCallback(
    (p: LatLon | null) => {
      if (computing || mission) return;
      setIncident(p);
      setError(null);
      if (p && !insideSimulatedArea(p)) push('warning', `Incident placed outside the simulated area (${formatCoord(p.lat, p.lon)}).`);
    },
    [computing, mission, push],
  );

  const onMapClick = useCallback((lat: number, lon: number) => placeIncident({ lat, lon }), [placeIncident]);

  const findRoute = async () => {
    if (!incident) return;
    setComputing(true);
    setElapsedMs(0);
    setError(null);
    const started = performance.now();
    try {
      const data = await planRoute({
        incident_lat: incident.lat,
        incident_lon: incident.lon,
        scenario_tier: tier,
        seed: randomSeed(),
        use_live_sumo: true,
        num_stops: 8,
      });
      const source = (data.source && SOURCE_WORD[data.source]) ?? 'Live result';
      const requestMs = performance.now() - started;
      setMission({ key: `live-${Date.now()}`, run: data, baseline: null, kind: 'live', source, incident, tier, note: null, requestMs });
      session.setCurrent({ run: data, baseline: null, kind: 'live', tier, requestMs });
      if (data.source === 'live_calibrated_scenario') {
        push('warning', 'Traffic simulation was unavailable, so this route uses estimated traffic. Treat times as approximate.');
      }
      pendingPlay.current = true;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      push('danger', `Route request failed: ${message}`);
    } finally {
      setComputing(false);
    }
  };

  const openRecorded = async () => {
    setError(null);
    try {
      const [adaptive, fixed] = await Promise.all([
        fetchRunData(recordedRunId(tier, 'va_qpso')),
        fetchRunData(recordedRunId(tier, 'fixed_beta_qpso')).catch(() => null),
      ]);
      const note = adaptive.isDefaultFallback
        ? `The ${SCENARIO_WORD[tier]} traffic run could not be loaded. Showing the default run (Moderate traffic, adaptive routing).`
        : null;
      if (note) push('warning', note);
      setMission({
        key: `recorded-${tier}-${Date.now()}`,
        run: adaptive.data,
        baseline: fixed && !fixed.isDefaultFallback && !adaptive.isDefaultFallback ? fixed.data : null,
        kind: 'recorded',
        source: 'Recorded run',
        incident: null,
        tier: adaptive.isDefaultFallback ? 'medium' : tier,
        note,
        requestMs: null,
      });
      session.setCurrent({
        run: adaptive.data,
        baseline: fixed && !fixed.isDefaultFallback && !adaptive.isDefaultFallback ? fixed.data : null,
        kind: 'recorded',
        tier: adaptive.isDefaultFallback ? 'medium' : tier,
        requestMs: null,
      });
      setIncident(null);
      pendingPlay.current = true;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      push('danger', `Recorded run could not be loaded: ${message}`);
    }
  };

  const newIncident = () => {
    session.setCurrent(null);
    setMission(null);
    setIncident(null);
    setError(null);
  };

  // Notify when playback passes a re-plan.
  const lastT = useRef(0);
  const replans = useMemo(() => (run ? replanEvents(run.events) : []), [run]);
  useEffect(() => {
    const prev = lastT.current;
    lastT.current = playback.t;
    if (!playback.playing || playback.t < prev) return;
    replans
      .filter((e) => prev < e.t && e.t <= playback.t)
      .forEach((e) => {
        const word = run ? trafficAt(run, e.t) : null;
        const traffic = word ? ` Traffic ${TIER_WORD[word.tier].toLowerCase()}.` : '';
        const saved =
          typeof e.eta_before === 'number' && typeof e.eta_after === 'number' && e.eta_before > e.eta_after
            ? ` Arrival ${Math.round(e.eta_before - e.eta_after)} s sooner.`
            : '';
        push('info', `T+${Math.round(e.t)} s: route re-planned.${traffic}${saved}`);
      });
  }, [playback.t, playback.playing, replans, push, run]);

  // Global status bar.
  const metric = run ? trafficAt(run, playback.t) : null;
  useEffect(() => {
    report({
      systemState: computing ? 'Computing route' : error && !mission ? 'Error' : mission ? (playback.playing ? 'Playing' : 'Paused') : 'Ready',
      dataSource: mission?.source,
      traffic: metric ? { word: TIER_WORD[metric.tier], v: metric.volatility_index } : undefined,
      clock: run ? { t: playback.t, total: run.completion_time, playing: playback.playing, speed: playback.speed } : undefined,
    });
  }, [report, computing, error, mission, run, metric, playback.t, playback.playing, playback.speed]);

  const sourceChip = mission ? (
    mission.kind === 'recorded' ? (
      <StatusChip tone="neutral" icon={Database} label="Recorded run" />
    ) : mission.run.source === 'live_calibrated_scenario' ? (
      <StatusChip tone="warning" icon={TriangleAlert} label="Estimated traffic" />
    ) : (
      <StatusChip tone="info" icon={Radio} label={mission.source} />
    )
  ) : null;

  return (
    <div
      className={cx(
        'relative grid h-full min-h-0',
        mission
          ? 'grid-rows-[1fr_var(--shell-dock-h)] [grid-template-areas:"map"_"dock"] desktop:grid-cols-[var(--shell-panel-w)_1fr_var(--shell-panel-w)] desktop:[grid-template-areas:"left_map_right"_"left_dock_right"]'
          : 'grid-rows-[1fr] [grid-template-areas:"map"] desktop:grid-cols-[var(--shell-panel-w)_1fr] desktop:[grid-template-areas:"left_map"]',
      )}
    >
      <aside
        aria-label="Mission panel"
        className={cx(
          'z-panel min-h-0 flex-col gap-4 overflow-y-auto border-r border-border bg-panel p-4 [grid-area:left]',
          'absolute inset-y-0 left-0 w-[360px] shadow-2 desktop:static desktop:w-auto desktop:shadow-none',
          panelOpen ? 'flex' : 'hidden desktop:flex',
        )}
      >
        {mission ? (
          <>
            <VehicleCard run={mission.run} t={playback.t} />
            <Panel raised title="Incident">
              <div className="flex flex-col gap-2 text-body-sm">
                <div className="flex flex-wrap gap-2">
                  {sourceChip}
                  <StatusChip tone="neutral" label={`${SCENARIO_WORD[mission.tier]} traffic`} />
                </div>
                {mission.note && <p className="text-caption text-warning">{mission.note}</p>}
                <p className="num text-text-2">
                  {mission.incident
                    ? formatCoord(mission.incident.lat, mission.incident.lon)
                    : mission.run.stops[0]
                      ? formatCoord(mission.run.stops[0].lat, mission.run.stops[0].lon)
                      : null}
                </p>
                <Button size="compact" icon={RotateCcw} onClick={newIncident} className="w-fit">
                  New incident
                </Button>
              </div>
            </Panel>
            <TrafficCard run={mission.run} t={playback.t} />
            <OptimizationDetails
              run={mission.run}
              kind={mission.kind}
              t={playback.t}
              requestMs={mission.requestMs}
              onOpenOptimization={() => session.navigate('optimization')}
            />
            {/* Below desktop the right panel folds into this drawer. */}
            <div className="desktop:hidden">
              <RouteDetailsTabs run={mission.run} kind={mission.kind} baseline={mission.baseline} t={playback.t} onSeek={playback.seek} />
            </div>
          </>
        ) : (
          <EmergencyPanel
            incident={incident}
            onIncidentChange={placeIncident}
            tier={tier}
            onTierChange={setTier}
            onFindRoute={() => void findRoute()}
            onOpenRecorded={() => void openRecorded()}
            computing={computing}
            elapsedMs={elapsedMs}
            error={error}
            blockedReason={blockedReason}
          />
        )}
      </aside>

      <div className="relative min-h-0 [grid-area:map]">
        <MissionMap
          run={run}
          baseline={mission?.baseline ?? null}
          isLive={mission?.kind === 'live'}
          playback={playback}
          incidentPreview={incident}
          onMapClick={mission ? undefined : onMapClick}
        />
        {/* Tablet: the panel is a drawer; the answer stays pinned on the map. */}
        <div className="absolute left-3 top-3 z-map-overlay flex items-center gap-2 desktop:hidden">
          <Button size="compact" icon={PanelLeft} aria-expanded={panelOpen} onClick={() => setPanelOpen((o) => !o)}>
            Mission panel
          </Button>
          {run && !panelOpen && (
            <span className="num rounded border border-border bg-panel px-2 py-1 text-body-sm text-text-1 shadow-1">
              {playback.t >= run.completion_time ? 'Arrived' : `Arrival in ${formatClock(run.completion_time - playback.t)}`}
            </span>
          )}
        </div>
      </div>

      {mission && (
        <aside
          aria-label="Route details"
          className="hidden min-h-0 flex-col gap-4 overflow-y-auto border-l border-border bg-panel p-4 [grid-area:right] desktop:flex"
        >
          <RouteDetailsTabs run={mission.run} kind={mission.kind} baseline={mission.baseline} t={playback.t} onSeek={playback.seek} />
        </aside>
      )}

      {mission && <PlaybackDock playback={playback} markers={replans.map((e) => e.t)} />}
    </div>
  );
});
