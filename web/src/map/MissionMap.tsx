import { memo, useCallback, useEffect, useRef, useState } from 'react';
import 'leaflet/dist/leaflet.css';
import 'leaflet.markercluster/dist/MarkerCluster.css';
import './map.css';
import type { Playback } from '../hooks/usePlayback';
import { useReducedMotion } from '../hooks/useReducedMotion';
import { formatClock } from '../lib/format';
import type { RunData } from '../lib/types';
import { DetailCard } from './DetailCard';
import { MapEngine, type LayerKey, type RouteStyle, type Selection } from './engine';
import { MapControls } from './MapControls';
import { Legend } from './Legend';
import { DEFAULT_LAYERS, type LayerState } from './layers';

interface MissionMapProps {
  run: RunData | null;
  /** Fixed-schedule run on the same recorded traffic, when there is one. */
  baseline: RunData | null;
  isLive: boolean;
  playback: Playback;
  incidentPreview: { lat: number; lon: number } | null;
  onMapClick?: (lat: number, lon: number) => void;
  /** Analytics draws the fixed-schedule map in the baseline style. */
  routeStyle?: RouteStyle;
  /** Compare view shows one shared legend instead of one per map. */
  showLegend?: boolean;
  /** Accessible name of the map region. */
  label?: string;
}

// This wrapper re-renders at display rate (about 4 times a second) for the
// arrival label and detail card. Leaflet layers are built only when the run
// changes; per-frame movement goes through playback.subscribe, not React.
export const MissionMap = memo(function MissionMap({
  run, baseline, isLive, playback, incidentPreview, onMapClick, routeStyle = 'optimized', showLegend = true, label = 'Map of the simulated area',
}: MissionMapProps) {
  const el = useRef<HTMLDivElement>(null);
  const engine = useRef<MapEngine | null>(null);
  const reducedMotion = useReducedMotion();
  const [selection, setSelection] = useState<Selection | null>(null);
  const [following, setFollowing] = useState(true);
  const [dark, setDark] = useState(true);
  const [layers, setLayers] = useState<LayerState>(DEFAULT_LAYERS);
  const clickRef = useRef(onMapClick);
  clickRef.current = onMapClick;

  // Create the map once.
  useEffect(() => {
    if (!el.current) return;
    const e = new MapEngine(el.current, {
      reducedMotion,
      routeStyle,
      onSelect: setSelection,
      onFollowChange: setFollowing,
      onMapClick: (lat, lon) => clickRef.current?.(lat, lon),
    });
    engine.current = e;
    if (import.meta.env.DEV) {
      const stress = Number(new URLSearchParams(window.location.search).get('stress'));
      if (stress > 0) e.addStressPoints(stress);
      (window as unknown as { __emsMap?: MapEngine }).__emsMap = e;
    }
    return () => {
      e.destroy();
      engine.current = null;
    };
    // reducedMotion is applied by the effect below; the map must not be rebuilt for it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => engine.current?.setReducedMotion(reducedMotion), [reducedMotion]);
  useEffect(() => engine.current?.setBaseDark(dark), [dark]);

  // Build layers once per run.
  useEffect(() => {
    engine.current?.setRun(run, baseline);
    (Object.keys(layers) as LayerKey[]).forEach((k) => engine.current?.setLayerVisible(k, layers[k]));
    // Layer visibility is re-applied here so a new run respects the current toggles.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, baseline]);

  useEffect(() => {
    if (!run) engine.current?.setPreviewIncident(incidentPreview);
  }, [incidentPreview, run]);

  // Per-frame updates bypass React.
  const { subscribe } = playback;
  useEffect(() => subscribe((t) => engine.current?.update(t)), [subscribe]);

  // Arrival label on the destination, at display rate.
  const remaining = run ? Math.max(0, run.completion_time - playback.t) : 0;
  const arrived = run !== null && playback.t >= run.completion_time;
  useEffect(() => {
    if (!run) return;
    engine.current?.setEtaLabel(arrived ? `Arrived ${formatClock(run.completion_time)}` : `Arrival in ${formatClock(remaining)}`);
  }, [run, remaining, arrived]);

  const toggleLayer = useCallback((key: LayerKey, on: boolean) => {
    setLayers((prev) => ({ ...prev, [key]: on }));
    engine.current?.setLayerVisible(key, on);
  }, []);

  const closeDetail = useCallback(() => {
    engine.current?.clearSelection();
    setSelection(null);
  }, []);

  return (
    <div className="relative h-full w-full">
      <div ref={el} className="absolute inset-0" aria-label={label} role="region" />

      <MapControls
        onZoomIn={() => engine.current?.zoomIn()}
        onZoomOut={() => engine.current?.zoomOut()}
        onFit={() => engine.current?.recenter()}
        dark={dark}
        onToggleBase={() => setDark((d) => !d)}
        following={following}
        onRecenter={() => engine.current?.recenter()}
      />

      {showLegend && (
        <div className="pointer-events-none absolute bottom-6 right-3 z-map-overlay">
          <Legend layers={layers} onToggle={toggleLayer} hasBaseline={baseline !== null} isLive={isLive} />
        </div>
      )}

      {selection && (
        <div className="pointer-events-none absolute bottom-10 left-3 z-map-overlay">
          <DetailCard
            selection={selection}
            onClose={closeDetail}
            ambulance={{ remaining, progressPct: run ? (Math.min(playback.t, run.completion_time) / run.completion_time) * 100 : 0, arrived }}
          />
        </div>
      )}
    </div>
  );
});
