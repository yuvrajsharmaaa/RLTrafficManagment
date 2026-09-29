import L from 'leaflet';
import 'leaflet.markercluster';
import type { DecisionEvent, Hospital, MetricPoint, RunData, Stop, TierKey } from '../lib/types';
import { bearing, positionAt, replanEvents, segmentIndexAt, trafficAt } from '../lib/timeline';
import { DELHI_BBOX, MAP_CENTER } from '../lib/geo';
import { chevronIcon, clusterIcon, markerIcon, type MarkerKind } from './icons';

// Imperative owner of the Leaflet map. React creates it once and sends it
// commands; per-frame playback updates go straight to Leaflet objects, so a
// React state change never rebuilds or re-renders the map.

export { DELHI_BBOX, MAP_CENTER };

const OSM_TILE_URL = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const OSM_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors';

export type LayerKey =
  | 'route' | 'baseline' | 'direction' | 'traffic' | 'heat' | 'hospitals' | 'checkpoints' | 'replans' | 'area';

export type Selection =
  | { kind: 'ambulance' }
  | { kind: 'incident'; stop: Stop | null; lat: number; lon: number }
  | { kind: 'destination'; hospital: Hospital; candidates: Hospital[] }
  | { kind: 'hospital'; hospital: Hospital; candidates: Hospital[] }
  | { kind: 'checkpoint'; stop: Stop; index: number }
  | { kind: 'replan'; event: DecisionEvent }
  | { kind: 'segment'; t0: number; t1: number; tier: TierKey; v: number }
  | { kind: 'baseline'; completion: number; arrived: boolean };

export type RouteStyle = 'optimized' | 'baseline';

interface EngineOptions {
  reducedMotion: boolean;
  /** 'baseline' draws this map's own route in the fixed-schedule style (Analytics). */
  routeStyle: RouteStyle;
  onSelect: (sel: Selection | null) => void;
  onFollowChange: (following: boolean) => void;
  onMapClick: (lat: number, lon: number) => void;
}

const TIER_COLOR: Record<TierKey, string> = {
  calm: 'var(--traffic-free)',
  moderate: 'var(--traffic-moderate)',
  turbulent: 'var(--traffic-heavy)',
};
const TIER_WEIGHT: Record<TierKey, number> = { calm: 7, moderate: 9, turbulent: 11 };
const HEAT = ['var(--heat-1)', 'var(--heat-2)', 'var(--heat-3)', 'var(--heat-4)', 'var(--heat-5)'];

// Canvas paths don't resolve CSS variables, so read the token values once.
function token(value: string): string {
  const m = /^var\((--[^)]+)\)$/.exec(value);
  return m?.[1] ? getComputedStyle(document.documentElement).getPropertyValue(m[1]).trim() || value : value;
}

const REROUTE_WINDOW_S = 3;
const POLYLINE_MIN_INTERVAL_MS = 33; // at most ~30 redraws of the progress line per second
const FOLLOW_CHECK_MS = 500;
const CHEVRON_SPACING_M = 120;

export class MapEngine {
  readonly map: L.Map;
  private readonly renderer: L.Canvas;
  private readonly tiles: L.TileLayer;
  private readonly opts: EngineOptions;
  private readonly groups: Record<LayerKey, L.LayerGroup>;
  private readonly always: L.LayerGroup;

  private run: RunData | null = null;
  private replans: DecisionEvent[] = [];
  private vehicle: L.Marker | null = null;
  private routeEnd: L.Marker | null = null;
  private travelled: L.Polyline | null = null;
  private remaining: L.Polyline | null = null;
  private chevrons: Array<{ marker: L.Marker; t: number }> = [];
  private previewIncident: L.Marker | null = null;
  private incident: L.Marker | null = null;
  private selectedEl: HTMLElement | null = null;

  private following = true;
  private programmaticMoves = 0;
  private lastT = 0;
  private lastPolylineUpdate = 0;
  private lastFollowCheck = 0;
  private arrivedHandled = false;

  constructor(el: HTMLElement, opts: EngineOptions) {
    this.opts = opts;
    this.map = L.map(el, { zoomControl: false, attributionControl: true, keyboard: true }).setView(MAP_CENTER, 15);
    this.renderer = L.canvas({ padding: 0.5 });
    this.tiles = L.tileLayer(OSM_TILE_URL, { maxZoom: 19, attribution: OSM_ATTRIBUTION, className: 'ems-tiles-dark' });
    this.tiles.addTo(this.map);
    L.control.scale({ imperial: false, position: 'bottomleft', maxWidth: 120 }).addTo(this.map);

    const cluster = () =>
      L.markerClusterGroup({
        iconCreateFunction: (c) => clusterIcon(c.getChildCount()),
        showCoverageOnHover: false,
        maxClusterRadius: 36,
        disableClusteringAtZoom: 15,
      });

    this.groups = {
      area: L.layerGroup(),
      heat: L.layerGroup(),
      traffic: L.layerGroup(),
      baseline: L.layerGroup(),
      route: L.layerGroup(),
      direction: L.layerGroup(),
      replans: L.layerGroup(),
      checkpoints: cluster(),
      hospitals: cluster(),
    };
    this.always = L.layerGroup().addTo(this.map);
    (Object.keys(this.groups) as LayerKey[]).forEach((k) => {
      if (k !== 'heat') this.groups[k].addTo(this.map);
    });

    L.rectangle(DELHI_BBOX, {
      renderer: this.renderer,
      color: token('var(--color-border-strong)'),
      weight: 1,
      dashArray: '4 4',
      fill: false,
      interactive: false,
    }).addTo(this.groups.area);

    this.map.on('click', (e: L.LeafletMouseEvent) => {
      this.select(null, null);
      opts.onMapClick(e.latlng.lat, e.latlng.lng);
    });
    // An interrupted animation never fires its own moveend, so a counter could
    // stay above zero forever; any completed move clears the flag instead.
    this.map.on('moveend', () => {
      this.programmaticMoves = 0;
    });
    // Any direct manipulation by the user stops camera automation until Recenter.
    const takeOver = () => this.setFollowing(false);
    this.map.on('dragstart', takeOver);
    this.map.on('boxzoomstart', takeOver);
    el.addEventListener('wheel', takeOver, { passive: true });
    el.addEventListener('dblclick', takeOver);
    el.addEventListener('keydown', (e) => {
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', '+', '-', '='].includes(e.key)) takeOver();
    });
  }

  destroy() {
    this.map.remove();
  }

  setReducedMotion(reduced: boolean) {
    this.opts.reducedMotion = reduced;
  }

  setBaseDark(dark: boolean) {
    const c = this.tiles.getContainer();
    if (c) c.classList.toggle('ems-tiles-dark', dark);
  }

  setLayerVisible(key: LayerKey, visible: boolean) {
    const g = this.groups[key];
    if (visible && !this.map.hasLayer(g)) g.addTo(this.map);
    if (!visible && this.map.hasLayer(g)) this.map.removeLayer(g);
  }

  zoomIn() {
    this.setFollowing(false);
    this.map.zoomIn();
  }

  zoomOut() {
    this.setFollowing(false);
    this.map.zoomOut();
  }

  /** Re-enables camera automation and frames the route (or the simulated area). */
  recenter() {
    this.setFollowing(true);
    this.frame(this.run ? L.latLngBounds(this.run.path.map((p) => [p.lat, p.lon])) : L.latLngBounds(DELHI_BBOX));
  }

  /** Incident chosen on the map before a route exists. */
  setPreviewIncident(point: { lat: number; lon: number } | null) {
    if (this.previewIncident) {
      this.previewIncident.remove();
      this.previewIncident = null;
    }
    if (!point) return;
    this.previewIncident = this.marker('incident', [point.lat, point.lon], 'Incident location (not dispatched)', {
      kind: 'incident',
      stop: null,
      lat: point.lat,
      lon: point.lon,
    }, 1200).addTo(this.always);
    this.previewIncident.getElement()?.querySelector('.ems-mk')?.classList.add('ems-pulse-on');
  }

  /** Builds every layer for a run once. Playback afterwards only calls update(). */
  setRun(run: RunData | null, baseline: RunData | null) {
    (Object.keys(this.groups) as LayerKey[]).forEach((k) => {
      if (k !== 'area') this.groups[k].clearLayers();
    });
    this.always.clearLayers();
    this.previewIncident = null;
    this.incident = null;
    this.vehicle = null;
    this.routeEnd = null;
    this.travelled = null;
    this.remaining = null;
    this.chevrons = [];
    this.select(null, null);
    this.run = run;
    this.arrivedHandled = false;
    this.lastT = 0;
    if (!run || run.path.length === 0) return;

    const latlngs = run.path.map((p) => L.latLng(p.lat, p.lon));
    this.replans = replanEvents(run.events);

    this.buildTraffic(run);
    this.buildHeat(run);

    if (baseline && baseline.path.length > 0) {
      const line = L.polyline(
        baseline.path.map((p) => [p.lat, p.lon]),
        {
          renderer: this.renderer,
          color: token('var(--route-baseline)'),
          weight: 3,
          dashArray: '10 6',
          opacity: 0.9,
          bubblingMouseEvents: false,
        },
      );
      this.hoverable(line, 3);
      line.on('click', () => this.select({ kind: 'baseline', completion: baseline.completion_time, arrived: baseline.timing ? baseline.timing.status === 'arrived' : true }, null));
      line.addTo(this.groups.baseline);
    }

    // This map's route: dark casing, lighter remaining part, solid travelled part.
    // Adaptive = accent solid; fixed schedule = baseline grey, dashed (Phase 2 route types).
    const isBaseline = this.opts.routeStyle === 'baseline';
    const routeColor = token(isBaseline ? 'var(--route-baseline)' : 'var(--route-optimized)');
    const dash = isBaseline ? '10 6' : undefined;
    const casing = L.polyline(latlngs, {
      renderer: this.renderer, color: token('var(--color-bg)'), weight: 8, opacity: 0.9, interactive: false,
    });
    this.remaining = L.polyline(latlngs, {
      renderer: this.renderer, color: routeColor, weight: 4, opacity: 0.55, dashArray: dash, bubblingMouseEvents: false,
    });
    this.travelled = L.polyline([latlngs[0] ?? L.latLng(MAP_CENTER)], {
      renderer: this.renderer, color: routeColor, weight: 5, opacity: 1, dashArray: dash, interactive: false,
    });
    this.hoverable(this.remaining, 4);
    this.groups.route.addLayer(casing).addLayer(this.remaining).addLayer(this.travelled);

    this.buildChevrons(run);

    this.replans.forEach((ev) => {
      const pos = positionAt(run.path, ev.t);
      if (!pos) return;
      this.marker('replan', pos, `Route re-planned at T+${Math.round(ev.t)} s`, { kind: 'replan', event: ev }, 700).addTo(
        this.groups.replans,
      );
    });

    const pickup = run.stops[0];
    const dest = run.stops.at(-1);
    if (pickup) {
      this.incident = this.marker('incident', [pickup.lat, pickup.lon], 'Incident and patient pickup', {
        kind: 'incident', stop: pickup, lat: pickup.lat, lon: pickup.lon,
      }, 900).addTo(this.always);
    }

    run.stops.slice(1, -1).forEach((stop, i) => {
      this.marker('checkpoint', [stop.lat, stop.lon], `Planner waypoint ${i + 1}`, { kind: 'checkpoint', stop, index: i + 1 }, 400, String(i + 1)).addTo(
        this.groups.checkpoints,
      );
    });

    const hospital = run.selected_hospital;
    const destLatLng: L.LatLngTuple = hospital ? [hospital.lat, hospital.lon] : dest ? [dest.lat, dest.lon] : [latlngs.at(-1)?.lat ?? 0, latlngs.at(-1)?.lng ?? 0];
    if (hospital) {
      this.marker('destination', destLatLng, `Destination: ${hospital.name}`, {
        kind: 'destination', hospital, candidates: run.hospital_candidates,
      }, 950).addTo(this.always);
    }
    // No hospital lies on the simulated network, so the label sits on the route's
    // last point: the network exit toward the hospital.
    const end = latlngs.at(-1);
    if (end) {
      this.routeEnd = L.marker(end, { icon: markerIcon('routeEnd'), interactive: false, keyboard: false, zIndexOffset: 800 }).addTo(this.always);
      this.routeEnd.bindTooltip('', { permanent: true, direction: 'right', offset: [10, 0], className: 'ems-eta' });
    }
    run.hospital_candidates
      .filter((h) => h.name !== hospital?.name)
      .forEach((h) => {
        this.marker('hospital', [h.lat, h.lon], `Other hospital: ${h.name}`, { kind: 'hospital', hospital: h, candidates: run.hospital_candidates }, 500).addTo(
          this.groups.hospitals,
        );
      });

    this.vehicle = this.marker('ambulance', [latlngs[0]?.lat ?? 0, latlngs[0]?.lng ?? 0], 'Ambulance', { kind: 'ambulance' }, 1000).addTo(this.always);

    this.setFollowing(true);
    this.frame(L.latLngBounds(latlngs));
    this.update(0, true);
  }

  /** Per-frame update from the playback clock. */
  update(t: number, force = false) {
    const run = this.run;
    if (!run || !this.vehicle) return;
    const duration = run.completion_time;
    const pos = positionAt(run.path, t);
    if (!pos) return;
    this.vehicle.setLatLng(pos);

    const inner = this.vehicle.getElement()?.querySelector<HTMLElement>('.ems-mk');
    if (inner) {
      const i = segmentIndexAt(run.path, t);
      const a = run.path[i];
      const b = run.path[Math.min(run.path.length - 1, i + 1)];
      if (a && b && (a.lat !== b.lat || a.lon !== b.lon)) {
        const heading = inner.querySelector<HTMLElement>('[data-heading]');
        if (heading) heading.style.transform = `rotate(${bearing([a.lat, a.lon], [b.lat, b.lon])}deg)`;
      }
      const arrived = t >= duration;
      const rerouting = this.replans.some((e) => t >= e.t && t - e.t < REROUTE_WINDOW_S);
      inner.dataset.state = arrived ? 'arrived' : rerouting ? 'rerouting' : 'en-route';
      // The incident pulses only while the mission is under way.
      this.incident?.getElement()?.querySelector('.ems-mk')?.classList.toggle('ems-pulse-on', !arrived);
    }

    const now = performance.now();
    if (force || now - this.lastPolylineUpdate >= POLYLINE_MIN_INTERVAL_MS) {
      this.lastPolylineUpdate = now;
      const i = segmentIndexAt(run.path, t);
      const done = run.path.slice(0, i + 1).map((p) => L.latLng(p.lat, p.lon));
      done.push(L.latLng(pos[0], pos[1]));
      this.travelled?.setLatLngs(done);
      this.chevrons.forEach((c) => {
        c.marker.getElement()?.classList.toggle('ems-chevron-passed', c.t <= t);
      });
    }

    // Camera: frame the new route after a re-plan, keep the ambulance in view,
    // frame the whole route on arrival. Only while the user hasn't taken over.
    if (this.following) {
      const crossed = this.replans.find((e) => this.lastT < e.t && e.t <= t);
      if (crossed && t - this.lastT < 2) {
        const idx = segmentIndexAt(run.path, crossed.t);
        const ahead = run.path.slice(idx).map((p) => L.latLng(p.lat, p.lon));
        if (ahead.length > 1) this.frame(L.latLngBounds(ahead), 17);
      } else if (t >= duration && !this.arrivedHandled) {
        this.arrivedHandled = true;
        this.frame(L.latLngBounds(run.path.map((p) => [p.lat, p.lon])));
      } else if (now - this.lastFollowCheck >= FOLLOW_CHECK_MS && this.programmaticMoves === 0) {
        this.lastFollowCheck = now;
        const view = this.map.getBounds().pad(-0.15);
        if (!view.contains(pos)) this.move(() => this.map.panTo(pos, { animate: !this.opts.reducedMotion }));
      }
    }
    if (t < duration) this.arrivedHandled = false;
    this.lastT = t;
  }

  /** Clears the selected-marker halo (the detail card was closed). */
  clearSelection() {
    this.selectedEl?.classList.remove('ems-selected');
    this.selectedEl = null;
  }

  setEtaLabel(text: string) {
    this.routeEnd?.setTooltipContent(text);
  }

  /** Dev-only stress layer: n canvas points inside the simulated area. Returns the count drawn. */
  addStressPoints(n: number): number {
    const [[s, w], [nLat, e]] = DELHI_BBOX;
    const g = L.layerGroup();
    for (let i = 0; i < n; i++) {
      L.circleMarker([s + Math.random() * (nLat - s), w + Math.random() * (e - w)], {
        renderer: this.renderer, radius: 3, weight: 0, fillOpacity: 0.6, fillColor: token('var(--color-text-3)'), interactive: false,
      }).addTo(g);
    }
    g.addTo(this.map);
    return n;
  }

  // ---- internals -------------------------------------------------------

  private setFollowing(on: boolean) {
    if (this.following === on) return;
    this.following = on;
    this.opts.onFollowChange(on);
  }

  private move(fn: () => void) {
    this.programmaticMoves += 1;
    fn();
  }

  private frame(bounds: L.LatLngBounds, maxZoom = 17) {
    if (!bounds.isValid()) return;
    const options: L.FitBoundsOptions = { padding: [48, 48], maxZoom };
    this.move(() => {
      if (this.opts.reducedMotion) this.map.fitBounds(bounds, { ...options, animate: false });
      else this.map.flyToBounds(bounds, { ...options, duration: 0.8 });
    });
  }

  private marker(kind: MarkerKind, at: L.LatLngExpression, label: string, sel: Selection, z: number, text = ''): L.Marker {
    const m = L.marker(at, { icon: markerIcon(kind, text), keyboard: true, zIndexOffset: z, title: label, riseOnHover: true });
    m.on('add', () => {
      const el = m.getElement();
      if (el) {
        el.setAttribute('aria-label', label);
        el.setAttribute('role', 'button');
      }
    });
    // Marker clicks don't bubble to the map in Leaflet, so no incident is placed.
    m.on('click', () => this.select(sel, m.getElement() ?? null));
    // Leaflet maps Enter to click only for markers with a bound popup, so do it here.
    m.on('keydown', (e: L.LeafletKeyboardEvent) => {
      const key = e.originalEvent.key;
      if (key === 'Enter' || key === ' ') {
        e.originalEvent.preventDefault();
        this.select(sel, m.getElement() ?? null);
      }
    });
    return m;
  }

  private hoverable(line: L.Polyline, weight: number) {
    line.on('mouseover', () => line.setStyle({ weight: weight + 2 }));
    line.on('mouseout', () => line.setStyle({ weight }));
  }

  private select(sel: Selection | null, el: HTMLElement | null) {
    this.selectedEl?.classList.remove('ems-selected');
    this.selectedEl = el;
    el?.classList.add('ems-selected');
    this.opts.onSelect(sel);
  }

  /** Route coloured by the network-wide traffic level at the time the ambulance passed each part. */
  private buildTraffic(run: RunData) {
    const runs: Array<{ tier: TierKey; metric: MetricPoint; t0: number; t1: number; pts: L.LatLng[] }> = [];
    for (let i = 0; i < run.path.length - 1; i++) {
      const a = run.path[i];
      const b = run.path[i + 1];
      if (!a || !b) continue;
      const metric = trafficAt(run, a.t);
      if (!metric) continue;
      const last = runs.at(-1);
      if (last && last.tier === metric.tier) {
        last.pts.push(L.latLng(b.lat, b.lon));
        last.t1 = b.t;
      } else {
        runs.push({ tier: metric.tier, metric, t0: a.t, t1: b.t, pts: [L.latLng(a.lat, a.lon), L.latLng(b.lat, b.lon)] });
      }
    }
    runs.forEach((r) => {
      const weight = TIER_WEIGHT[r.tier];
      const line = L.polyline(r.pts, {
        renderer: this.renderer,
        color: token(TIER_COLOR[r.tier]),
        weight,
        opacity: 0.45,
        lineCap: 'butt',
        bubblingMouseEvents: false,
      });
      this.hoverable(line, weight);
      line.on('click', () =>
        this.select({ kind: 'segment', t0: r.t0, t1: r.t1, tier: r.tier, v: r.metric.volatility_index }, null),
      );
      line.addTo(this.groups.traffic);
    });
  }

  /** Heat along the route: each path point shaded by traffic unpredictability V at that time. */
  private buildHeat(run: RunData) {
    run.path.forEach((p) => {
      const v = trafficAt(run, p.t)?.volatility_index ?? 0;
      const bin = Math.min(HEAT.length - 1, Math.floor(Math.max(0, Math.min(0.999, v)) * HEAT.length));
      L.circleMarker([p.lat, p.lon], {
        renderer: this.renderer,
        radius: 12,
        stroke: false,
        fillColor: token(HEAT[bin] ?? 'var(--heat-1)'),
        fillOpacity: 0.55,
        interactive: false,
      }).addTo(this.groups.heat);
    });
  }

  private buildChevrons(run: RunData) {
    let since = 0;
    for (let i = 1; i < run.path.length; i++) {
      const a = run.path[i - 1];
      const b = run.path[i];
      if (!a || !b) continue;
      since += this.map.distance([a.lat, a.lon], [b.lat, b.lon]);
      if (since >= CHEVRON_SPACING_M && (a.lat !== b.lat || a.lon !== b.lon)) {
        since = 0;
        const m = L.marker([(a.lat + b.lat) / 2, (a.lon + b.lon) / 2], {
          icon: chevronIcon(bearing([a.lat, a.lon], [b.lat, b.lon])),
          interactive: false,
          keyboard: false,
        });
        m.addTo(this.groups.direction);
        this.chevrons.push({ marker: m, t: b.t });
      }
    }
  }
}
