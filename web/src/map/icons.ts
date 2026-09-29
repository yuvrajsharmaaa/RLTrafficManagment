import L from 'leaflet';
import { createElement, type ComponentType } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { Ambulance, Hospital, Route, TrafficCone, TriangleAlert, type LucideProps } from 'lucide-react';

// SVG divIcons. Every marker kind has its own SHAPE as well as its colour, so
// kinds stay distinguishable without colour (Phase 2 colour-blind rule):
//   ambulance  rounded square + heading notch     incident  diamond
//   hospital   circle (destination: double ring)   checkpoint small square
//   re-plan    ring                                 signal    triangle
// "Selected" adds an accent halo to any of them.

export type MarkerKind = 'ambulance' | 'incident' | 'destination' | 'hospital' | 'checkpoint' | 'replan' | 'signal' | 'routeEnd';

// Lucide glyphs rendered to SVG strings once per icon and size.
const svgCache = new Map<ComponentType<LucideProps>, Map<number, string>>();
function glyph(icon: ComponentType<LucideProps>, size: number): string {
  let bySize = svgCache.get(icon);
  if (!bySize) {
    bySize = new Map();
    svgCache.set(icon, bySize);
  }
  let markup = bySize.get(size);
  if (!markup) {
    markup = renderToStaticMarkup(createElement(icon, { size, strokeWidth: 1.75, 'aria-hidden': true }));
    bySize.set(size, markup);
  }
  return markup;
}

interface Spec {
  size: number;
  html: (label: string) => string;
}

const specs: Record<MarkerKind, Spec> = {
  ambulance: {
    size: 32,
    html: () =>
      `<div class="ems-mk ems-mk-ambulance"><span class="ems-heading" data-heading></span>${glyph(Ambulance, 16)}</div>`,
  },
  incident: {
    size: 30,
    html: () => `<div class="ems-mk ems-mk-incident"><span class="ems-pulse" aria-hidden="true"></span><span class="ems-diamond">${glyph(TriangleAlert, 14)}</span></div>`,
  },
  destination: {
    size: 30,
    html: () => `<div class="ems-mk ems-mk-destination">${glyph(Hospital, 16)}</div>`,
  },
  hospital: {
    size: 22,
    html: () => `<div class="ems-mk ems-mk-hospital">${glyph(Hospital, 12)}</div>`,
  },
  checkpoint: {
    size: 16,
    html: (label) => `<div class="ems-mk ems-mk-checkpoint num">${label}</div>`,
  },
  replan: {
    size: 22,
    html: () => `<div class="ems-mk ems-mk-replan">${glyph(Route, 12)}</div>`,
  },
  routeEnd: {
    size: 12,
    html: () => `<div class="ems-mk ems-mk-route-end"></div>`,
  },
  signal: {
    size: 20,
    html: () => `<div class="ems-mk ems-mk-signal">${glyph(TrafficCone, 10)}</div>`,
  },
};

export function markerIcon(kind: MarkerKind, label = ''): L.DivIcon {
  const spec = specs[kind];
  return L.divIcon({
    className: 'ems-mk-wrap',
    html: spec.html(label),
    iconSize: [spec.size, spec.size],
    iconAnchor: [spec.size / 2, spec.size / 2],
  });
}

/** Small arrow placed along the route to show direction of travel. */
export function chevronIcon(deg: number): L.DivIcon {
  return L.divIcon({
    className: 'ems-chevron-wrap',
    html: `<svg class="ems-chevron" viewBox="0 0 10 10" width="10" height="10" style="transform: rotate(${deg}deg)"><path d="M2 7 L5 3 L8 7" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    iconSize: [10, 10],
    iconAnchor: [5, 5],
  });
}

export function clusterIcon(count: number): L.DivIcon {
  return L.divIcon({
    className: 'ems-mk-wrap',
    html: `<div class="ems-mk ems-mk-cluster num">${count}</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });
}
