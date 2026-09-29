export interface LatLon {
  lat: number;
  lon: number;
}

/** Bounds of the simulated Connaught Place network (south-west, north-east). Same values as the old app. */
export const DELHI_BBOX: [[number, number], [number, number]] = [
  [28.625885, 77.21573],
  [28.641579, 77.236486],
];

export const MAP_CENTER: [number, number] = [28.6325, 77.2215];

export function insideSimulatedArea(p: LatLon): boolean {
  const [[s, w], [n, e]] = DELHI_BBOX;
  return p.lat >= s && p.lat <= n && p.lon >= w && p.lon <= e;
}
