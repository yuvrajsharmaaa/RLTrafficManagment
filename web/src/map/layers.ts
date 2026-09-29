import type { LayerKey } from './engine';

export type LayerState = Record<LayerKey, boolean>;

export const DEFAULT_LAYERS: LayerState = {
  route: true,
  baseline: true,
  direction: true,
  traffic: false,
  heat: false,
  hospitals: true,
  checkpoints: true,
  replans: true,
  area: true,
};
