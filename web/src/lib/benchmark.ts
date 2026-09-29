import type { ScenarioTier } from './types';

// Figures copied from README.md, section 12 ("Results & Experimental
// Validation"). The server does not serve results/, so these are a labelled
// snapshot, not live data (Phase 1 question 1 is still open).

export const README_SOURCE = 'README section 12';

/** Table 4: paired comparison, 10 seeds per traffic level. Differences are adaptive minus fixed. */
export interface PairedRow {
  tier: ScenarioTier;
  adaptiveMean: number;
  fixedMean: number;
  /** Adaptive minus fixed, seconds; negative = adaptive sooner. */
  diff: number;
  pValue: number;
  /** Vargha-Delaney A12: probability a random adaptive trial beats a random fixed trial. */
  a12: number;
  /** Congestion score difference, adaptive minus fixed; negative = adaptive less congestion. */
  congestionDiff: number;
  /** Stated in the README only for Heavy traffic. */
  congestionNote?: string;
}

export const PAIRED: PairedRow[] = [
  { tier: 'low', adaptiveMean: 93.55, fixedMean: 93.71, diff: -0.16, pValue: 0.002, a12: 1.0, congestionDiff: 0.0013 },
  { tier: 'medium', adaptiveMean: 95.22, fixedMean: 98.62, diff: -3.4, pValue: 0.002, a12: 1.0, congestionDiff: 0.0047 },
  {
    tier: 'high', adaptiveMean: 125.7, fixedMean: 109.37, diff: 16.33, pValue: 0.002, a12: 0.0, congestionDiff: -0.084,
    congestionNote: '25.62% less congestion exposure',
  },
];

/**
 * results/experiments.csv (the source of table 4) shows every one of the 10
 * seeds in a traffic level produced exactly the same time and congestion score.
 * The "10 trials" are therefore one outcome repeated; the p-value reflects ten
 * identical differences, not variation across conditions.
 */
export const PAIRED_SEEDS_IDENTICAL = true;

/** Table 5: 30 seeded trials, 8 stops, Moderate traffic, 600 iterations or generations. */
export interface MethodRow {
  key: string;
  method: string;
  technical: string;
  best: number;
  mean: number;
  meanStd: number;
  /** null for methods that do not iterate. */
  to95: number | null;
  to95Std: number | null;
  toMargin: number | null;
  toMarginStd: number | null;
  hitRate: number;
}

export const METHODS: MethodRow[] = [
  { key: 'va', method: 'Adaptive routing', technical: 'VA-QPSO', best: 71.076, mean: 76.968, meanStd: 3.029, to95: 25.3, to95Std: 33.8, toMargin: 17.2, toMarginStd: 15.5, hitRate: 100 },
  { key: 'fixed', method: 'Fixed schedule', technical: 'Fixed-β QPSO', best: 71.076, mean: 76.968, meanStd: 3.029, to95: 41.8, to95Std: 56.7, toMargin: 32.2, toMarginStd: 43.8, hitRate: 100 },
  { key: 'pso', method: 'Standard swarm search', technical: 'PSO', best: 71.076, mean: 76.968, meanStd: 3.029, to95: 41.3, to95Std: 57.2, toMargin: 40.8, toMarginStd: 57.4, hitRate: 100 },
  { key: 'ga', method: 'Genetic search', technical: 'GA (OX / swap / elitist)', best: 71.076, mean: 76.968, meanStd: 3.029, to95: 115.1, to95Std: 110.2, toMargin: 68.9, toMarginStd: 68.5, hitRate: 100 },
  { key: 'sa', method: 'Annealing search', technical: 'SA', best: 71.076, mean: 78.1, meanStd: 4.1, to95: 88.4, to95Std: 72.1, toMargin: 51.2, toMarginStd: 48.3, hitRate: 96.7 },
  { key: 'dijkstra', method: 'Shortest-path baseline', technical: 'Dijkstra (nearest neighbour)', best: 103.541, mean: 110.061, meanStd: 3.443, to95: null, to95Std: null, toMargin: null, toMarginStd: null, hitRate: 0 },
];

export const METHODS_SETUP = '30 seeded trials, 8 stops, Moderate traffic, up to 600 iterations';
