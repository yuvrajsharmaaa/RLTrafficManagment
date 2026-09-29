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
  { key: 'va', method: 'Adaptive routing', technical: 'VA-QPSO', best: 361.158, mean: 366.966, meanStd: 3.212, to95: 35.3, to95Std: 27.4, toMargin: 33.4, toMarginStd: 26.9, hitRate: 100 },
  { key: 'fixed', method: 'Fixed schedule', technical: 'Fixed-β QPSO', best: 361.158, mean: 366.966, meanStd: 3.212, to95: 37.4, to95Std: 39.0, toMargin: 33.1, toMarginStd: 34.1, hitRate: 100 },
  { key: 'pso', method: 'Standard swarm search', technical: 'PSO', best: 361.158, mean: 366.966, meanStd: 3.212, to95: 51.7, to95Std: 61.3, toMargin: 50.8, toMarginStd: 61.6, hitRate: 100 },
  { key: 'ga', method: 'Genetic search', technical: 'GA (OX / swap / elitist)', best: 361.158, mean: 366.966, meanStd: 3.212, to95: 74.9, to95Std: 73.1, toMargin: 68.0, toMarginStd: 68.9, hitRate: 100 },
  { key: 'sa', method: 'Annealing search', technical: 'SA', best: 361.158, mean: 366.966, meanStd: 3.212, to95: 23.2, to95Std: 19.0, toMargin: 21.4, toMarginStd: 18.5, hitRate: 100 },
  { key: 'dijkstra', method: 'Shortest-path baseline', technical: 'Dijkstra (nearest neighbour)', best: 686.082, mean: 744.049, meanStd: 43.088, to95: null, to95Std: null, toMargin: null, toMarginStd: null, hitRate: 0 },
];

export const METHODS_SETUP = '30 seeded trials, 8 stops, Moderate traffic, synthetic congestion model (not measured SUMO travel time)';

/** Table 6: Real-world performance measured in Eclipse SUMO across 5 random seeds per traffic tier. */
export interface RealBenchmarkRow {
  tier: ScenarioTier;
  algorithm: string;
  technical: string;
  completed: number;
  total: number;
  meanArrivalS: number | null;
  minArrivalS: number | null;
  maxArrivalS: number | null;
  meanDrivenM: number;
}

export const REAL_BENCHMARK: RealBenchmarkRow[] = [
  // Low traffic
  { tier: 'low', algorithm: 'Adaptive routing', technical: 'VA-QPSO', completed: 1, total: 5, meanArrivalS: 777.0, minArrivalS: 777.0, maxArrivalS: 777.0, meanDrivenM: 5146 },
  { tier: 'low', algorithm: 'Fixed schedule', technical: 'Fixed-β QPSO', completed: 5, total: 5, meanArrivalS: 852.0, minArrivalS: 852.0, maxArrivalS: 852.0, meanDrivenM: 4774 },
  { tier: 'low', algorithm: 'Standard swarm search', technical: 'PSO', completed: 3, total: 5, meanArrivalS: 852.0, minArrivalS: 852.0, maxArrivalS: 852.0, meanDrivenM: 4850 },
  { tier: 'low', algorithm: 'Genetic search', technical: 'GA', completed: 4, total: 5, meanArrivalS: 795.8, minArrivalS: 777.0, maxArrivalS: 852.0, meanDrivenM: 4889 },
  { tier: 'low', algorithm: 'Annealing search', technical: 'SA', completed: 3, total: 5, meanArrivalS: 804.3, minArrivalS: 777.0, maxArrivalS: 859.0, meanDrivenM: 5017 },
  { tier: 'low', algorithm: 'Shortest-path baseline', technical: 'Dijkstra (NN)', completed: 5, total: 5, meanArrivalS: 866.0, minArrivalS: 866.0, maxArrivalS: 866.0, meanDrivenM: 4771 },

  // Medium traffic (all capped at 900s)
  { tier: 'medium', algorithm: 'Adaptive routing', technical: 'VA-QPSO', completed: 0, total: 5, meanArrivalS: null, minArrivalS: null, maxArrivalS: null, meanDrivenM: 2741 },
  { tier: 'medium', algorithm: 'Fixed schedule', technical: 'Fixed-β QPSO', completed: 0, total: 5, meanArrivalS: null, minArrivalS: null, maxArrivalS: null, meanDrivenM: 2797 },
  { tier: 'medium', algorithm: 'Standard swarm search', technical: 'PSO', completed: 0, total: 5, meanArrivalS: null, minArrivalS: null, maxArrivalS: null, meanDrivenM: 2806 },
  { tier: 'medium', algorithm: 'Genetic search', technical: 'GA', completed: 0, total: 5, meanArrivalS: null, minArrivalS: null, maxArrivalS: null, meanDrivenM: 2874 },
  { tier: 'medium', algorithm: 'Annealing search', technical: 'SA', completed: 0, total: 5, meanArrivalS: null, minArrivalS: null, maxArrivalS: null, meanDrivenM: 2060 },
  { tier: 'medium', algorithm: 'Shortest-path baseline', technical: 'Dijkstra (NN)', completed: 0, total: 5, meanArrivalS: null, minArrivalS: null, maxArrivalS: null, meanDrivenM: 1647 },

  // High traffic (all capped at 900s)
  { tier: 'high', algorithm: 'Adaptive routing', technical: 'VA-QPSO', completed: 0, total: 5, meanArrivalS: null, minArrivalS: null, maxArrivalS: null, meanDrivenM: 898 },
  { tier: 'high', algorithm: 'Fixed schedule', technical: 'Fixed-β QPSO', completed: 0, total: 5, meanArrivalS: null, minArrivalS: null, maxArrivalS: null, meanDrivenM: 708 },
  { tier: 'high', algorithm: 'Standard swarm search', technical: 'PSO', completed: 0, total: 5, meanArrivalS: null, minArrivalS: null, maxArrivalS: null, meanDrivenM: 1154 },
  { tier: 'high', algorithm: 'Genetic search', technical: 'GA', completed: 0, total: 5, meanArrivalS: null, minArrivalS: null, maxArrivalS: null, meanDrivenM: 702 },
  { tier: 'high', algorithm: 'Annealing search', technical: 'SA', completed: 0, total: 5, meanArrivalS: null, minArrivalS: null, maxArrivalS: null, meanDrivenM: 631 },
  { tier: 'high', algorithm: 'Shortest-path baseline', technical: 'Dijkstra (NN)', completed: 0, total: 5, meanArrivalS: null, minArrivalS: null, maxArrivalS: null, meanDrivenM: 454 },
];

