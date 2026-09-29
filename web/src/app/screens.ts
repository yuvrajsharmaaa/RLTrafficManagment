import {
  ChartColumn, Columns2, Cpu, History, Map as MapIcon, Navigation, Settings, type LucideIcon,
} from 'lucide-react';

export type ScreenId =
  | 'mission' | 'simulation' | 'optimization' | 'analytics' | 'benchmark' | 'network' | 'settings';

export interface ScreenDef {
  id: ScreenId;
  /** Sidebar word (fits 80 px at 12 px). */
  nav: string;
  /** Header name. */
  title: string;
  /** Header secondary label; technical names live here, not in the nav. */
  secondary?: string;
  icon: LucideIcon;
}

// Order is the sidebar order (Phase 3, section 0). Settings is pinned to the bottom.
export const SCREENS: readonly ScreenDef[] = [
  { id: 'mission', nav: 'Mission', title: 'Mission Control', secondary: 'Live dispatch', icon: Navigation },
  { id: 'simulation', nav: 'Simulation', title: 'Simulation', secondary: 'Recorded runs', icon: History },
  { id: 'optimization', nav: 'Optimization', title: 'Quantum-Inspired Optimization', secondary: 'QPSO on a classical computer', icon: Cpu },
  { id: 'analytics', nav: 'Analytics', title: 'Analytics', secondary: 'Adaptive vs fixed', icon: Columns2 },
  { id: 'benchmark', nav: 'Benchmark', title: 'Benchmark', secondary: 'All methods', icon: ChartColumn },
  { id: 'network', nav: 'Network', title: 'Network', secondary: 'Roads and hospitals', icon: MapIcon },
  { id: 'settings', nav: 'Settings', title: 'Settings', icon: Settings },
];

export const DEFAULT_SCREEN: ScreenId = 'mission';

export function isScreenId(value: string | null): value is ScreenId {
  return SCREENS.some((s) => s.id === value);
}

export function screenDef(id: ScreenId): ScreenDef {
  const def = SCREENS.find((s) => s.id === id);
  if (!def) throw new Error(`Unknown screen: ${id}`);
  return def;
}
