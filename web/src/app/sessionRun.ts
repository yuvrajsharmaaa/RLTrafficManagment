import { createContext, useContext } from 'react';
import type { RunData, ScenarioTier } from '../lib/types';
import type { ScreenId } from './screens';

/** The mission most recently opened in Mission Control, shared with the analysis screens. */
export interface SessionRun {
  run: RunData;
  baseline: RunData | null;
  kind: 'live' | 'recorded';
  tier: ScenarioTier;
  /** Browser-measured request time for live runs. */
  requestMs: number | null;
}

export interface SessionApi {
  current: SessionRun | null;
  setCurrent: (run: SessionRun | null) => void;
  navigate: (screen: ScreenId) => void;
}

export const SessionContext = createContext<SessionApi>({
  current: null,
  setCurrent: () => undefined,
  navigate: () => undefined,
});

export function useSession(): SessionApi {
  return useContext(SessionContext);
}
