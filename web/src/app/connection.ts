import { createContext, useContext } from 'react';
import type { Connection } from './useConnection';

export const ConnectionContext = createContext<Connection>({ state: 'checking', health: null, latencyMs: null });

export function useConnectionState(): Connection {
  return useContext(ConnectionContext);
}
