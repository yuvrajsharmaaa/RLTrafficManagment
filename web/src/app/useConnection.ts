import { useEffect, useState } from 'react';

export type ConnectionState = 'checking' | 'connected' | 'server-unreachable' | 'offline';

/** Shape returned by GET /health in server.py. */
export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  sumo_available: boolean;
  live_pipeline: string;
}

export interface Connection {
  state: ConnectionState;
  health: HealthResponse | null;
  /** Round-trip time of the last successful check, in ms. */
  latencyMs: number | null;
}

const POLL_MS = 30_000;
const TIMEOUT_MS = 5_000;

/** Polls the existing GET /health endpoint and listens to the browser's online/offline events. */
export function useConnection(): Connection {
  const [conn, setConn] = useState<Connection>({
    state: navigator.onLine ? 'checking' : 'offline',
    health: null,
    latencyMs: null,
  });

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      if (!navigator.onLine) {
        setConn((c) => ({ ...c, state: 'offline' }));
        return;
      }
      const controller = new AbortController();
      const timer = window.setTimeout(() => controller.abort(), TIMEOUT_MS);
      const started = performance.now();
      try {
        const res = await fetch('/health', { cache: 'no-store', signal: controller.signal });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const health = (await res.json()) as HealthResponse;
        if (!cancelled) {
          setConn({ state: 'connected', health, latencyMs: Math.round(performance.now() - started) });
        }
      } catch {
        if (!cancelled) setConn((c) => ({ ...c, state: 'server-unreachable', latencyMs: null }));
      } finally {
        window.clearTimeout(timer);
      }
    };

    void check();
    const interval = window.setInterval(() => void check(), POLL_MS);
    const onOnline = () => void check();
    const onOffline = () => setConn((c) => ({ ...c, state: 'offline' }));
    window.addEventListener('online', onOnline);
    window.addEventListener('offline', onOffline);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
      window.removeEventListener('online', onOnline);
      window.removeEventListener('offline', onOffline);
    };
  }, []);

  return conn;
}
