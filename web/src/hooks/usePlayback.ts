import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

export type FrameListener = (t: number) => void;

export interface Playback {
  /** Throttled time for React displays (about 4 updates per second). */
  t: number;
  playing: boolean;
  speed: number;
  duration: number;
  /** Exact current time, for imperative readers. */
  getTime: () => number;
  /** Called on every animation frame and on seek; used by the map, which never re-renders per frame. */
  subscribe: (fn: FrameListener) => () => void;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  seek: (t: number) => void;
  restart: () => void;
  setSpeed: (s: number) => void;
}

const DISPLAY_INTERVAL_MS = 250;

/** One playback clock shared by the map, panels, dock and status bar. */
export function usePlayback(duration: number, runKey: string): Playback {
  const timeRef = useRef(0);
  const listeners = useRef(new Set<FrameListener>());
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeedState] = useState(1);
  const speedRef = useRef(1);
  const lastDisplay = useRef(0);

  const emit = useCallback((now: number, forceDisplay: boolean) => {
    timeRef.current = now;
    listeners.current.forEach((fn) => fn(now));
    const wall = performance.now();
    if (forceDisplay || wall - lastDisplay.current >= DISPLAY_INTERVAL_MS) {
      lastDisplay.current = wall;
      setT(now);
    }
  }, []);

  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    let last: number | null = null;
    const loop = (ts: number) => {
      if (last !== null) {
        const next = Math.min(duration, timeRef.current + ((ts - last) / 1000) * speedRef.current);
        if (next >= duration) {
          emit(duration, true);
          setPlaying(false);
          return;
        }
        emit(next, false);
      }
      last = ts;
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [playing, duration, emit]);

  // A new run resets the clock: React state during render, listeners in an effect.
  const [clockKey, setClockKey] = useState(runKey);
  if (clockKey !== runKey) {
    setClockKey(runKey);
    setPlaying(false);
    setT(0);
  }
  useEffect(() => {
    timeRef.current = 0;
    listeners.current.forEach((fn) => fn(0));
  }, [runKey]);

  const seek = useCallback(
    (next: number) => emit(Math.max(0, Math.min(duration, next)), true),
    [duration, emit],
  );

  const play = useCallback(() => {
    if (timeRef.current >= duration) emit(0, true);
    setPlaying(duration > 0);
  }, [duration, emit]);

  const pause = useCallback(() => {
    setPlaying(false);
    emit(timeRef.current, true);
  }, [emit]);

  const setSpeed = useCallback((s: number) => {
    speedRef.current = s;
    setSpeedState(s);
  }, []);

  return useMemo<Playback>(
    () => ({
      t,
      playing,
      speed,
      duration,
      getTime: () => timeRef.current,
      subscribe: (fn) => {
        listeners.current.add(fn);
        fn(timeRef.current);
        return () => listeners.current.delete(fn);
      },
      play,
      pause,
      toggle: () => (playing ? pause() : play()),
      seek,
      restart: () => {
        setPlaying(false);
        emit(0, true);
      },
      setSpeed,
    }),
    [t, playing, speed, duration, play, pause, seek, setSpeed, emit],
  );
}
