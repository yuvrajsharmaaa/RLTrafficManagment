import { useEffect, useRef, useState } from 'react';

/**
 * Width of an element in CSS pixels, kept current with a ResizeObserver.
 * Charts draw in real pixels so their 12 px labels stay 12 px at any size.
 */
export function useElementWidth<T extends HTMLElement>(fallback = 320) {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const w = entry ? Math.round(entry.contentRect.width) : 0;
      if (w > 0) setWidth(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}
