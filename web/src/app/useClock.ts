import { useEffect, useState } from 'react';

const formatter = new Intl.DateTimeFormat('en-IN', {
  timeZone: 'Asia/Kolkata',
  hour: '2-digit',
  minute: '2-digit',
  hourCycle: 'h23',
});

/** Local time in IST, 24-hour, refreshed every 15 s. */
export function useIstClock(): string {
  const [now, setNow] = useState(() => formatter.format(new Date()));
  useEffect(() => {
    const id = window.setInterval(() => setNow(formatter.format(new Date())), 15_000);
    return () => window.clearInterval(id);
  }, []);
  return now;
}
