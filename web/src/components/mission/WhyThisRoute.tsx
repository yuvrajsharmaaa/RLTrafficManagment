import { useId, useMemo, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { whyThisRoute } from '../../lib/explain';
import { cx } from '../../lib/cx';
import type { RunData } from '../../lib/types';

interface WhyThisRouteProps {
  run: RunData;
  kind: 'live' | 'recorded';
  baseline: RunData | null;
}

/** Numbered facts from fixed templates; each lists the fields it came from. Not a chat. */
export function WhyThisRoute({ run, kind, baseline }: WhyThisRouteProps) {
  const sentences = useMemo(() => whyThisRoute(run, kind, baseline), [run, kind, baseline]);
  const [showData, setShowData] = useState(false);
  const dataId = useId();

  return (
    <div className="flex flex-col gap-3">
      <ol className="flex flex-col gap-2">
        {sentences.map((s, i) => (
          <li key={s.text} className="flex gap-2 text-body-sm text-text-1">
            <span className="num w-4 shrink-0 text-text-3">{i + 1}</span>
            <span>{s.text}</span>
          </li>
        ))}
      </ol>
      <button
        type="button"
        aria-expanded={showData}
        aria-controls={dataId}
        onClick={() => setShowData((v) => !v)}
        className="flex w-fit items-center gap-1 rounded text-body-sm text-text-2 hover:text-text-1"
      >
        <ChevronDown size={14} strokeWidth={1.75} aria-hidden className={cx('transition-transform duration-base', !showData && '-rotate-90')} />
        Show the data behind this
      </button>
      {showData && (
        <ol id={dataId} className="flex flex-col gap-1 rounded border border-border-subtle bg-bg p-2">
          {sentences.map((s, i) => (
            <li key={s.text} className="num text-caption text-text-2">
              {i + 1}: {s.fields.join(' · ')}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
