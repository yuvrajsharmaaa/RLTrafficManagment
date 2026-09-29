import { useCallback, useEffect, useState } from 'react';
import { DEFAULT_SCREEN, isScreenId, type ScreenId } from './screens';

// The current screen lives in the ?screen= query parameter. Other parameters
// (the existing ?run= and ?compare=true deep links) are left untouched.

function readScreen(): ScreenId {
  const params = new URLSearchParams(window.location.search);
  const value = params.get('screen');
  if (isScreenId(value)) return value;
  // The old app's ?compare=true deep link opens the comparison.
  if (params.get('compare') === 'true') return 'analytics';
  return DEFAULT_SCREEN;
}

export function useScreen(): [ScreenId, (next: ScreenId) => void] {
  const [screen, setScreenState] = useState<ScreenId>(readScreen);

  useEffect(() => {
    const onPop = () => setScreenState(readScreen());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  const setScreen = useCallback((next: ScreenId) => {
    const url = new URL(window.location.href);
    url.searchParams.set('screen', next);
    window.history.pushState(null, '', url);
    setScreenState(next);
  }, []);

  return [screen, setScreen];
}
