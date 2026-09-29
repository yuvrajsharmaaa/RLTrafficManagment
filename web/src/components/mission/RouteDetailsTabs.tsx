import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui';
import type { RunData } from '../../lib/types';
import { HospitalCard } from './HospitalCard';
import { MissionTimeline } from './MissionTimeline';
import { WhyThisRoute } from './WhyThisRoute';

interface RouteDetailsTabsProps {
  run: RunData;
  kind: 'live' | 'recorded';
  baseline: RunData | null;
  t: number;
  onSeek: (t: number) => void;
}

/** Mission Control's right panel (Phase 3): Why this route first, then Timeline and Hospitals. */
export function RouteDetailsTabs({ run, kind, baseline, t, onSeek }: RouteDetailsTabsProps) {
  return (
    <Tabs defaultValue="why">
      <TabsList aria-label="Route details">
        <TabsTrigger value="why">Why this route</TabsTrigger>
        <TabsTrigger value="timeline">Timeline</TabsTrigger>
        <TabsTrigger value="hospitals">Hospitals</TabsTrigger>
      </TabsList>
      <TabsContent value="why">
        <WhyThisRoute run={run} kind={kind} baseline={baseline} />
      </TabsContent>
      <TabsContent value="timeline">
        <MissionTimeline run={run} t={t} onSeek={onSeek} />
      </TabsContent>
      <TabsContent value="hospitals">
        <HospitalCard run={run} />
      </TabsContent>
    </Tabs>
  );
}
