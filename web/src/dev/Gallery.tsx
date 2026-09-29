import { Navigation, Radio, WifiOff } from 'lucide-react';
import {
  Button, EmptyState, ErrorState, Panel, Skeleton, StatBlock, StatusChip, Table, Tabs, TabsContent,
  TabsList, TabsTrigger, Tooltip, type Column,
} from '../components/ui';

// Dev-only visual check of the shared primitives. Open with ?gallery=1 on the
// Vite dev server; it is excluded from production builds.

interface Row { method: string; best: number; mean: number }
const rows: Row[] = [
  { method: 'Adaptive routing', best: 361.158, mean: 366.966 },
  { method: 'Shortest-path baseline', best: 686.082, mean: 744.049 },
];
const columns: Column<Row>[] = [
  { key: 'method', header: 'Method', render: (r) => r.method, sortValue: (r) => r.method },
  { key: 'best', header: 'Best score (s)', numeric: true, render: (r) => r.best.toFixed(3), sortValue: (r) => r.best },
  { key: 'mean', header: 'Mean (s)', numeric: true, render: (r) => r.mean.toFixed(3), sortValue: (r) => r.mean },
];

export function Gallery() {
  return (
    <main className="grid max-w-[1200px] grid-cols-2 gap-4 p-6">
      <Panel title="Buttons">
        <div className="flex flex-wrap gap-2">
          <Button variant="primary" size="large" icon={Navigation}>Find route</Button>
          <Button variant="primary" size="large" loading loadingLabel="Finding route">Find route</Button>
          <Button>Secondary</Button>
          <Button variant="ghost">Ghost</Button>
          <Button variant="danger">Clear cache</Button>
          <Button disabled>Disabled</Button>
          <Button size="compact">Compact</Button>
        </div>
      </Panel>
      <Panel title="Status chips">
        <div className="flex flex-wrap gap-2">
          <StatusChip tone="success" label="Arrived" />
          <StatusChip tone="warning" label="Estimated traffic" />
          <StatusChip tone="danger" label="Offline" icon={WifiOff} />
          <StatusChip tone="info" label="Live simulation snapshot" icon={Radio} />
          <StatusChip tone="neutral" label="Recorded run" />
        </div>
      </Panel>
      <Panel title="Stat blocks">
        <div className="flex gap-8">
          <StatBlock label="Arrival in" value="00:41" size="xl" />
          <StatBlock label="Total route time" value="95.2" unit="s" size="lg" delta="3.4 s sooner" />
          <StatBlock label="Hospital load" value={null} />
          <StatBlock label="Traffic" value="0.39" staleNote="as of T+55 s" />
        </div>
      </Panel>
      <Panel title="Tabs and tooltip">
        <Tabs defaultValue="why">
          <TabsList aria-label="Route explanation">
            <TabsTrigger value="why">Why this route</TabsTrigger>
            <TabsTrigger value="timeline">Timeline</TabsTrigger>
          </TabsList>
          <TabsContent value="why">
            <Tooltip content="Search breadth (β): contraction-expansion coefficient">
              <button type="button" className="text-body text-text-2 underline decoration-dotted">Search breadth</button>
            </Tooltip>
          </TabsContent>
          <TabsContent value="timeline">T+55 s Route re-planned</TabsContent>
        </Tabs>
      </Panel>
      <Panel title="Table" className="col-span-2">
        <Table caption="Benchmark sample" columns={columns} rows={rows} rowKey={(r) => r.method} source="Source: README snapshot" />
      </Panel>
      <Panel title="Details (collapsible)" collapsible defaultOpen={false}>
        <Skeleton lines={3} />
      </Panel>
      <Panel title="Empty and error">
        <div className="flex flex-col gap-4">
          <EmptyState icon={Navigation} title="No active incident" message="Click the map or type coordinates to begin." />
          <ErrorState title="Route request failed" message="Recorded runs still work." onRetry={() => undefined} />
        </div>
      </Panel>
    </main>
  );
}
