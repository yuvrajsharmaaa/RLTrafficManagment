import type { ScreenDef } from '../app/screens';
import { EmptyState } from '../components/ui';

// Temporary content for screens not yet rebuilt in React. The current app at
// the server root (/) still provides these features until the switch-over.
export function NotBuiltYet({ screen }: { screen: ScreenDef }) {
  return (
    <div className="p-6">
      <EmptyState
        icon={screen.icon}
        title={`${screen.title} is not rebuilt yet`}
        message="This screen arrives in a later step. The current app at the server root still has this feature."
      />
    </div>
  );
}
