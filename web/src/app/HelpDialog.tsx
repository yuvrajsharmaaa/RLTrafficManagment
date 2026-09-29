import * as Dialog from '@radix-ui/react-dialog';
import { X } from 'lucide-react';
import { Button } from '../components/ui';

interface HelpDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

// Carries the content of the old intro popup (#intro-modal), in plain language.
// Closing it no longer starts playback.
export function HelpDialog({ open, onOpenChange }: HelpDialogProps) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-dialog bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-dialog flex max-h-[85vh] w-[560px] max-w-[calc(100vw-32px)] -translate-x-1/2 -translate-y-1/2 flex-col gap-4 overflow-y-auto rounded-lg border border-border-strong bg-panel p-6 shadow-3">
          <div className="flex items-start justify-between gap-4">
            <Dialog.Title className="text-title-lg text-text-1">How to read this console</Dialog.Title>
            <Dialog.Close
              aria-label="Close help"
              className="grid h-8 w-8 shrink-0 place-items-center rounded text-text-2 hover:bg-raised hover:text-text-1"
            >
              <X size={16} strokeWidth={1.75} aria-hidden />
            </Dialog.Close>
          </div>
          <Dialog.Description className="text-body text-text-2">
            This console routes an emergency ambulance through simulated Delhi traffic (Connaught Place) to the
            nearest trauma centre.
          </Dialog.Description>
          <dl className="flex flex-col gap-3 text-body">
            <div>
              <dt className="font-semibold text-text-1">Arrival time</dt>
              <dd className="text-text-2">Remaining drive time to the hospital. It changes when the route is re-planned.</dd>
            </div>
            <div>
              <dt className="font-semibold text-text-1">Traffic unpredictability</dt>
              <dd className="text-text-2">
                How much road speeds are changing across the network, from 0 (steady) to 1 (unstable). When it rises,
                the route search looks wider and re-plans sooner.
              </dd>
            </div>
            <div>
              <dt className="font-semibold text-text-1">Adaptive routing vs fixed schedule</dt>
              <dd className="text-text-2">
                Adaptive routing reacts to live traffic; the fixed schedule ignores it. Analytics compares the two on
                the same recorded traffic.
              </dd>
            </div>
          </dl>
          <div className="flex justify-end">
            <Dialog.Close asChild>
              <Button variant="primary">Close</Button>
            </Dialog.Close>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
