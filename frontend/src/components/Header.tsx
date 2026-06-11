import { motion } from 'framer-motion';
import ControlPanel from './ControlPanel';

interface Props {
  running: boolean;
  connected: boolean;
  busy?: boolean;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
  onEvaluate: () => void;
}

export default function Header({ running, connected, busy, ...handlers }: Props) {
  const label = !connected ? 'Offline' : running ? 'Running' : 'Paused';
  const tone = !connected
    ? { dot: 'bg-danger', text: 'text-danger', bg: 'bg-danger/10' }
    : running
      ? { dot: 'bg-success', text: 'text-success', bg: 'bg-success/10' }
      : { dot: 'bg-warning', text: 'text-warning', bg: 'bg-warning/10' };

  return (
    <header className="flex flex-col gap-4 border-b border-line bg-white/70 px-6 py-4 backdrop-blur lg:flex-row lg:items-center lg:justify-between">
      <div className="flex items-center gap-3">
        <div>
          <h1 className="text-lg font-bold tracking-tight text-ink">PPO vs Dyna-Q Solo Arena</h1>
          <p className="text-xs text-sub">Two agents · one shared map · live competitive training</p>
        </div>
        <span className={`ml-1 inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ${tone.bg} ${tone.text}`}>
          <motion.span
            className={`h-2 w-2 rounded-full ${tone.dot}`}
            animate={running && connected ? { scale: [1, 1.4, 1], opacity: [1, 0.5, 1] } : {}}
            transition={{ duration: 1, repeat: Infinity }}
          />
          {label}
        </span>
      </div>

      <ControlPanel running={running} busy={busy} {...handlers} />
    </header>
  );
}
