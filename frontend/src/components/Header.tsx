import { motion } from 'framer-motion';
import ControlPanel from './ControlPanel';
import type { Theme } from '../hooks/useTheme';

interface Props {
  running: boolean;
  connected: boolean;
  busy?: boolean;
  episode: number;
  step: number;
  queueLen: number;
  theme: Theme;
  onToggleTheme: () => void;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
  onNewMap: () => void;
  onEvaluate: () => void;
  onSpeed: (delay: number) => void;
  onSave: () => void;
  onLoad: () => void;
}

const SunIcon = () => (
  <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" />
  </svg>
);
const MoonIcon = () => (
  <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z" />
  </svg>
);

export default function Header({
  running, connected, busy, episode, step, queueLen, theme, onToggleTheme, ...handlers
}: Props) {
  const label = !connected ? 'Offline' : running ? 'Running' : 'Paused';
  const tone = !connected
    ? { dot: 'bg-danger', text: 'text-danger', bg: 'bg-danger/10' }
    : running
      ? { dot: 'bg-success', text: 'text-success', bg: 'bg-success/10' }
      : { dot: 'bg-warning', text: 'text-warning', bg: 'bg-warning/10' };

  return (
    <header className="flex flex-col gap-4 border-b border-line bg-surface/70 px-6 py-4 backdrop-blur xl:flex-row xl:items-center xl:justify-between">
      <div className="flex items-center gap-3">
        <div>
          <h1 className="text-lg font-bold tracking-tight text-ink">PPO vs Dyna-Q Solo Arena</h1>
          <p className="text-xs text-sub">Two agents · one shared map · realtime step replay</p>
        </div>
        <span className={`ml-1 inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ${tone.bg} ${tone.text}`}>
          <motion.span
            className={`h-2 w-2 rounded-full ${tone.dot}`}
            animate={running && connected ? { scale: [1, 1.4, 1], opacity: [1, 0.5, 1] } : {}}
            transition={{ duration: 1, repeat: Infinity }}
          />
          {label}
        </span>
        {connected && (
          <span className="hidden items-center gap-3 font-mono text-[11px] text-sub sm:inline-flex">
            <span>ep <span className="font-semibold text-ink">{episode}</span></span>
            <span>step <span className="font-semibold text-ink">{step}</span></span>
            {queueLen > 80 && (
              <span className="rounded bg-warning/10 px-1.5 py-0.5 text-warning" title="Replay buffer behind training">
                buffer {queueLen}
              </span>
            )}
          </span>
        )}
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onToggleTheme}
          className="btn-ghost px-2.5"
          title={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}
          aria-label="Toggle theme"
        >
          {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
        </button>
        <ControlPanel running={running} busy={busy} {...handlers} />
      </div>
    </header>
  );
}
