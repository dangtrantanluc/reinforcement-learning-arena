import { AnimatePresence, motion } from 'framer-motion';

interface Props {
  logs: string[];
}

// Pick a tag colour from the log line prefix.
function tone(line: string): string {
  if (line.includes('[Winner]')) return 'text-success';
  if (line.includes('[Collision]')) return 'text-warning';
  if (line.includes('[Reward]')) return 'text-success';
  if (line.includes('PPO')) return 'text-[#60a5fa]';
  if (line.includes('Dyna-Q')) return 'text-[#a78bfa]';
  return 'text-white/70';
}

export default function TrainingLog({ logs }: Props) {
  // Newest first; key by index+content since the backend sends plain strings.
  const items = logs.slice().reverse();
  return (
    <div className="card flex h-full flex-col p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className="panel-title">Console</p>
          <p className="text-sm font-semibold text-ink">Training Log</p>
        </div>
        <span className="inline-flex items-center gap-1.5 text-[11px] text-sub">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-success" /> live
        </span>
      </div>

      <div className="scroll-thin flex-1 overflow-y-auto rounded-lg bg-[#1f2535] p-3 font-mono text-[12px] leading-relaxed">
        <AnimatePresence initial={false}>
          {items.map((line, i) => (
            <motion.div
              key={`${items.length - i}-${line}`}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2 }}
              className={`py-0.5 ${tone(line)}`}
            >
              {line}
            </motion.div>
          ))}
        </AnimatePresence>
        {items.length === 0 && <p className="text-white/40">// waiting for events…</p>}
      </div>
    </div>
  );
}
