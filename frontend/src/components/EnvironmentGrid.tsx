import { motion } from 'framer-motion';
import { CELL, type AgentId, type Winner } from '../types';
import AgentCharacter from './AgentCharacter';

interface Props {
  grid: number[][];
  ppoPos: [number, number];
  dynaqPos: [number, number];
  ppoMood: 'idle' | 'moving' | 'reward' | 'danger' | 'winner';
  dynaqMood: 'idle' | 'moving' | 'reward' | 'danger' | 'winner';
  winner: Winner;
}

// Background style per static cell code (agents drawn separately on top).
function cellStyle(code: number): string {
  switch (code) {
    case CELL.WALL:
      return 'bg-[#3a3f4b]';
    case CELL.REWARD:
      return 'bg-success/15';
    case CELL.DANGER:
      return 'bg-danger/15';
    case CELL.GOAL:
      return 'bg-gold/25';
    default:
      return 'bg-white';
  }
}

function CellGlyph({ code, r, c }: { code: number; r: number; c: number }) {
  if (code === CELL.REWARD)
    return (
      <motion.span
        className="block h-3 w-3 rounded-full bg-success shadow-[0_0_8px_rgba(34,197,94,0.6)]"
        animate={{ scale: [1, 1.18, 1] }}
        transition={{ duration: 1.4, repeat: Infinity }}
      />
    );
  if (code === CELL.GOAL) return <span className="text-sm">🏁</span>;
  if (code === CELL.DANGER) {
    return (
      <span
        key={`${r}-${c}`}
        className="absolute inset-0 bg-[repeating-linear-gradient(45deg,transparent,transparent_5px,rgba(239,68,68,0.14)_5px,rgba(239,68,68,0.14)_10px)]"
      />
    );
  }
  return null;
}

const LEGEND: { label: string; swatch: string }[] = [
  { label: 'PPO (P)', swatch: 'bg-ppo' },
  { label: 'Dyna-Q (D)', swatch: 'bg-dynaq' },
  { label: 'Reward', swatch: 'bg-success' },
  { label: 'Wall', swatch: 'bg-[#3a3f4b]' },
  { label: 'Danger', swatch: 'bg-danger/40' },
  { label: 'Goal', swatch: 'bg-gold' },
];

export default function EnvironmentGrid({
  grid,
  ppoPos,
  dynaqPos,
  ppoMood,
  dynaqMood,
  winner,
}: Props) {
  const n = grid.length || 10;
  const cellPct = 100 / n;

  const renderAgent = (pos: [number, number], agent: AgentId, mood: Props['ppoMood']) => (
    <motion.div
      className="pointer-events-none absolute z-10 grid place-items-center"
      style={{ width: `${cellPct}%`, height: `${cellPct}%` }}
      animate={{ left: `${pos[1] * cellPct}%`, top: `${pos[0] * cellPct}%` }}
      transition={{ type: 'spring', stiffness: 260, damping: 24 }}
    >
      <AgentCharacter agent={agent} mood={mood} size={26} />
    </motion.div>
  );

  return (
    <div className="card p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className="panel-title">Arena</p>
          <p className="text-sm font-semibold text-ink">Shared Grid-World 10 × 10</p>
        </div>
        {winner && (
          <motion.span
            initial={{ scale: 0.6, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className={`rounded-full px-3 py-1 text-xs font-bold ${
              winner === 'ppo'
                ? 'bg-ppo-soft text-ppo'
                : winner === 'dynaq'
                  ? 'bg-dynaq-soft text-dynaq'
                  : 'bg-sub/10 text-sub'
            }`}
          >
            {winner === 'draw' ? 'Draw' : `${winner.toUpperCase()} wins`}
          </motion.span>
        )}
      </div>

      <div className="relative mx-auto aspect-square w-full max-w-[460px] overflow-hidden rounded-lg border border-line bg-bg">
        <div
          className="grid h-full w-full"
          style={{
            gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${n}, minmax(0, 1fr))`,
          }}
        >
          {grid.flatMap((row, r) =>
            row.map((code, c) => {
              // Agents are rendered as overlays; treat their cells as empty bg.
              const isAgent = code === CELL.PPO_AGENT || code === CELL.DYNAQ_AGENT;
              const bgCode = isAgent ? CELL.EMPTY : code;
              return (
                <div
                  key={`${r}-${c}`}
                  className={`relative grid place-items-center border-[0.5px] border-line/60 ${cellStyle(bgCode)}`}
                >
                  <CellGlyph code={bgCode} r={r} c={c} />
                </div>
              );
            }),
          )}
        </div>

        {renderAgent(ppoPos, 'ppo', ppoMood)}
        {renderAgent(dynaqPos, 'dynaq', dynaqMood)}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2">
        {LEGEND.map((l) => (
          <span key={l.label} className="inline-flex items-center gap-1.5 text-xs text-sub">
            <span className={`h-3 w-3 rounded-[3px] ${l.swatch}`} />
            {l.label}
          </span>
        ))}
      </div>
    </div>
  );
}
