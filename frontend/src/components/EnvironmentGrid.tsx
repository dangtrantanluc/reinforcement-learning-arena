import { motion } from 'framer-motion';
import { CELL, AGENT_META, type AgentId, type Frame, type Winner } from '../types';
import AgentCharacter from './AgentCharacter';

interface Props {
  frame: Frame;
  winner: Winner;
}

// Background style per STATIC cell code (agents/bombs drawn as overlays).
function cellStyle(code: number): string {
  switch (code) {
    case CELL.WALL: return 'bg-[#3a3f4b]';
    case CELL.BOX: return 'bg-box/25';
    case CELL.REWARD: return 'bg-success/15';
    case CELL.DANGER: return 'bg-danger/15';
    case CELL.GOAL: return 'bg-gold/25';
    case CELL.EXPLOSION: return 'bg-warning/40';
    default: return 'bg-surface';
  }
}

function CellGlyph({ code }: { code: number }) {
  if (code === CELL.REWARD)
    return (
      <motion.span
        className="block h-3 w-3 rounded-full bg-success shadow-[0_0_8px_rgba(34,197,94,0.6)]"
        animate={{ scale: [1, 1.18, 1] }}
        transition={{ duration: 1.4, repeat: Infinity }}
      />
    );
  if (code === CELL.GOAL) return <span className="text-sm">🏁</span>;
  if (code === CELL.BOX)
    return <span className="grid h-5 w-5 place-items-center rounded-[3px] bg-box/70 text-[9px]">📦</span>;
  if (code === CELL.EXPLOSION)
    return (
      <motion.span
        className="text-sm"
        initial={{ scale: 0.4, opacity: 0 }}
        animate={{ scale: [0.4, 1.2, 1], opacity: [0, 1, 0.8] }}
        transition={{ duration: 0.4 }}
      >💥</motion.span>
    );
  if (code === CELL.BOMB)
    return (
      <motion.span
        className="grid h-5 w-5 place-items-center rounded-full bg-ink text-[11px]"
        animate={{ scale: [1, 1.15, 1] }}
        transition={{ duration: 0.5, repeat: Infinity }}
      >💣</motion.span>
    );
  return null;
}

const LEGEND: { label: string; swatch: string }[] = [
  { label: 'PPO (P)', swatch: 'bg-ppo' },
  { label: 'Dyna-Q (D)', swatch: 'bg-dynaq' },
  { label: 'DQN (Q)', swatch: 'bg-dqn-c' },
  { label: 'Reward', swatch: 'bg-success' },
  { label: 'Box', swatch: 'bg-box/70' },
  { label: 'Bomb', swatch: 'bg-ink' },
  { label: 'Wall', swatch: 'bg-[#3a3f4b]' },
  { label: 'Danger', swatch: 'bg-danger/40' },
  { label: 'Goal', swatch: 'bg-gold' },
];

const AGENT_CODES = [CELL.PPO_AGENT, CELL.DYNAQ_AGENT, CELL.DQN_AGENT] as number[];

// Static (purge-safe) winner badge classes.
function winnerBadge(winner: Winner): string {
  switch (winner) {
    case 'ppo': return 'bg-ppo-soft text-ppo';
    case 'dynaq': return 'bg-dynaq-soft text-dynaq';
    case 'dqn': return 'bg-dqn-soft text-dqn-c';
    default: return 'bg-sub/10 text-sub';
  }
}

export default function EnvironmentGrid({ frame, winner }: Props) {
  const grid = frame.grid;
  const n = grid.length || 10;
  const cellPct = 100 / n;

  // Which agents are present + alive this frame.
  const agentIds = Object.keys(frame.positions) as AgentId[];

  function moodFor(agent: AgentId): 'idle' | 'moving' | 'reward' | 'danger' | 'winner' {
    if (frame.winner === agent) return 'winner';
    const evs = frame.events.filter((e) => e.agent === agent || e.agent === 'both');
    if (evs.some((e) => e.type === 'danger' || e.type === 'kill')) return 'danger';
    if (evs.some((e) => e.type === 'reward')) return 'reward';
    return 'moving';
  }

  return (
    <div className="card p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className="panel-title">Arena</p>
          <p className="text-sm font-semibold text-ink">Shared Bomberman Grid 10 × 10</p>
        </div>
        {winner && (
          <motion.span
            initial={{ scale: 0.6, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className={`rounded-full px-3 py-1 text-xs font-bold ${winnerBadge(winner)}`}
          >
            {winner === 'draw' ? 'Draw' : `${AGENT_META[winner as AgentId].name} wins`}
          </motion.span>
        )}
      </div>

      <div className="relative mx-auto aspect-square w-full max-w-[460px] overflow-hidden rounded-lg border border-line bg-bg">
        <div
          className="grid h-full w-full"
          style={{
            gridTemplateColumns: `repeat(${n}, minmax(0,1fr))`,
            gridTemplateRows: `repeat(${n}, minmax(0,1fr))`,
          }}
        >
          {grid.flatMap((row, r) =>
            row.map((code, c) => {
              const isAgent = AGENT_CODES.includes(code);
              const bg = isAgent ? CELL.EMPTY : code;
              return (
                <div
                  key={`${r}-${c}`}
                  className={`relative grid place-items-center border-[0.5px] border-line/60 ${cellStyle(bg)}`}
                >
                  <CellGlyph code={bg} />
                </div>
              );
            }),
          )}
        </div>

        {agentIds.map((a) => {
          const pos = frame.positions[a];
          const alive = frame.alive[a];
          if (!alive) return null;
          return (
            <motion.div
              key={a}
              className="pointer-events-none absolute z-10 grid place-items-center"
              style={{ width: `${cellPct}%`, height: `${cellPct}%` }}
              animate={{ left: `${pos[1] * cellPct}%`, top: `${pos[0] * cellPct}%` }}
              transition={{ type: 'spring', stiffness: 260, damping: 24 }}
            >
              <AgentCharacter agent={a} mood={moodFor(a)} size={26} />
            </motion.div>
          );
        })}
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
