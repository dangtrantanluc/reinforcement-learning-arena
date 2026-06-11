import { motion } from 'framer-motion';
import {
  ACTION_NAMES,
  AGENT_META,
  type ActionName,
  type AgentId,
  type DQNState,
  type DynaQState,
  type PPOState,
} from '../types';

interface Props {
  agent: AgentId;
  ppo?: PPOState;
  dynaq?: DynaQState;
  dqn?: DQNState;
  winRate: number;
}

const ARROW: Record<ActionName, string> = {
  UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→', STAY: '•', BOMB: '✸',
};

// Static (purge-safe) class sets per agent.
const CLS: Record<AgentId, { text: string; bg: string; soft: string; bar: string }> = {
  ppo: { text: 'text-ppo', bg: 'bg-ppo', soft: 'bg-ppo-soft', bar: 'bg-ppo/40' },
  dynaq: { text: 'text-dynaq', bg: 'bg-dynaq', soft: 'bg-dynaq-soft', bar: 'bg-dynaq/40' },
  dqn: { text: 'text-dqn-c', bg: 'bg-dqn-c', soft: 'bg-dqn-soft', bar: 'bg-dqn-c/40' },
};

const SUBTITLE: Record<AgentId, string> = {
  ppo: 'Neural policy · Actor-Critic',
  dynaq: 'Tabular · Model-based planning',
  dqn: 'Neural · Off-policy Q-learning',
};

function Stat({ label, value, mono = true }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rounded-lg border border-line bg-bg/50 px-3 py-2">
      <p className="text-[10px] font-medium uppercase tracking-wide text-sub">{label}</p>
      <p className={`mt-0.5 text-sm font-semibold text-ink ${mono ? 'font-mono' : ''}`}>{value}</p>
    </div>
  );
}

export default function AgentStatsPanel({ agent, ppo, dynaq, dqn, winRate }: Props) {
  const cls = CLS[agent];
  const meta = AGENT_META[agent];

  const state = (agent === 'ppo' ? ppo : agent === 'dynaq' ? dynaq : dqn)!;
  const isPolicy = agent === 'ppo';
  const rawValues = isPolicy ? ppo!.action_probs : (state as DynaQState | DQNState).q_values;
  const bars = isPolicy ? ppo!.action_probs : normaliseQ(rawValues);
  const chosen = argmax(rawValues);
  const dead = !state.alive;

  return (
    <div className="card flex h-full flex-col p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`grid h-7 w-7 place-items-center rounded-md ${cls.bg} font-mono text-sm font-bold text-white`}>
            {meta.label}
          </span>
          <div>
            <p className={`text-sm font-bold ${cls.text}`}>
              {meta.name}{dead && <span className="ml-1 text-[10px] text-danger">✗ dead</span>}
            </p>
            <p className="text-[10px] text-sub">{SUBTITLE[agent]}</p>
          </div>
        </div>
        <span className={`rounded-full px-2.5 py-1 text-xs font-bold ${cls.soft} ${cls.text}`}>
          {(winRate * 100).toFixed(0)}% win
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <Stat label="Ep. Reward" value={state.episode_reward.toFixed(1)} />
        <Stat label="Total Wins" value={String(state.total_wins)} />
        <Stat label="Last Action" value={`${ARROW[state.last_action]} ${state.last_action}`} mono={false} />
        {agent === 'ppo' && <Stat label="Entropy" value={ppo!.entropy.toFixed(3)} />}
        {agent === 'dynaq' && <Stat label="Epsilon ε" value={dynaq!.epsilon.toFixed(3)} />}
        {agent === 'dqn' && <Stat label="Epsilon ε" value={dqn!.epsilon.toFixed(3)} />}
        {agent === 'ppo' && (<><Stat label="Policy Loss" value={ppo!.policy_loss.toFixed(3)} /><Stat label="Value Loss" value={ppo!.value_loss.toFixed(3)} /></>)}
        {agent === 'dynaq' && (<><Stat label="Q-table Size" value={dynaq!.q_table_size.toLocaleString()} /><Stat label="Planning" value={String(dynaq!.planning_steps)} /></>)}
        {agent === 'dqn' && (<><Stat label="TD Loss" value={dqn!.loss.toFixed(3)} /><Stat label="Replay Buffer" value={dqn!.buffer.toLocaleString()} /></>)}
      </div>

      <div className="mt-4">
        <p className="panel-title mb-2">{isPolicy ? 'Action Probabilities π(a|s)' : 'Q-values Q(s,a)'}</p>
        <div className="space-y-2">
          {ACTION_NAMES.map((a) => {
            const isChosen = a === chosen;
            const pct = Math.max(2, bars[a] * 100);
            const raw = rawValues[a];
            return (
              <div key={a}>
                <div className="mb-1 flex items-center justify-between text-xs">
                  <span className={`inline-flex items-center gap-1.5 font-medium ${isChosen ? cls.text : 'text-sub'}`}>
                    <span className="font-mono">{ARROW[a]}</span>
                    {a}
                    {isChosen && <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${cls.soft} ${cls.text}`}>best</span>}
                  </span>
                  <span className="font-mono font-semibold text-ink">
                    {isPolicy ? `${(raw * 100).toFixed(0)}%` : raw.toFixed(2)}
                  </span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-bg">
                  <motion.div
                    className={`h-full rounded-full ${isChosen ? cls.bg : cls.bar}`}
                    animate={{ width: `${pct}%` }}
                    transition={{ type: 'spring', stiffness: 200, damping: 26 }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function normaliseQ(q: Record<ActionName, number>): Record<ActionName, number> {
  const vals = ACTION_NAMES.map((a) => q[a]);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = max - min || 1;
  const out = {} as Record<ActionName, number>;
  for (const a of ACTION_NAMES) out[a] = (q[a] - min) / span;
  return out;
}

function argmax(rec: Record<ActionName, number>): ActionName {
  return ACTION_NAMES.reduce((best, a) => (rec[a] > rec[best] ? a : best), ACTION_NAMES[0]);
}
