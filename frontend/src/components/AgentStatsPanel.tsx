import { motion } from 'framer-motion';
import {
  ACTION_NAMES,
  type ActionName,
  type AgentId,
  type DynaQState,
  type PPOState,
} from '../types';

interface Props {
  agent: AgentId;
  ppo?: PPOState;
  dynaq?: DynaQState;
  winRate: number;
}

const ARROW: Record<ActionName, string> = {
  UP: '↑',
  DOWN: '↓',
  LEFT: '←',
  RIGHT: '→',
  STAY: '•',
};

function Stat({ label, value, mono = true }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rounded-lg border border-line bg-bg/50 px-3 py-2">
      <p className="text-[10px] font-medium uppercase tracking-wide text-sub">{label}</p>
      <p className={`mt-0.5 text-sm font-semibold text-ink ${mono ? 'font-mono' : ''}`}>{value}</p>
    </div>
  );
}

export default function AgentStatsPanel({ agent, ppo, dynaq, winRate }: Props) {
  const isPPO = agent === 'ppo';
  const accent = isPPO ? 'text-ppo' : 'text-dynaq';
  const accentBg = isPPO ? 'bg-ppo' : 'bg-dynaq';
  const softBg = isPPO ? 'bg-ppo-soft' : 'bg-dynaq-soft';

  const reward = isPPO ? ppo!.episode_reward : dynaq!.episode_reward;
  const wins = isPPO ? ppo!.total_wins : dynaq!.total_wins;
  const lastAction = isPPO ? ppo!.last_action : dynaq!.last_action;

  // Bar values: PPO uses probabilities (0..1), Dyna-Q uses normalised Q-values.
  const bars = isPPO ? ppo!.action_probs : normaliseQ(dynaq!.q_values);
  const rawValues = isPPO ? ppo!.action_probs : dynaq!.q_values;
  const chosen = argmax(rawValues);

  return (
    <div className="card flex h-full flex-col p-4">
      {/* header */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`grid h-7 w-7 place-items-center rounded-md ${accentBg} font-mono text-sm font-bold text-white`}>
            {isPPO ? 'P' : 'D'}
          </span>
          <div>
            <p className={`text-sm font-bold ${accent}`}>{isPPO ? 'PPO' : 'Dyna-Q'}</p>
            <p className="text-[10px] text-sub">
              {isPPO ? 'Neural policy · Actor-Critic' : 'Tabular · Model-based planning'}
            </p>
          </div>
        </div>
        <span className={`rounded-full px-2.5 py-1 text-xs font-bold ${softBg} ${accent}`}>
          {(winRate * 100).toFixed(0)}% win
        </span>
      </div>

      {/* stats grid */}
      <div className="grid grid-cols-2 gap-2">
        <Stat label="Ep. Reward" value={reward.toFixed(1)} />
        <Stat label="Total Wins" value={String(wins)} />
        <Stat label="Last Action" value={`${ARROW[lastAction]} ${lastAction}`} mono={false} />
        {isPPO ? (
          <Stat label="Entropy" value={ppo!.entropy.toFixed(3)} />
        ) : (
          <Stat label="Epsilon ε" value={dynaq!.epsilon.toFixed(3)} />
        )}
        {isPPO ? (
          <>
            <Stat label="Policy Loss" value={ppo!.policy_loss.toFixed(3)} />
            <Stat label="Value Loss" value={ppo!.value_loss.toFixed(3)} />
          </>
        ) : (
          <>
            <Stat label="Q-table Size" value={dynaq!.q_table_size.toLocaleString()} />
            <Stat label="Planning Steps" value={String(dynaq!.planning_steps)} />
          </>
        )}
      </div>

      {/* action bars */}
      <div className="mt-4">
        <p className="panel-title mb-2">
          {isPPO ? 'Action Probabilities π(a|s)' : 'Q-values Q(s,a)'}
        </p>
        <div className="space-y-2">
          {ACTION_NAMES.map((a) => {
            const isChosen = a === chosen;
            const pct = Math.max(2, bars[a] * 100);
            const raw = rawValues[a];
            return (
              <div key={a}>
                <div className="mb-1 flex items-center justify-between text-xs">
                  <span className={`inline-flex items-center gap-1.5 font-medium ${isChosen ? accent : 'text-sub'}`}>
                    <span className="font-mono">{ARROW[a]}</span>
                    {a}
                    {isChosen && (
                      <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${softBg} ${accent}`}>
                        best
                      </span>
                    )}
                  </span>
                  <span className="font-mono font-semibold text-ink">
                    {isPPO ? `${(raw * 100).toFixed(0)}%` : raw.toFixed(2)}
                  </span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-bg">
                  <motion.div
                    className={`h-full rounded-full ${isChosen ? accentBg : isPPO ? 'bg-ppo/40' : 'bg-dynaq/40'}`}
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

/** Map Q-values to [0,1] bar widths via min-max (handles negatives). */
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
