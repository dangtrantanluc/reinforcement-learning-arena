import type { DynaQState, PPOState } from '../types';

interface Props {
  ppo: PPOState;
  dynaq: DynaQState;
}

/** Side-by-side "what each agent just decided and why" explainer. */
export default function ActionPanel({ ppo, dynaq }: Props) {
  const ppoProb = ppo.action_probs[ppo.last_action] ?? 0;
  const dynaqQ = dynaq.q_values[dynaq.last_action] ?? 0;

  return (
    <div className="card p-4">
      <p className="panel-title mb-3">Current Decision</p>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="rounded-lg border border-ppo/20 bg-ppo-soft/40 p-3">
          <p className="text-xs font-bold text-ppo">PPO</p>
          <p className="mt-1 text-sm text-ink">
            chose <span className="font-mono font-semibold">{ppo.last_action}</span> with probability{' '}
            <span className="font-mono font-semibold">{(ppoProb * 100).toFixed(0)}%</span>
          </p>
          <p className="mt-1 text-[11px] text-sub">
            Sampled from the neural policy π(a|s) — highest expected advantage.
          </p>
        </div>
        <div className="rounded-lg border border-dynaq/20 bg-dynaq-soft/40 p-3">
          <p className="text-xs font-bold text-dynaq">Dyna-Q</p>
          <p className="mt-1 text-sm text-ink">
            chose <span className="font-mono font-semibold">{dynaq.last_action}</span> with Q-value{' '}
            <span className="font-mono font-semibold">{dynaqQ.toFixed(2)}</span>
          </p>
          <p className="mt-1 text-[11px] text-sub">
            ε-greedy over the Q-table (ε={dynaq.epsilon.toFixed(2)}), refined by planning.
          </p>
        </div>
      </div>
    </div>
  );
}
