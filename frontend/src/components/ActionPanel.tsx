import type { DQNState, DynaQState, PPOState } from '../types';

interface Props {
  ppo: PPOState;
  dynaq: DynaQState;
  dqn?: DQNState | null;
}

/** Side-by-side "what each agent just decided and why" explainer. */
export default function ActionPanel({ ppo, dynaq, dqn }: Props) {
  const ppoProb = ppo.action_probs[ppo.last_action] ?? 0;
  const dynaqQ = dynaq.q_values[dynaq.last_action] ?? 0;
  const dqnQ = dqn?.q_values[dqn.last_action] ?? 0;

  return (
    <div className="card p-4">
      <p className="panel-title mb-3">Current Decision</p>
      <div className={`grid grid-cols-1 gap-3 ${dqn ? 'sm:grid-cols-3' : 'sm:grid-cols-2'}`}>
        <div className="rounded-lg border border-ppo/20 bg-ppo-soft/40 p-3">
          <p className="text-xs font-bold text-ppo">PPO</p>
          <p className="mt-1 text-sm text-ink">
            chose <span className="font-mono font-semibold">{ppo.last_action}</span> @{' '}
            <span className="font-mono font-semibold">{(ppoProb * 100).toFixed(0)}%</span>
          </p>
          <p className="mt-1 text-[11px] text-sub">Sampled from neural policy π(a|s).</p>
        </div>
        <div className="rounded-lg border border-dynaq/20 bg-dynaq-soft/40 p-3">
          <p className="text-xs font-bold text-dynaq">Dyna-Q</p>
          <p className="mt-1 text-sm text-ink">
            chose <span className="font-mono font-semibold">{dynaq.last_action}</span> Q=
            <span className="font-mono font-semibold">{dynaqQ.toFixed(2)}</span>
          </p>
          <p className="mt-1 text-[11px] text-sub">ε-greedy + planning (ε={dynaq.epsilon.toFixed(2)}).</p>
        </div>
        {dqn && (
          <div className="rounded-lg border border-dqn-c/20 bg-dqn-soft/40 p-3">
            <p className="text-xs font-bold text-dqn-c">DQN</p>
            <p className="mt-1 text-sm text-ink">
              chose <span className="font-mono font-semibold">{dqn.last_action}</span> Q=
              <span className="font-mono font-semibold">{dqnQ.toFixed(2)}</span>
            </p>
            <p className="mt-1 text-[11px] text-sub">argmax Q-net (ε={dqn.epsilon.toFixed(2)}), replay-trained.</p>
          </div>
        )}
      </div>
    </div>
  );
}
