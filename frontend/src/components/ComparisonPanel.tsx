import type { EvaluateResult, MetricsState } from '../types';

interface Props {
  metrics: MetricsState;
  ppoWins: number;
  dynaqWins: number;
  dqnWins?: number;
  evalResult: EvaluateResult | null;
}

/** Horizontal "tug of war" bar comparing two values. */
function VersusBar({
  label,
  ppo,
  dynaq,
  fmt = (v: number) => v.toFixed(2),
}: {
  label: string;
  ppo: number;
  dynaq: number;
  fmt?: (v: number) => string;
}) {
  const total = Math.abs(ppo) + Math.abs(dynaq) || 1;
  const ppoPct = (Math.abs(ppo) / total) * 100;
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="font-mono font-semibold text-ppo">{fmt(ppo)}</span>
        <span className="text-[11px] font-medium text-sub">{label}</span>
        <span className="font-mono font-semibold text-dynaq">{fmt(dynaq)}</span>
      </div>
      <div className="flex h-2.5 overflow-hidden rounded-full bg-bg">
        <div className="h-full bg-ppo transition-all" style={{ width: `${ppoPct}%` }} />
        <div className="h-full bg-dynaq transition-all" style={{ width: `${100 - ppoPct}%` }} />
      </div>
    </div>
  );
}

export default function ComparisonPanel({ metrics, ppoWins, dynaqWins, dqnWins, evalResult }: Props) {
  const hasDqn = dqnWins !== undefined;
  const winBars: { name: string; rate: number; bg: string; text: string }[] = [
    { name: 'PPO', rate: metrics.ppo_win_rate, bg: 'bg-ppo', text: 'text-ppo' },
    { name: 'Dyna-Q', rate: metrics.dynaq_win_rate, bg: 'bg-dynaq', text: 'text-dynaq' },
  ];
  if (hasDqn) winBars.push({ name: 'DQN', rate: metrics.dqn_win_rate ?? 0, bg: 'bg-dqn-c', text: 'text-dqn-c' });

  return (
    <div className="card p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className="panel-title">Head-to-Head</p>
          <p className="text-sm font-semibold text-ink">{hasDqn ? 'PPO vs Dyna-Q vs DQN' : 'PPO vs Dyna-Q'}</p>
        </div>
        <p className="font-mono text-sm">
          <span className="font-bold text-ppo">{ppoWins}</span>
          <span className="text-sub"> · </span>
          <span className="font-bold text-dynaq">{dynaqWins}</span>
          {hasDqn && (<><span className="text-sub"> · </span><span className="font-bold text-dqn-c">{dqnWins}</span></>)}
        </p>
      </div>

      {hasDqn ? (
        <div className="space-y-2.5">
          {winBars.map((b) => (
            <div key={b.name}>
              <div className="mb-1 flex justify-between text-xs">
                <span className={`font-medium ${b.text}`}>{b.name}</span>
                <span className="font-mono font-semibold text-ink">{(b.rate * 100).toFixed(0)}%</span>
              </div>
              <div className="h-2.5 overflow-hidden rounded-full bg-bg">
                <div className={`h-full rounded-full ${b.bg}`} style={{ width: `${b.rate * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="space-y-3">
          <VersusBar label="Win rate (last 100)" ppo={metrics.ppo_win_rate} dynaq={metrics.dynaq_win_rate} fmt={(v) => `${(v * 100).toFixed(0)}%`} />
          <VersusBar label="Avg reward" ppo={metrics.ppo_avg_reward} dynaq={metrics.dynaq_avg_reward} />
        </div>
      )}

      <div className="mt-3 rounded-lg bg-bg/60 px-3 py-2 text-center text-[11px] text-sub">
        Draw rate: <span className="font-mono font-semibold text-ink">{(metrics.draw_rate * 100).toFixed(0)}%</span>
      </div>

      {evalResult && (
        <div className="mt-4 rounded-lg border border-line bg-surface p-3">
          <p className="panel-title mb-2">
            Evaluation ({evalResult.episodes} eps · exploration off)
          </p>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <EvalStat label="PPO win" value={`${(evalResult.ppo_win_rate * 100).toFixed(0)}%`} tone="ppo" />
            <EvalStat label="Dyna-Q win" value={`${(evalResult.dynaq_win_rate * 100).toFixed(0)}%`} tone="dynaq" />
            <EvalStat label="PPO reward" value={evalResult.ppo_avg_reward.toFixed(1)} tone="ppo" />
            <EvalStat label="Dyna-Q reward" value={evalResult.dynaq_avg_reward.toFixed(1)} tone="dynaq" />
            <EvalStat label="PPO danger hits" value={String(evalResult.ppo_danger_hits)} tone="ppo" />
            <EvalStat label="Dyna-Q danger hits" value={String(evalResult.dynaq_danger_hits)} tone="dynaq" />
          </div>
          <p className="mt-2 text-center text-[11px] text-sub">
            Avg episode length: <span className="font-mono text-ink">{evalResult.avg_episode_length.toFixed(1)}</span>
          </p>
        </div>
      )}
    </div>
  );
}

function EvalStat({ label, value, tone }: { label: string; value: string; tone: 'ppo' | 'dynaq' }) {
  return (
    <div className="rounded-md bg-bg/50 px-2 py-1.5">
      <p className="text-[10px] text-sub">{label}</p>
      <p className={`font-mono text-sm font-semibold ${tone === 'ppo' ? 'text-ppo' : 'text-dynaq'}`}>{value}</p>
    </div>
  );
}
