interface Props {
  epsilon: number;
  planningSteps: number;
  qTableSize: number;
}

// Read-only hyperparameter overview for both algorithms (fixed in this demo).
export default function SettingsPanel({ epsilon, planningSteps, qTableSize }: Props) {
  const ppo: [string, string][] = [
    ['Learning rate', '3e-4'],
    ['Gamma γ', '0.99'],
    ['GAE λ', '0.95'],
    ['Clip ε', '0.20'],
    ['Entropy coef', '0.01'],
    ['Value coef', '0.50'],
    ['Rollout steps', '1024'],
    ['Batch size', '64'],
  ];
  const dynaq: [string, string][] = [
    ['Alpha α', '0.10'],
    ['Gamma γ', '0.99'],
    ['Epsilon (now)', epsilon.toFixed(3)],
    ['Epsilon end', '0.05'],
    ['Epsilon decay', '0.995'],
    ['Planning steps', String(planningSteps)],
    ['Q-table size', qTableSize.toLocaleString()],
    ['Actions', '5 (U/D/L/R/Stay)'],
  ];

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <Card title="PPO — Neural Policy" tone="ppo" rows={ppo} />
      <Card title="Dyna-Q — Tabular Planner" tone="dynaq" rows={dynaq} />
    </div>
  );
}

function Card({ title, tone, rows }: { title: string; tone: 'ppo' | 'dynaq'; rows: [string, string][] }) {
  const accent = tone === 'ppo' ? 'text-ppo' : 'text-dynaq';
  return (
    <div className="card p-5">
      <p className={`mb-3 text-sm font-semibold ${accent}`}>{title}</p>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {rows.map(([k, v]) => (
          <div key={k} className="rounded-lg border border-line bg-bg/50 p-2.5">
            <p className="text-[10px] font-medium uppercase tracking-wide text-sub">{k}</p>
            <p className="mt-1 font-mono text-sm font-semibold text-ink">{v}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
