import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { HyperConfig } from '../types';

interface Props {
  epsilon: number;
  planningSteps: number;
  qTableSize: number;
}

type Field = { section: keyof HyperConfig; key: string; label: string; step: number };
const FIELDS: Field[] = [
  { section: 'ppo', key: 'lr', label: 'PPO learning rate', step: 0.0001 },
  { section: 'ppo', key: 'clip_eps', label: 'PPO clip ε', step: 0.05 },
  { section: 'ppo', key: 'entropy_coef', label: 'PPO entropy coef', step: 0.005 },
  { section: 'dynaq', key: 'alpha', label: 'Dyna-Q α', step: 0.05 },
  { section: 'dynaq', key: 'planning_steps', label: 'Dyna-Q planning', step: 1 },
  { section: 'dynaq', key: 'epsilon_decay', label: 'Dyna-Q ε-decay', step: 0.001 },
  { section: 'dqn', key: 'lr', label: 'DQN learning rate', step: 0.0001 },
  { section: 'dqn', key: 'batch_size', label: 'DQN batch size', step: 8 },
  { section: 'dqn', key: 'target_update', label: 'DQN target sync', step: 50 },
  { section: 'env', key: 'max_steps', label: 'Max steps / ep', step: 10 },
  { section: 'env', key: 'bomb_fuse', label: 'Bomb fuse', step: 1 },
  { section: 'env', key: 'bomb_range', label: 'Bomb range', step: 1 },
];

export default function SettingsPanel({ epsilon, planningSteps, qTableSize }: Props) {
  const [cfg, setCfg] = useState<HyperConfig | null>(null);
  const [draft, setDraft] = useState<Record<string, number>>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api.getConfig().then((c) => {
      setCfg(c);
      const d: Record<string, number> = {};
      for (const f of FIELDS) d[`${f.section}.${f.key}`] = (c[f.section] as Record<string, number>)[f.key];
      setDraft(d);
    }).catch(() => {});
  }, []);

  const apply = async () => {
    if (!cfg) return;
    setSaving(true);
    setSaved(false);
    const patch: Record<string, Record<string, number>> = {};
    for (const f of FIELDS) {
      patch[f.section] = patch[f.section] ?? {};
      patch[f.section][f.key] = draft[`${f.section}.${f.key}`];
    }
    try {
      const res = await api.updateConfig(patch as never);
      setCfg(res.config);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch { /* offline */ }
    finally { setSaving(false); }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Read label="Dyna-Q ε (live)" value={epsilon.toFixed(3)} />
        <Read label="Planning steps" value={String(planningSteps)} />
        <Read label="Q-table size" value={qTableSize.toLocaleString()} />
        <Read label="Algorithms" value="PPO·Dyna-Q·DQN" />
      </div>

      <div className="card p-5">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <p className="panel-title">Hyperparameters</p>
            <p className="text-sm font-semibold text-ink">Tune & re-train</p>
            <p className="mt-0.5 text-[11px] text-sub">Applying rebuilds the agents and restarts training from scratch.</p>
          </div>
          <div className="flex items-center gap-2">
            {saved && <span className="text-xs font-semibold text-success">Applied ✓</span>}
            <button className="btn-primary text-sm" onClick={apply} disabled={saving || !cfg}>
              {saving ? 'Applying…' : 'Apply & Reset'}
            </button>
          </div>
        </div>

        {cfg ? (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            {FIELDS.map((f) => {
              const k = `${f.section}.${f.key}`;
              return (
                <label key={k} className="rounded-lg border border-line bg-bg/50 p-2.5">
                  <span className="block text-[10px] font-medium uppercase tracking-wide text-sub">{f.label}</span>
                  <input
                    type="number"
                    step={f.step}
                    value={draft[k] ?? 0}
                    onChange={(e) => setDraft((d) => ({ ...d, [k]: Number(e.target.value) }))}
                    className="mt-1 w-full rounded border border-line bg-bg px-2 py-1 font-mono text-sm text-ink outline-none focus:border-primary"
                  />
                </label>
              );
            })}
          </div>
        ) : (
          <p className="text-xs text-sub">Loading config…</p>
        )}
      </div>
    </div>
  );
}

function Read({ label, value }: { label: string; value: string }) {
  return (
    <div className="card p-3">
      <p className="text-[10px] font-medium uppercase tracking-wide text-sub">{label}</p>
      <p className="mt-1 font-mono text-sm font-semibold text-ink">{value}</p>
    </div>
  );
}
