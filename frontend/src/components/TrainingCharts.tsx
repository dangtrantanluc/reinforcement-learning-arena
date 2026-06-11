import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { HistoryPoint } from '../types';

const AXIS = { fontSize: 11, fill: '#6e6f73' };
const GRID = '#ececec';
const tooltip = {
  borderRadius: 8,
  border: '1px solid #dcdcdc',
  fontSize: 12,
  boxShadow: '0 4px 16px rgba(47,58,85,0.12)',
};

function ChartCard({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <div className="card p-4">
      <div className="mb-2">
        <p className="text-sm font-semibold text-ink">{title}</p>
        <p className="text-[11px] text-sub">{subtitle}</p>
      </div>
      <div className="h-[160px] w-full">{children}</div>
    </div>
  );
}

function Empty() {
  return <div className="grid h-full place-items-center text-xs text-sub">Start training to populate charts.</div>;
}

export default function TrainingCharts({ history }: { history: HistoryPoint[] }) {
  const has = history.length > 0;
  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
      <ChartCard title="Reward per Episode" subtitle="PPO vs Dyna-Q cumulative reward">
        {has ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
              <XAxis dataKey="episode" tick={AXIS} stroke={GRID} />
              <YAxis tick={AXIS} stroke={GRID} width={42} />
              <Tooltip contentStyle={tooltip} />
              <Line type="monotone" dataKey="ppo_reward" name="PPO" stroke="#2563eb" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="dynaq_reward" name="Dyna-Q" stroke="#7c3aed" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : <Empty />}
      </ChartCard>

      <ChartCard title="Win Rate" subtitle="Rolling win rate (last 100 episodes)">
        {has ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
              <XAxis dataKey="episode" tick={AXIS} stroke={GRID} />
              <YAxis tick={AXIS} stroke={GRID} width={42} domain={[0, 1]} />
              <Tooltip contentStyle={tooltip} formatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
              <Line type="monotone" dataKey="ppo_win_rate" name="PPO" stroke="#2563eb" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="dynaq_win_rate" name="Dyna-Q" stroke="#7c3aed" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : <Empty />}
      </ChartCard>

      <ChartCard title="Episode Length" subtitle="Steps until an agent wins / timeout">
        {has ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
              <XAxis dataKey="episode" tick={AXIS} stroke={GRID} />
              <YAxis tick={AXIS} stroke={GRID} width={42} />
              <Tooltip contentStyle={tooltip} />
              <Line type="monotone" dataKey="episode_length" name="Length" stroke="#a29f76" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : <Empty />}
      </ChartCard>

      <ChartCard title="PPO Loss & Dyna-Q ε" subtitle="PPO policy loss vs Dyna-Q exploration decay">
        {has ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
              <XAxis dataKey="episode" tick={AXIS} stroke={GRID} />
              <YAxis yAxisId="l" tick={AXIS} stroke={GRID} width={42} />
              <YAxis yAxisId="r" orientation="right" tick={AXIS} stroke={GRID} width={36} domain={[0, 1]} />
              <Tooltip contentStyle={tooltip} />
              <Line yAxisId="l" type="monotone" dataKey="ppo_policy_loss" name="PPO loss" stroke="#2563eb" strokeWidth={2} dot={false} />
              <Line yAxisId="r" type="monotone" dataKey="dynaq_epsilon" name="ε" stroke="#7c3aed" strokeWidth={2} dot={false} strokeDasharray="4 2" />
            </LineChart>
          </ResponsiveContainer>
        ) : <Empty />}
      </ChartCard>
    </div>
  );
}
