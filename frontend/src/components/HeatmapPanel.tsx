import type { Heatmaps } from '../types';

interface Props {
  heatmaps: Heatmaps | null;
  onRefresh: () => void;
}

// Map a 0..1 intensity to a colour ramp (transparent → agent colour).
function cellColor(v: number, agent: 'ppo' | 'dynaq'): string {
  if (v <= 0) return 'transparent';
  const rgb = agent === 'ppo' ? '37,99,235' : '124,58,237';
  return `rgba(${rgb},${0.1 + v * 0.85})`;
}

function Grid({ data, agent }: { data: number[][]; agent: 'ppo' | 'dynaq' }) {
  const n = data.length || 10;
  return (
    <div
      className="grid aspect-square w-full overflow-hidden rounded-lg border border-line"
      style={{
        gridTemplateColumns: `repeat(${n}, minmax(0,1fr))`,
        gridTemplateRows: `repeat(${n}, minmax(0,1fr))`,
      }}
    >
      {data.flatMap((row, r) =>
        row.map((v, c) => (
          <div
            key={`${r}-${c}`}
            className="border-[0.5px] border-line/40"
            style={{ background: cellColor(v, agent) }}
            title={`(${r},${c}) ${(v * 100).toFixed(0)}%`}
          />
        )),
      )}
    </div>
  );
}

export default function HeatmapPanel({ heatmaps, onRefresh }: Props) {
  return (
    <div className="card p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className="panel-title">Visit Heatmaps</p>
          <p className="text-sm font-semibold text-ink">Where each agent spends its time</p>
        </div>
        <button className="btn-ghost text-xs" onClick={onRefresh}>Refresh</button>
      </div>
      {heatmaps ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <p className="mb-1.5 text-xs font-semibold text-ppo">PPO</p>
            <Grid data={heatmaps.ppo} agent="ppo" />
          </div>
          <div>
            <p className="mb-1.5 text-xs font-semibold text-dynaq">Dyna-Q</p>
            <Grid data={heatmaps.dynaq} agent="dynaq" />
          </div>
        </div>
      ) : (
        <div className="grid h-32 place-items-center text-xs text-sub">
          Click Refresh (or open this tab) to load heatmaps.
        </div>
      )}
      <p className="mt-3 text-[11px] text-sub">
        Brighter = visited more often. Reveals each agent's preferred routes and
        habits across all episodes so far.
      </p>
    </div>
  );
}
