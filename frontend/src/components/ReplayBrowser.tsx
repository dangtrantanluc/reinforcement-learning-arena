import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import { AGENT_META, type AgentId, type ReplayFull, type ReplaySummary } from '../types';
import EnvironmentGrid from './EnvironmentGrid';

const PLAY_MS = 180;

// Purge-safe winner tone classes.
function winnerTone(w: AgentId | 'draw'): string {
  switch (w) {
    case 'ppo': return 'text-ppo';
    case 'dynaq': return 'text-dynaq';
    case 'dqn': return 'text-dqn-c';
    default: return 'text-sub';
  }
}

/** Browse finished matches and replay them step-by-step (B7). */
export default function ReplayBrowser() {
  const [list, setList] = useState<ReplaySummary[]>([]);
  const [replay, setReplay] = useState<ReplayFull | null>(null);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try { setList((await api.listReplays()).replays.slice().reverse()); } catch {}
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const open = useCallback(async (episode: number) => {
    try {
      const r = await api.getReplay(episode);
      setReplay(r);
      setIdx(0);
      setPlaying(true);
    } catch {}
  }, []);

  useEffect(() => {
    if (!playing || !replay) return;
    timer.current = window.setInterval(() => {
      setIdx((i) => {
        if (i >= replay.frames.length - 1) { setPlaying(false); return i; }
        return i + 1;
      });
    }, PLAY_MS);
    return () => { if (timer.current) window.clearInterval(timer.current); };
  }, [playing, replay]);

  const frame = replay?.frames[idx];

  return (
    <div className="card p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className="panel-title">Match Replays</p>
          <p className="text-sm font-semibold text-ink">Re-watch finished episodes</p>
        </div>
        <button className="btn-ghost text-xs" onClick={refresh}>Refresh</button>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,260px)_1fr]">
        {/* match list */}
        <div className="scroll-thin max-h-[420px] space-y-1.5 overflow-y-auto pr-1">
          {list.length === 0 && <p className="text-xs text-sub">No finished matches yet — let training run.</p>}
          {list.map((m) => {
            const w = m.winner as AgentId | 'draw';
            const tone = winnerTone(w);
            return (
              <button
                key={m.episode}
                onClick={() => open(m.episode)}
                className={`flex w-full items-center justify-between rounded-lg border px-3 py-2 text-left text-xs transition-colors ${
                  replay?.episode === m.episode ? 'border-primary bg-bg' : 'border-line hover:bg-bg'
                }`}
              >
                <span className="font-mono text-ink">ep {m.episode}</span>
                <span className={`font-semibold ${tone}`}>
                  {w === 'draw' ? 'Draw' : (AGENT_META[w as AgentId]?.name ?? w)}
                </span>
                <span className="text-sub">{m.length} steps</span>
              </button>
            );
          })}
        </div>

        {/* playback */}
        <div>
          {replay && frame ? (
            <>
              <EnvironmentGrid frame={frame} winner={frame.winner} />
              <div className="mt-2 flex items-center gap-3">
                <button className="btn-ghost text-xs" onClick={() => setPlaying((p) => !p)}>
                  {playing ? 'Pause' : 'Play'}
                </button>
                <input
                  type="range" min={0} max={Math.max(0, replay.frames.length - 1)} value={idx}
                  onChange={(e) => { setPlaying(false); setIdx(Number(e.target.value)); }}
                  className="flex-1 accent-primary"
                />
                <span className="font-mono text-xs text-sub">{idx + 1}/{replay.frames.length}</span>
              </div>
            </>
          ) : (
            <div className="grid h-[320px] place-items-center rounded-lg border border-dashed border-line text-sm text-sub">
              Select a match on the left to replay it.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
