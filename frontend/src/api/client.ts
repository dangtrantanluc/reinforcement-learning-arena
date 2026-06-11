// ─────────────────────────────────────────────────────────────
// API client for the FastAPI arena backend.
// By default we use SAME-ORIGIN requests (empty base) so the browser hits
// the Vite dev server, which proxies /api → backend (see vite.config.ts).
// This sidesteps CORS and the localhost IPv4/IPv6 mismatch entirely.
// Set VITE_API_URL to call a backend directly (e.g. in production).
// ─────────────────────────────────────────────────────────────

import type {
  ArenaState,
  EvaluateResult,
  FramesPayload,
  Heatmaps,
  HyperConfig,
  ReplayFull,
  ReplaySummary,
} from '../types';

const BASE = import.meta.env.VITE_API_URL ?? '';

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

/** Build the WebSocket URL for /ws/frames (same-origin → through Vite proxy). */
export function wsUrl(path: string): string {
  if (BASE) {
    return BASE.replace(/^http/, 'ws') + path;
  }
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}${path}`;
}

export const api = {
  getState: () => req<ArenaState>('/api/state'),
  getFrames: (since = 0, limit = 240) =>
    req<FramesPayload>(`/api/frames?since=${since}&limit=${limit}`),
  getHeatmaps: () => req<Heatmaps>('/api/heatmaps'),
  listReplays: () => req<{ replays: ReplaySummary[] }>('/api/replays'),
  getReplay: (episode: number) => req<ReplayFull>(`/api/replays/${episode}`),
  getConfig: () => req<HyperConfig>('/api/config'),
  updateConfig: (patch: Partial<HyperConfig>) =>
    req<{ ok: boolean; config: HyperConfig }>('/api/config', {
      method: 'POST',
      body: JSON.stringify(patch),
    }),
  startTraining: () => req<{ ok: boolean }>('/api/train/start', { method: 'POST' }),
  pauseTraining: () => req<{ ok: boolean }>('/api/train/pause', { method: 'POST' }),
  resetTraining: (seed?: number) =>
    req<{ ok: boolean }>(
      `/api/train/reset${seed !== undefined ? `?seed=${seed}` : ''}`,
      { method: 'POST' },
    ),
  setSpeed: (delay: number) =>
    req<{ ok: boolean }>(`/api/train/speed?delay=${delay}`, { method: 'POST' }),
  saveCheckpoint: () => req<{ ok: boolean }>('/api/checkpoint/save', { method: 'POST' }),
  loadCheckpoint: () =>
    req<{ ok: boolean; loaded: { ppo: boolean; dynaq: boolean } }>(
      '/api/checkpoint/load',
      { method: 'POST' },
    ),
  evaluateAgents: (episodes = 50) =>
    req<{ ok: boolean; result: EvaluateResult }>(
      `/api/evaluate?episodes=${episodes}`,
      { method: 'POST' },
    ),
};

export { BASE as API_BASE };
