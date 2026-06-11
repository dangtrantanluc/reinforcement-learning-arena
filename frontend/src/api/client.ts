// ─────────────────────────────────────────────────────────────
// API client for the FastAPI arena backend.
// By default we use SAME-ORIGIN requests (empty base) so the browser hits
// the Vite dev server, which proxies /api → backend (see vite.config.ts).
// This sidesteps CORS and the localhost IPv4/IPv6 mismatch entirely.
// Set VITE_API_URL to call a backend directly (e.g. in production).
// ─────────────────────────────────────────────────────────────

import type { ArenaState, EvaluateResult } from '../types';

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

export const api = {
  getState: () => req<ArenaState>('/api/state'),
  startTraining: () => req<{ ok: boolean }>('/api/train/start', { method: 'POST' }),
  pauseTraining: () => req<{ ok: boolean }>('/api/train/pause', { method: 'POST' }),
  resetTraining: () => req<{ ok: boolean }>('/api/train/reset', { method: 'POST' }),
  evaluateAgents: (episodes = 50) =>
    req<{ ok: boolean; result: EvaluateResult }>(
      `/api/evaluate?episodes=${episodes}`,
      { method: 'POST' },
    ),
};

export { BASE as API_BASE };
