import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import type { ArenaState, EvaluateResult } from '../types';

const POLL_MS = 500;

/**
 * Polls GET /api/state every 500ms and exposes control actions.
 * Tracks a `connected` flag so the UI can show an offline state if the
 * backend isn't running yet.
 */
export function useArenaState() {
  const [state, setState] = useState<ArenaState | null>(null);
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [evalResult, setEvalResult] = useState<EvaluateResult | null>(null);
  const timer = useRef<number | null>(null);

  const poll = useCallback(async () => {
    try {
      const s = await api.getState();
      setState(s);
      setConnected(true);
    } catch {
      setConnected(false);
    }
  }, []);

  useEffect(() => {
    poll();
    timer.current = window.setInterval(poll, POLL_MS);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [poll]);

  const start = useCallback(async () => {
    try { await api.startTraining(); await poll(); } catch { /* offline */ }
  }, [poll]);

  const pause = useCallback(async () => {
    try { await api.pauseTraining(); await poll(); } catch { /* offline */ }
  }, [poll]);

  const reset = useCallback(async () => {
    try { setEvalResult(null); await api.resetTraining(); await poll(); } catch { /* offline */ }
  }, [poll]);

  const evaluate = useCallback(async () => {
    setBusy(true);
    try {
      const res = await api.evaluateAgents(50);
      setEvalResult(res.result);
    } catch {
      /* offline */
    } finally {
      setBusy(false);
    }
  }, []);

  return { state, connected, busy, evalResult, start, pause, reset, evaluate };
}
