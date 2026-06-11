import { useCallback, useEffect, useRef, useState } from 'react';
import { api, wsUrl } from '../api/client';
import type {
  DQNState,
  DynaQState,
  EvaluateResult,
  Frame,
  FramesPayload,
  Heatmaps,
  HistoryPoint,
  MetricsState,
  PPOState,
} from '../types';

// Display cadence: how fast we replay frames from the queue (ms per frame).
// This is the *viewing* speed — independent of how fast the backend trains.
const REPLAY_MS = 120;
// If the queue grows beyond this, fast-forward (training outran the viewer).
const MAX_QUEUE = 400;
const DROP_TO = 80;

/**
 * Streams per-step frames over WebSocket (falling back to polling) and replays
 * them ONE AT A TIME at a fixed cadence. Because every frame is a single env
 * step, agents move cell-by-cell and never appear to slide through walls (A1).
 *
 * Panels (agent stats / metrics / charts) update from the latest payload, while
 * the grid follows the replayed frame.
 */
export function useArenaStream() {
  const [current, setCurrent] = useState<Frame | null>(null);
  const [ppo, setPpo] = useState<PPOState | null>(null);
  const [dynaq, setDynaq] = useState<DynaQState | null>(null);
  const [dqn, setDqn] = useState<DQNState | null>(null);
  const [metrics, setMetrics] = useState<MetricsState | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [connected, setConnected] = useState(false);
  const [queueLen, setQueueLen] = useState(0);

  const [busy, setBusy] = useState(false);
  const [evalResult, setEvalResult] = useState<EvaluateResult | null>(null);
  const [heatmaps, setHeatmaps] = useState<Heatmaps | null>(null);

  const queue = useRef<Frame[]>([]);
  const lastSeq = useRef(0);
  const wsRef = useRef<WebSocket | null>(null);
  const pollRef = useRef<number | null>(null);

  // Absorb a payload: append new frames to the replay queue + refresh panels.
  const absorb = useCallback((p: FramesPayload) => {
    setConnected(true);
    setRunning(p.running);
    setPpo(p.ppo);
    setDynaq(p.dynaq);
    setDqn(p.dqn ?? null);
    setMetrics(p.metrics);
    setHistory(p.history);
    setLogs(p.logs);
    if (p.frames.length) {
      lastSeq.current = p.frames[p.frames.length - 1].seq;
      queue.current.push(...p.frames);
      // Fast-forward if we fell too far behind training.
      if (queue.current.length > MAX_QUEUE) {
        queue.current = queue.current.slice(-DROP_TO);
      }
    }
  }, []);

  // ── Frame fetching: HTTP polling primary, WebSocket as a bonus ──
  // Polling guarantees data even if the WS handshake races page load (which
  // otherwise logs a noisy "connection interrupted"). The WS, once open,
  // delivers frames faster; if it never opens we lose nothing.
  useEffect(() => {
    let stopped = false;

    const tick = async () => {
      try {
        absorb(await api.getFrames(lastSeq.current));
      } catch {
        setConnected(false);
      }
    };
    tick();
    pollRef.current = window.setInterval(tick, 350);

    // Try WebSocket after the page has settled (avoids the load-time abort).
    const wsTimer = window.setTimeout(() => {
      if (stopped) return;
      try {
        const ws = new WebSocket(wsUrl('/ws/frames'));
        wsRef.current = ws;
        ws.onmessage = (ev) => {
          try { absorb(JSON.parse(ev.data) as FramesPayload); } catch { /* ignore */ }
        };
        // On any WS trouble just close it — polling keeps the app alive silently.
        ws.onerror = () => { try { ws.close(); } catch { /* noop */ } };
        ws.onclose = () => { wsRef.current = null; };
      } catch { /* polling continues */ }
    }, 1200);

    return () => {
      stopped = true;
      window.clearTimeout(wsTimer);
      try { wsRef.current?.close(); } catch { /* noop */ }
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, [absorb]);

  // ── Replay loop: pop one frame per tick ───────────────────
  useEffect(() => {
    const id = window.setInterval(() => {
      const q = queue.current;
      setQueueLen(q.length);
      if (q.length === 0) return;
      // If badly behind, step 2 frames to catch up without losing smoothness.
      const step = q.length > 60 ? 2 : 1;
      let frame: Frame | undefined;
      for (let i = 0; i < step; i++) frame = q.shift();
      if (frame) setCurrent(frame);
    }, REPLAY_MS);
    return () => window.clearInterval(id);
  }, []);

  // ── Controls ──────────────────────────────────────────────
  const start = useCallback(async () => { try { await api.startTraining(); setRunning(true); } catch {} }, []);
  const pause = useCallback(async () => { try { await api.pauseTraining(); setRunning(false); } catch {} }, []);
  const reset = useCallback(async (seed?: number) => {
    try {
      setEvalResult(null);
      queue.current = [];
      lastSeq.current = 0;
      setCurrent(null);
      await api.resetTraining(seed);
    } catch {}
  }, []);
  const setSpeed = useCallback(async (delay: number) => { try { await api.setSpeed(delay); } catch {} }, []);
  const saveCheckpoint = useCallback(async () => { try { await api.saveCheckpoint(); } catch {} }, []);
  const loadCheckpoint = useCallback(async () => {
    try { queue.current = []; lastSeq.current = 0; await api.loadCheckpoint(); } catch {}
  }, []);
  const evaluate = useCallback(async () => {
    setBusy(true);
    try { const r = await api.evaluateAgents(50); setEvalResult(r.result); }
    catch {} finally { setBusy(false); }
  }, []);
  const fetchHeatmaps = useCallback(async () => {
    try { setHeatmaps(await api.getHeatmaps()); } catch {}
  }, []);

  return {
    current, ppo, dynaq, dqn, metrics, history, logs, running, connected, queueLen,
    busy, evalResult, heatmaps,
    start, pause, reset, setSpeed, saveCheckpoint, loadCheckpoint, evaluate, fetchHeatmaps,
  };
}
