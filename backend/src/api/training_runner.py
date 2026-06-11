"""TrainingRunner — runs SoloTrainer in a background thread with safe shared state.

The API thread reads `snapshot()` while the worker thread advances training. A
lock guards every access to the trainer so the two threads never touch torch
tensors / the Q-table concurrently.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from ..config import Config, default_config
from ..training.train_solo import SoloTrainer
from ..training.evaluate_solo import evaluate


class TrainingRunner:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or default_config()
        self._lock = threading.Lock()
        self._running = threading.Event()      # set = actively training
        self._stop = threading.Event()         # set = kill worker thread
        self._thread: Optional[threading.Thread] = None
        self.trainer = SoloTrainer(self.cfg)
        # Small delay between steps so the frontend (polling 500ms) can animate.
        self.step_delay = 0.06

    # ── Worker loop ─────────────────────────────────────────
    def _worker(self) -> None:
        while not self._stop.is_set():
            if not self._running.is_set():
                time.sleep(0.05)
                continue
            with self._lock:
                done = self.trainer.step_once()
                if done:
                    self.trainer._start_episode()
            if self.step_delay:
                time.sleep(self.step_delay)

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    # ── Controls ────────────────────────────────────────────
    def start(self) -> None:
        self._ensure_thread()
        self._running.set()

    def pause(self) -> None:
        self._running.clear()

    def reset(self) -> None:
        was_running = self._running.is_set()
        self._running.clear()
        with self._lock:
            self.trainer = SoloTrainer(self.cfg)
        if was_running:
            self._running.set()

    def evaluate(self, episodes: int = 100) -> dict:
        """Run evaluation using the trainer's CURRENT in-memory agents."""
        with self._lock:
            # Save current agents so evaluate() (which loads checkpoints) sees them.
            self.trainer.save()
        return evaluate(self.cfg, episodes=episodes, render=False)

    # ── State for the API ───────────────────────────────────
    @property
    def running(self) -> bool:
        return self._running.is_set()

    def snapshot(self) -> dict:
        with self._lock:
            state = self.trainer.live_state()
        state["running"] = self.running
        return state

    def shutdown(self) -> None:
        self._stop.set()
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
