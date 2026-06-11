"""TrainingRunner — runs SoloTrainer in a background thread with safe shared state.

The API thread reads `snapshot()` while the worker thread advances training. A
lock guards every access to the trainer so the two threads never touch torch
tensors / the Q-table concurrently.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Optional

from ..config import Config, default_config
from ..training.train_solo import SoloTrainer
from ..training.evaluate_solo import evaluate


def _json_safe(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with 0.0 so JSON serialisation never
    fails. NaN can appear if training diverges; the UI shows 0 instead of 500."""
    if isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


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

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset everything. Optionally reseed the map (A3 New Map / seed)."""
        was_running = self._running.is_set()
        self._running.clear()
        with self._lock:
            if seed is not None:
                self.cfg.env.seed = seed
                self.cfg.train.seed = seed
            self.trainer = SoloTrainer(self.cfg)
        if was_running:
            self._running.set()

    def evaluate(self, episodes: int = 100) -> dict:
        """Run evaluation using the trainer's CURRENT in-memory agents."""
        with self._lock:
            # Save current agents so evaluate() (which loads checkpoints) sees them.
            self.trainer.save()
        return evaluate(self.cfg, episodes=episodes, render=False)

    def save_checkpoint(self) -> dict:
        with self._lock:
            self.trainer.save()
            return {
                "ppo": self.cfg.train.ppo_ckpt,
                "dynaq": self.cfg.train.dynaq_ckpt,
                "episode": self.trainer.episode,
            }

    def load_checkpoint(self) -> dict:
        """Load saved PPO + Dyna-Q weights into the live trainer (A4)."""
        was_running = self._running.is_set()
        self._running.clear()
        with self._lock:
            loaded = self.trainer.load_checkpoints()
        if was_running:
            self._running.set()
        return loaded

    # ── State for the API ───────────────────────────────────
    @property
    def running(self) -> bool:
        return self._running.is_set()

    def snapshot(self) -> dict:
        with self._lock:
            state = self.trainer.live_state()
        state["running"] = self.running
        return _json_safe(state)

    def frames_since(self, seq: int, limit: int = 240) -> dict:
        """Frames newer than `seq` for step-by-step replay, plus the metadata
        slice the UI needs (agent stats / metrics) so it can render panels."""
        with self._lock:
            frames = self.trainer.get_frames_since(seq, limit)
            state = self.trainer.live_state()
        payload = {
            "running": self.running,
            "frames": frames,
            "ppo": state["ppo"],
            "dynaq": state["dynaq"],
            "metrics": state["metrics"],
            "history": state["history"],
            "logs": state["logs"],
            "last_frame_seq": state["last_frame_seq"],
        }
        if "dqn" in state:
            payload["dqn"] = state["dqn"]
        return _json_safe(payload)

    def heatmaps(self) -> dict:
        with self._lock:
            return self.trainer.heatmaps()

    def list_replays(self) -> list:
        with self._lock:
            return self.trainer.list_replays()

    def get_replay(self, episode: int):
        with self._lock:
            return self.trainer.get_replay(episode)

    def set_speed(self, delay: float) -> None:
        """Adjust env-step cadence (seconds). 0 = as fast as possible."""
        self.step_delay = max(0.0, min(2.0, delay))

    # ── Hyperparameters (C11) ───────────────────────────────
    def get_config(self) -> dict:
        """Editable hyperparameters surfaced to the tuning panel."""
        c = self.cfg
        return {
            "ppo": {"lr": c.ppo.lr, "gamma": c.ppo.gamma, "gae_lambda": c.ppo.gae_lambda,
                    "clip_eps": c.ppo.clip_eps, "entropy_coef": c.ppo.entropy_coef,
                    "rollout_steps": c.ppo.rollout_steps},
            "dynaq": {"alpha": c.dynaq.alpha, "gamma": c.dynaq.gamma,
                      "epsilon_decay": c.dynaq.epsilon_decay, "planning_steps": c.dynaq.planning_steps},
            "dqn": {"lr": c.dqn.lr, "gamma": c.dqn.gamma, "epsilon_decay": c.dqn.epsilon_decay,
                    "batch_size": c.dqn.batch_size, "target_update": c.dqn.target_update},
            "env": {"max_steps": c.env.max_steps, "bomb_fuse": c.env.bomb_fuse,
                    "bomb_range": c.env.bomb_range, "enable_bombs": c.env.enable_bombs},
            "curriculum": {"enabled": c.curriculum.enabled},
            "train": {"enable_dqn": c.train.enable_dqn},
        }

    def update_config(self, patch: dict) -> dict:
        """Apply a hyperparameter patch and REBUILD the trainer (resets training)."""
        was_running = self._running.is_set()
        self._running.clear()
        with self._lock:
            for section, vals in (patch or {}).items():
                target = getattr(self.cfg, section, None)
                if target is None or not isinstance(vals, dict):
                    continue
                for k, v in vals.items():
                    if hasattr(target, k):
                        # cast to the existing field's type to stay consistent
                        cur = getattr(target, k)
                        try:
                            setattr(target, k, type(cur)(v))
                        except Exception:
                            setattr(target, k, v)
            self.trainer = SoloTrainer(self.cfg)
        if was_running:
            self._running.set()
        return self.get_config()

    def shutdown(self) -> None:
        self._stop.set()
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
