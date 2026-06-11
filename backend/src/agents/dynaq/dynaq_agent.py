"""DynaQAgent — tabular Q-learning + model-based planning (Dyna-Q).

The "dynaq" player in the arena. Each real step does a Q-learning update, stores
the transition in a world model, then replays `planning_steps` simulated updates.
Epsilon-greedy exploration with geometric decay per episode.
"""

from __future__ import annotations

import pickle
from typing import Dict, Tuple

import numpy as np

from ...config import DynaQConfig
from ..base_agent import BaseAgent
from .q_table import QTable
from .world_model import WorldModel

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
StateT = Tuple[int, ...]


class DynaQAgent(BaseAgent):
    name = "dynaq"

    def __init__(self, config: DynaQConfig, seed: int = 0):
        self.cfg = config
        self.q = QTable(n_actions=config.n_actions)
        self.model = WorldModel()
        self.epsilon = config.epsilon_start
        self._rng = np.random.default_rng(seed)

    # ── Acting ──────────────────────────────────────────────
    def select_action(self, state: StateT, training: bool = True) -> int:
        """Epsilon-greedy during training; pure greedy (argmax) at eval."""
        eps = self.epsilon if training else 0.0
        if training and self._rng.random() < eps:
            return int(self._rng.integers(0, self.cfg.n_actions))
        # Greedy with random tie-breaking among equal-max actions.
        q = self.q.get(state)
        max_q = q.max()
        best = np.flatnonzero(q == max_q)
        return int(self._rng.choice(best))

    def get_q_values(self, state: StateT) -> Dict[str, float]:
        """Q-value dict for the UI panel."""
        q = self.q.get(state)
        return {ACTION_NAMES[i]: float(q[i]) for i in range(self.cfg.n_actions)}

    # ── Learning ────────────────────────────────────────────
    def _q_learn(self, state: StateT, action: int, reward: float, next_state: StateT, done: bool) -> None:
        """One Q-learning backup."""
        target = reward if done else reward + self.cfg.gamma * self.q.max_value(next_state)
        current = self.q.value(state, action)
        self.q.set(state, action, current + self.cfg.alpha * (target - current))

    def update(self, state: StateT, action: int, reward: float, next_state: StateT, done: bool) -> None:
        """Real-experience update + model store + planning sweep."""
        # 1) Direct RL update from real experience.
        self._q_learn(state, action, reward, next_state, done)
        # 2) Remember the transition.
        self.model.add(state, action, next_state, reward, done)
        # 3) Planning: replay imagined experience from the model.
        self.planning_update()

    def planning_update(self) -> None:
        if len(self.model) == 0:
            return
        for _ in range(self.cfg.planning_steps):
            s, a, ns, r, d = self.model.sample(self._rng)
            self._q_learn(s, a, r, ns, d)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

    @property
    def q_table_size(self) -> int:
        return len(self.q)

    # ── Persistence ─────────────────────────────────────────
    def save(self, path: str) -> None:
        payload = {
            "q": self.q.to_serializable(),
            "model": self.model.to_serializable(),
            "epsilon": self.epsilon,
            "config": self.cfg.__dict__,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.q = QTable.from_serializable(payload["q"])
        self.model = WorldModel.from_serializable(payload["model"])
        self.epsilon = payload.get("epsilon", self.cfg.epsilon_end)
