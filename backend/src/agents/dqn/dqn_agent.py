"""DQNAgent — Deep Q-Network with target network + experience replay.

The third arena player (B6). Off-policy and neural, it contrasts with PPO
(on-policy neural) and Dyna-Q (tabular). Double-DQN target to reduce overestimation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import DQNConfig
from ..base_agent import BaseAgent
from .q_network import QNetwork
from .replay_buffer import ReplayBuffer

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY", "BOMB"]


class DQNAgent(BaseAgent):
    name = "dqn"

    def __init__(self, obs_dim: int, n_actions: int, config: DQNConfig, device: torch.device):
        self.cfg = config
        self.device = device
        self.n_actions = n_actions
        self.online = QNetwork(obs_dim, n_actions, config.hidden_sizes).to(device)
        self.target = QNetwork(obs_dim, n_actions, config.hidden_sizes).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, device)
        self.epsilon = config.epsilon_start
        self._learn_steps = 0
        self.last_loss = 0.0
        self._rng = np.random.default_rng(0)

    # ── Acting ──────────────────────────────────────────────
    @torch.no_grad()
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        eps = self.epsilon if training else 0.0
        if training and self._rng.random() < eps:
            return int(self._rng.integers(0, self.n_actions))
        obs = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(obs).argmax(dim=-1).item())

    @torch.no_grad()
    def get_q_values(self, state: np.ndarray) -> Dict[str, float]:
        obs = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.online(obs).squeeze(0).cpu().numpy()
        return {ACTION_NAMES[i]: float(q[i]) for i in range(self.n_actions)}

    # ── Learning ────────────────────────────────────────────
    def store(self, s, a, r, ns, done) -> None:
        self.buffer.add(s, a, r, ns, done)

    def learn(self) -> None:
        """One gradient step (Double-DQN), gated by cadence + warmup."""
        if len(self.buffer) < max(self.cfg.min_buffer, self.cfg.batch_size):
            return
        self._learn_steps += 1
        if self._learn_steps % self.cfg.learn_every != 0:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.cfg.batch_size)
        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: online picks the action, target evaluates it.
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + self.cfg.gamma * next_q * (1.0 - dones)
        loss = F.smooth_l1_loss(q, target)
        # Guard against divergence: a NaN/Inf loss would poison the weights
        # permanently, so skip the step instead of applying a bad gradient.
        if not torch.isfinite(loss):
            return
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        self.last_loss = float(loss.item())

        if self._learn_steps % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

    # ── Persistence ─────────────────────────────────────────
    def save(self, path: str) -> None:
        torch.save({"online": self.online.state_dict(),
                    "target": self.target.state_dict(),
                    "epsilon": self.epsilon}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt.get("target", ckpt["online"]))
        self.epsilon = ckpt.get("epsilon", self.cfg.epsilon_end)
