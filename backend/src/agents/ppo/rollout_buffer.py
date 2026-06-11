"""On-policy rollout buffer with GAE-λ advantage estimation."""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: torch.device) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.device = device
        self.ptr = 0

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)

        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, done, log_prob, value) -> None:
        # Grow gracefully if an episode pushes us slightly past capacity.
        if self.ptr >= self.capacity:
            return
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.ptr += 1

    def clear(self) -> None:
        self.ptr = 0

    def __len__(self) -> int:
        return self.ptr

    @property
    def full(self) -> bool:
        return self.ptr >= self.capacity

    def compute_returns_and_advantages(
        self, last_value: float, last_done: bool, gamma: float, gae_lambda: float
    ) -> None:
        n = self.ptr
        adv = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_nonterminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_nonterminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            adv = delta + gamma * gae_lambda * next_nonterminal * adv
            self.advantages[t] = adv
        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def iter_minibatches(self, batch_size: int) -> Iterator[Tuple[torch.Tensor, ...]]:
        n = self.ptr
        idx = np.random.permutation(n)
        to_t = lambda a: torch.as_tensor(a[:n], device=self.device)
        states = to_t(self.states)
        actions = to_t(self.actions)
        log_probs = to_t(self.log_probs)
        advantages = to_t(self.advantages)
        returns = to_t(self.returns)
        values = to_t(self.values)
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            yield states[b], actions[b], log_probs[b], advantages[b], returns[b], values[b]
