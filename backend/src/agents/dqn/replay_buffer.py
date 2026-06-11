"""Experience replay buffer for DQN (uniform sampling, ring buffer)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, s, a, r, ns, done) -> None:
        i = self.ptr
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = ns
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        t = lambda a: torch.as_tensor(a[idx], device=self.device)
        return (t(self.states), t(self.actions), t(self.rewards),
                t(self.next_states), t(self.dones))

    def __len__(self) -> int:
        return self.size
