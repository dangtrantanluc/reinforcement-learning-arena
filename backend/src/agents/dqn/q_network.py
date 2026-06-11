"""Q-network for DQN — an MLP mapping observation → Q(s, a) for each action."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def _ortho(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Tuple[int, int] = (128, 128)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            _ortho(nn.Linear(obs_dim, h1)),
            nn.ReLU(),
            _ortho(nn.Linear(h1, h2)),
            nn.ReLU(),
            _ortho(nn.Linear(h2, n_actions), gain=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
