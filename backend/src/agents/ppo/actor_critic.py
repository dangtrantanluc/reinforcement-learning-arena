"""Actor-Critic network for PPO.

Shared MLP trunk → actor head (action logits) + critic head (state value).
Orthogonal init with a small policy-head gain keeps the initial policy near-uniform.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _ortho(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(
        self, obs_dim: int, n_actions: int, hidden_sizes: Tuple[int, int] = (128, 128)
    ) -> None:
        super().__init__()
        h1, h2 = hidden_sizes
        self.trunk = nn.Sequential(
            _ortho(nn.Linear(obs_dim, h1)),
            nn.Tanh(),
            _ortho(nn.Linear(h1, h2)),
            nn.Tanh(),
        )
        self.actor = _ortho(nn.Linear(h2, n_actions), gain=0.01)
        self.critic = _ortho(nn.Linear(h2, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(obs)
        return self.actor(z), self.critic(z).squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.trunk(obs)).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
