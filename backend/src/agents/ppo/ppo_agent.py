"""PPOAgent — clipped-surrogate PPO from scratch (no stable-baselines3).

Used as the "ppo" player in the competitive arena. Collects on-policy
transitions into a RolloutBuffer and updates once the buffer fills.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from ...config import PPOConfig
from ..base_agent import BaseAgent
from .actor_critic import ActorCritic
from .rollout_buffer import RolloutBuffer

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY", "BOMB"]


class PPOAgent(BaseAgent):
    name = "ppo"

    def __init__(self, obs_dim: int, n_actions: int, config: PPOConfig, device: torch.device):
        self.cfg = config
        self.device = device
        self.n_actions = n_actions
        self.net = ActorCritic(obs_dim, n_actions, config.hidden_sizes).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr, eps=1e-5)
        self.buffer = RolloutBuffer(config.rollout_steps, obs_dim, device)

        # Diagnostics surfaced to the API after each update.
        self.last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    # ── Acting ──────────────────────────────────────────────
    @torch.no_grad()
    def select_action(self, state: np.ndarray, training: bool = True):
        """Return (action, log_prob, value).

        training=False → deterministic argmax (used in evaluation).
        """
        obs = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if not training else dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    @torch.no_grad()
    def get_action_probs(self, state: np.ndarray) -> Dict[str, float]:
        """Action-probability dict for the UI policy panel."""
        obs = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.net(obs)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return {ACTION_NAMES[i]: float(probs[i]) for i in range(self.n_actions)}

    @torch.no_grad()
    def get_value(self, state: np.ndarray) -> float:
        obs = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.net.get_value(obs).item())

    # ── Buffering ───────────────────────────────────────────
    def store_transition(self, state, action, reward, done, log_prob, value) -> None:
        self.buffer.add(state, action, reward, done, log_prob, value)

    def buffer_size(self) -> int:
        return len(self.buffer)

    # ── Learning ────────────────────────────────────────────
    def update(self, last_state: np.ndarray, last_done: bool = False) -> Dict[str, float]:
        """Run PPO epochs over the buffer, then clear it. Returns loss dict."""
        cfg = self.cfg
        last_value = self.get_value(last_state)
        self.buffer.compute_returns_and_advantages(
            last_value, last_done, cfg.gamma, cfg.gae_lambda
        )

        clip = cfg.clip_eps
        pg, vl, ent = [], [], []

        for _ in range(cfg.ppo_epochs):
            for batch in self.buffer.iter_minibatches(cfg.batch_size):
                states, actions, old_lp, adv, returns, old_v = batch
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                _, new_lp, entropy, new_v = self.net.get_action_and_value(states, actions)
                log_ratio = new_lp - old_lp
                ratio = torch.exp(log_ratio)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                v_unclipped = (new_v - returns) ** 2
                v_clipped = old_v + torch.clamp(new_v - old_v, -clip, clip)
                v_clipped = (v_clipped - returns) ** 2
                critic_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                entropy_loss = entropy.mean()
                loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * entropy_loss

                # Skip non-finite updates so divergence can't poison the policy.
                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                pg.append(actor_loss.item())
                vl.append(critic_loss.item())
                ent.append(entropy_loss.item())

        self.buffer.clear()
        safe_mean = lambda xs: float(np.mean(xs)) if xs else 0.0
        self.last_losses = {
            "policy_loss": safe_mean(pg),
            "value_loss": safe_mean(vl),
            "entropy": safe_mean(ent),
        }
        return self.last_losses

    # ── Persistence ─────────────────────────────────────────
    def save(self, path: str) -> None:
        torch.save({"model": self.net.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
