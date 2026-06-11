"""Solo training: PPO vs Dyna-Q on one shared map.

Exposes:
  • SoloTrainer — drives env + both agents one *step* or one *episode* at a time,
    so the FastAPI runner can advance training inside a background thread while
    publishing live state. It also owns the shared metrics + logs.
  • train() — a plain CLI loop:  python -m src.training.train_solo
"""

from __future__ import annotations

import os
import sys
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Config, default_config
from src.env.competitive_grid_env import CompetitiveGridEnv, ACTION_NAMES
from src.env.state_encoder import ppo_obs_dim
from src.agents.ppo.ppo_agent import PPOAgent
from src.agents.dynaq.dynaq_agent import DynaQAgent
from src.training.metrics import MetricsTracker
from src.training.checkpoint import save_all
from src.utils import set_seed, get_device, ensure_dir


class SoloTrainer:
    """Stateful trainer usable both as a CLI loop and behind the API."""

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or default_config()
        set_seed(self.cfg.train.seed)
        self.device = get_device(self.cfg.train.device)

        self.env = CompetitiveGridEnv(self.cfg.env)
        obs_dim = ppo_obs_dim(self.cfg.env.grid_size)

        self.ppo = PPOAgent(obs_dim, 5, self.cfg.ppo, self.device)
        self.dynaq = DynaQAgent(self.cfg.dynaq, seed=self.cfg.train.seed)
        self.metrics = MetricsTracker(window=100)

        self.total_wins = {"ppo": 0, "dynaq": 0, "draw": 0}
        self.logs: Deque[str] = deque(maxlen=200)

        # Live per-step state (consumed by the API).
        self.obs: Dict[str, np.ndarray] = {}
        self.ep_step = 0
        self.episode = 0
        self.ep_reward = {"ppo": 0.0, "dynaq": 0.0}
        self.last_winner: Optional[str] = None
        self._start_episode()

    # ── Episode lifecycle ───────────────────────────────────
    def _log(self, msg: str) -> None:
        self.logs.append(msg)

    def _start_episode(self) -> None:
        self.obs, _ = self.env.reset()
        self.ep_step = 0
        self.ep_reward = {"ppo": 0.0, "dynaq": 0.0}
        self.episode += 1
        self.last_winner = None
        self._log(f"[Episode {self.episode}] started")

    def step_once(self) -> bool:
        """Advance a single env step for both agents. Returns True if episode ended."""
        ppo_state = self.obs["ppo"]
        dynaq_state = self.env.get_dynaq_state("dynaq")

        ppo_action, ppo_lp, ppo_val = self.ppo.select_action(ppo_state, training=True)
        dynaq_action = self.dynaq.select_action(dynaq_state, training=True)

        next_obs, rewards, dones, infos = self.env.step(
            {"ppo": ppo_action, "dynaq": dynaq_action}
        )
        next_dynaq_state = self.env.get_dynaq_state("dynaq")

        # PPO: store transition; update when buffer is full.
        self.ppo.store_transition(
            ppo_state, ppo_action, rewards["ppo"], dones["ppo"], ppo_lp, ppo_val
        )
        # Dyna-Q: online update + planning.
        self.dynaq.update(
            dynaq_state, dynaq_action, rewards["dynaq"], next_dynaq_state, dones["dynaq"]
        )

        if self.ppo.buffer_size() >= self.cfg.ppo.rollout_steps:
            self.ppo.update(last_state=next_obs["ppo"], last_done=dones["ppo"])

        self.ep_reward["ppo"] += rewards["ppo"]
        self.ep_reward["dynaq"] += rewards["dynaq"]
        self.ep_step += 1

        # Lightweight per-step logging (sampled to avoid flooding).
        self._log(f"[Episode {self.episode}] PPO {ACTION_NAMES[ppo_action]} | "
                  f"Dyna-Q {ACTION_NAMES[dynaq_action]}")
        if abs(rewards["ppo"] - self.cfg.env.r_reward_item) < 1e-6:
            self._log(f"[Reward] PPO collected item +{int(self.cfg.env.r_reward_item)}")
        if abs(rewards["dynaq"] - self.cfg.env.r_reward_item) < 1e-6:
            self._log(f"[Reward] Dyna-Q collected item +{int(self.cfg.env.r_reward_item)}")
        if rewards["ppo"] <= self.cfg.env.r_collision and rewards["dynaq"] <= self.cfg.env.r_collision:
            self._log("[Collision] Both agents blocked")

        self.obs = next_obs

        if dones["__all__"]:
            self._finish_episode(infos)
            return True
        return False

    def _finish_episode(self, infos: dict) -> None:
        winner = infos.get("winner")
        self.last_winner = winner
        key = winner if winner in ("ppo", "dynaq") else "draw"
        self.total_wins[key] += 1

        if winner == "ppo":
            self._log(f"[Winner] PPO ({infos.get('reason')})")
        elif winner == "dynaq":
            self._log(f"[Winner] Dyna-Q ({infos.get('reason')})")
        else:
            self._log("[Result] Draw")

        # Flush any remaining PPO transitions so learning still happens on short eps.
        if self.ppo.buffer_size() > 0:
            self.ppo.update(last_state=self.obs["ppo"], last_done=True)

        self.dynaq.decay_epsilon()
        self.metrics.record_episode(
            ppo_reward=self.ep_reward["ppo"],
            dynaq_reward=self.ep_reward["dynaq"],
            winner=winner,
            length=self.ep_step,
            dynaq_epsilon=self.dynaq.epsilon,
            q_table_size=self.dynaq.q_table_size,
            ppo_losses=self.ppo.last_losses,
        )

    def run_episode(self) -> None:
        """Run a full episode then auto-start the next one."""
        done = False
        while not done:
            done = self.step_once()
        self._start_episode()

    # ── Snapshot for the API ────────────────────────────────
    def live_state(self) -> dict:
        ppo_state = self.obs["ppo"]
        dynaq_state = self.env.get_dynaq_state("dynaq")
        m = self.metrics
        return {
            "episode": self.episode,
            "step": self.ep_step,
            "grid": self.env.grid_with_agents(),
            "winner": self.last_winner,
            "ppo": {
                "position": list(self.env.pos["ppo"]),
                "episode_reward": round(self.ep_reward["ppo"], 2),
                "total_wins": self.total_wins["ppo"],
                "last_action": ACTION_NAMES[self.env.last_action["ppo"]],
                "action_probs": self.ppo.get_action_probs(ppo_state),
                "policy_loss": round(self.ppo.last_losses["policy_loss"], 4),
                "value_loss": round(self.ppo.last_losses["value_loss"], 4),
                "entropy": round(self.ppo.last_losses["entropy"], 4),
            },
            "dynaq": {
                "position": list(self.env.pos["dynaq"]),
                "episode_reward": round(self.ep_reward["dynaq"], 2),
                "total_wins": self.total_wins["dynaq"],
                "last_action": ACTION_NAMES[self.env.last_action["dynaq"]],
                "epsilon": round(self.dynaq.epsilon, 4),
                "q_table_size": self.dynaq.q_table_size,
                "planning_steps": self.cfg.dynaq.planning_steps,
                "q_values": self.dynaq.get_q_values(dynaq_state),
            },
            "metrics": {
                "ppo_win_rate": round(m.win_rate("ppo"), 3),
                "dynaq_win_rate": round(m.win_rate("dynaq"), 3),
                "draw_rate": round(m.win_rate("draw"), 3),
                "ppo_avg_reward": round(m.avg("ppo"), 2),
                "dynaq_avg_reward": round(m.avg("dynaq"), 2),
            },
            "history": m.history[-200:],
            "logs": list(self.logs)[-30:],
        }

    def save(self) -> None:
        save_all(self.ppo, self.dynaq, self.cfg.train.ppo_ckpt, self.cfg.train.dynaq_ckpt)
        ensure_dir(self.cfg.train.log_dir)
        self.metrics.save(self.cfg.train.metrics_file)


def train(cfg: Optional[Config] = None) -> None:
    """CLI training loop."""
    cfg = cfg or default_config()
    trainer = SoloTrainer(cfg)
    print(f"[setup] device={trainer.device} episodes={cfg.train.total_episodes}")

    for ep in range(cfg.train.total_episodes):
        trainer.run_episode()
        if (ep + 1) % 20 == 0:
            m = trainer.metrics
            print(
                f"[ep {ep + 1:>5}/{cfg.train.total_episodes}] "
                f"PPO_win={m.win_rate('ppo')*100:5.1f}% "
                f"DynaQ_win={m.win_rate('dynaq')*100:5.1f}% "
                f"draw={m.win_rate('draw')*100:5.1f}% "
                f"R(ppo)={m.avg('ppo'):7.2f} R(dyn)={m.avg('dynaq'):7.2f} "
                f"len={m.avg_length:4.1f} eps={trainer.dynaq.epsilon:.3f} "
                f"|Q|={trainer.dynaq.q_table_size}"
            )
        if (ep + 1) % cfg.train.save_every == 0:
            trainer.save()

    trainer.save()
    print(f"[done] checkpoints + metrics saved")


if __name__ == "__main__":
    train()
