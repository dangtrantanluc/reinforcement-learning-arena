"""Central configuration for the PPO vs Dyna-Q solo arena.

All tunables live here as dataclasses so training / evaluation / API stay clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple


# ─────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────
@dataclass
class EnvConfig:
    grid_size: int = 10
    max_steps: int = 120
    n_rewards: int = 5
    n_dangers: int = 5
    n_walls: int = 8
    randomize: bool = True       # random layout each reset (False = fixed map)
    seed: int = 42

    # ── Reward shaping (solo competitive rules) ──
    r_goal_first: float = 50.0   # reach goal before opponent
    r_win_death: float = 20.0    # win because opponent died
    r_reward_item: float = 10.0
    r_danger: float = -10.0
    r_death: float = -30.0
    r_lose: float = -20.0
    r_step: float = -1.0
    r_closer: float = 1.0
    r_farther: float = -1.0
    r_collision: float = -2.0


# ─────────────────────────────────────────────────────────────
# PPO
# ─────────────────────────────────────────────────────────────
@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 1024
    ppo_epochs: int = 10
    batch_size: int = 64
    hidden_sizes: Tuple[int, int] = (128, 128)


# ─────────────────────────────────────────────────────────────
# Dyna-Q
# ─────────────────────────────────────────────────────────────
@dataclass
class DynaQConfig:
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    planning_steps: int = 20
    n_actions: int = 5


# ─────────────────────────────────────────────────────────────
# Training / IO
# ─────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    total_episodes: int = 2000
    log_dir: str = "logs"
    metrics_file: str = "logs/training_metrics.json"
    ppo_ckpt: str = "checkpoints/ppo/ppo_latest.pt"
    dynaq_ckpt: str = "checkpoints/dynaq/dynaq_qtable.pkl"
    save_every: int = 50          # episodes between checkpoints
    device: str = "auto"          # "auto" | "cpu" | "cuda"
    seed: int = 42


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    dynaq: DynaQConfig = field(default_factory=DynaQConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def default_config() -> Config:
    return Config()
