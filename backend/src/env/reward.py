"""Reward computation for the competitive solo rules.

Kept separate from the env so the shaping is easy to read, test and tweak.
The env calls these helpers while resolving a joint step.
"""

from __future__ import annotations

from ..config import EnvConfig


def step_cost(cfg: EnvConfig) -> float:
    """Flat per-step living cost (encourages reaching the goal quickly)."""
    return cfg.r_step


def distance_shaping(cfg: EnvConfig, prev_dist: int, new_dist: int) -> float:
    """+r_closer when nearer to goal, r_farther when further, 0 if unchanged."""
    if new_dist < prev_dist:
        return cfg.r_closer
    if new_dist > prev_dist:
        return cfg.r_farther
    return 0.0


def cell_reward(cfg: EnvConfig, stepped_on_reward: bool, stepped_on_danger: bool) -> float:
    """Reward for the cell an agent just entered."""
    r = 0.0
    if stepped_on_reward:
        r += cfg.r_reward_item
    if stepped_on_danger:
        r += cfg.r_danger
    return r
