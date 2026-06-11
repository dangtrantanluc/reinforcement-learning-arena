"""Observation encoders for both agents.

PPO   → dense flattened one-hot grid (7 channels, perspective-aware: self vs opponent).
Dyna-Q → compact discrete state tuple (positions, clipped deltas, local hazards).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .map_generator import WALL, REWARD, DANGER, GOAL

# PPO observation channels.
N_CHANNELS = 7
CH_EMPTY, CH_WALL, CH_REWARD, CH_DANGER, CH_GOAL, CH_SELF, CH_OPP = range(7)


def encode_ppo_obs(
    grid: np.ndarray,
    self_pos: Tuple[int, int],
    opp_pos: Tuple[int, int],
) -> np.ndarray:
    """7-channel one-hot grid flattened to a float32 vector.

    Channel layout: empty, wall, reward, danger, goal, self, opponent.
    The same encoder serves both agents — pass the agent's own pos as `self_pos`.
    """
    n = grid.shape[0]
    chans = np.zeros((N_CHANNELS, n, n), dtype=np.float32)
    chans[CH_EMPTY] = (grid == 0)
    chans[CH_WALL] = (grid == WALL)
    chans[CH_REWARD] = (grid == REWARD)
    chans[CH_DANGER] = (grid == DANGER)
    chans[CH_GOAL] = (grid == GOAL)
    chans[CH_SELF][self_pos] = 1.0
    chans[CH_OPP][opp_pos] = 1.0
    return chans.reshape(-1)


def ppo_obs_dim(grid_size: int) -> int:
    return N_CHANNELS * grid_size * grid_size


def _clip(v: int, lo: int = -5, hi: int = 5) -> int:
    return max(lo, min(hi, v))


def _nearest_reward_delta(
    grid: np.ndarray, pos: Tuple[int, int]
) -> Tuple[int, int]:
    """Manhattan-nearest reward item's (dr, dc) from `pos`, clipped. (0,0) if none."""
    rewards = np.argwhere(grid == REWARD)
    if len(rewards) == 0:
        return 0, 0
    dists = np.abs(rewards[:, 0] - pos[0]) + np.abs(rewards[:, 1] - pos[1])
    nr, nc = rewards[int(np.argmin(dists))]
    return _clip(int(nr - pos[0])), _clip(int(nc - pos[1]))


def _hazard(grid: np.ndarray, r: int, c: int, code: int) -> int:
    """1 if cell (r,c) is off-grid (for walls) or matches `code`, else 0."""
    n = grid.shape[0]
    if not (0 <= r < n and 0 <= c < n):
        return 1 if code == WALL else 0
    return int(grid[r, c] == code)


def encode_dynaq_state(
    grid: np.ndarray,
    self_pos: Tuple[int, int],
    opp_pos: Tuple[int, int],
    goal: Tuple[int, int],
) -> Tuple[int, ...]:
    """Discrete 16-tuple state for the Q-table.

    (ax, ay, ox, oy, gdx, gdy, rdx, rdy,
     danger_u, danger_d, danger_l, danger_r,
     wall_u, wall_d, wall_l, wall_r)
    """
    ax, ay = self_pos
    ox, oy = opp_pos
    gdx = _clip(goal[0] - ax)
    gdy = _clip(goal[1] - ay)
    rdx, rdy = _nearest_reward_delta(grid, self_pos)

    danger_u = _hazard(grid, ax - 1, ay, DANGER)
    danger_d = _hazard(grid, ax + 1, ay, DANGER)
    danger_l = _hazard(grid, ax, ay - 1, DANGER)
    danger_r = _hazard(grid, ax, ay + 1, DANGER)

    wall_u = _hazard(grid, ax - 1, ay, WALL)
    wall_d = _hazard(grid, ax + 1, ay, WALL)
    wall_l = _hazard(grid, ax, ay - 1, WALL)
    wall_r = _hazard(grid, ax, ay + 1, WALL)

    return (
        ax, ay, ox, oy, gdx, gdy, rdx, rdy,
        danger_u, danger_d, danger_l, danger_r,
        wall_u, wall_d, wall_l, wall_r,
    )
