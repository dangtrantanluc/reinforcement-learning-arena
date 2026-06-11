"""Observation encoders for the agents.

Neural agents (PPO, DQN) → dense flattened one-hot grid, perspective-aware
(self vs opponents), now with Bomberman channels (box / bomb / blast).
Dyna-Q → compact discrete state tuple (positions, clipped deltas, local hazards
including bomb-blast danger and adjacent boxes).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .map_generator import WALL, REWARD, DANGER, GOAL, BOX

# Neural observation channels.
#   empty, wall, reward, danger, goal, box, self, opponent(s), bomb, blast
N_CHANNELS = 10
(CH_EMPTY, CH_WALL, CH_REWARD, CH_DANGER, CH_GOAL, CH_BOX,
 CH_SELF, CH_OPP, CH_BOMB, CH_BLAST) = range(10)


def encode_neural_obs(
    grid: np.ndarray,
    self_pos: Tuple[int, int],
    opp_positions: list,
    bomb_map: Optional[np.ndarray] = None,
    blast_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """10-channel one-hot grid flattened to a float32 vector.

    `bomb_map`: float grid, fuse/maxfuse intensity per bomb cell (0 if none).
    `blast_map`: 1.0 where a blast is happening this step.
    `opp_positions`: list of (r,c) for every OTHER agent (1 or 2 of them).
    """
    n = grid.shape[0]
    chans = np.zeros((N_CHANNELS, n, n), dtype=np.float32)
    chans[CH_EMPTY] = (grid == 0)
    chans[CH_WALL] = (grid == WALL)
    chans[CH_REWARD] = (grid == REWARD)
    chans[CH_DANGER] = (grid == DANGER)
    chans[CH_GOAL] = (grid == GOAL)
    chans[CH_BOX] = (grid == BOX)
    chans[CH_SELF][self_pos] = 1.0
    for op in opp_positions:
        chans[CH_OPP][op] = 1.0
    if bomb_map is not None:
        chans[CH_BOMB] = bomb_map
    if blast_map is not None:
        chans[CH_BLAST] = blast_map
    return chans.reshape(-1)


def neural_obs_dim(grid_size: int) -> int:
    return N_CHANNELS * grid_size * grid_size


# Backwards-compatible aliases (older imports).
encode_ppo_obs = None  # replaced by encode_neural_obs


def ppo_obs_dim(grid_size: int) -> int:
    return neural_obs_dim(grid_size)


def _clip(v: int, lo: int = -5, hi: int = 5) -> int:
    return max(lo, min(hi, v))


def _nearest_delta(grid: np.ndarray, pos: Tuple[int, int], code: int) -> Tuple[int, int]:
    """Manhattan-nearest cell matching `code`: (dr, dc) clipped. (0,0) if none."""
    cells = np.argwhere(grid == code)
    if len(cells) == 0:
        return 0, 0
    dists = np.abs(cells[:, 0] - pos[0]) + np.abs(cells[:, 1] - pos[1])
    nr, nc = cells[int(np.argmin(dists))]
    return _clip(int(nr - pos[0])), _clip(int(nc - pos[1]))


def _cell_is(grid: np.ndarray, r: int, c: int, code: int) -> int:
    """1 if (r,c) is off-grid (walls only) or matches `code`, else 0."""
    n = grid.shape[0]
    if not (0 <= r < n and 0 <= c < n):
        return 1 if code == WALL else 0
    return int(grid[r, c] == code)


def _blast_near(blast_map: Optional[np.ndarray], r: int, c: int) -> int:
    n = blast_map.shape[0] if blast_map is not None else 0
    if blast_map is None or not (0 <= r < n and 0 <= c < n):
        return 0
    return int(blast_map[r, c] > 0)


def encode_dynaq_state(
    grid: np.ndarray,
    self_pos: Tuple[int, int],
    opp_pos: Tuple[int, int],
    goal: Tuple[int, int],
    danger_map: Optional[np.ndarray] = None,
) -> Tuple[int, ...]:
    """Discrete 20-tuple state for the Q-table.

    Adds box-awareness and bomb-blast danger to the original 16-tuple:
    (ax, ay, ox, oy, gdx, gdy, rdx, rdy,
     hazard_u, hazard_d, hazard_l, hazard_r,   # danger zone OR live blast
     wall_u, wall_d, wall_l, wall_r,
     box_u, box_d, box_l, box_r)

    `danger_map`: optional float grid marking cells that will be hit by a blast
    soon (so the tabular agent can learn to step away from bombs).
    """
    ax, ay = self_pos
    ox, oy = opp_pos
    gdx = _clip(goal[0] - ax)
    gdy = _clip(goal[1] - ay)
    rdx, rdy = _nearest_delta(grid, self_pos, REWARD)

    def hazard(r: int, c: int) -> int:
        return max(_cell_is(grid, r, c, DANGER), _blast_near(danger_map, r, c))

    return (
        ax, ay, ox, oy, gdx, gdy, rdx, rdy,
        hazard(ax - 1, ay), hazard(ax + 1, ay), hazard(ax, ay - 1), hazard(ax, ay + 1),
        _cell_is(grid, ax - 1, ay, WALL), _cell_is(grid, ax + 1, ay, WALL),
        _cell_is(grid, ax, ay - 1, WALL), _cell_is(grid, ax, ay + 1, WALL),
        _cell_is(grid, ax - 1, ay, BOX), _cell_is(grid, ax + 1, ay, BOX),
        _cell_is(grid, ax, ay - 1, BOX), _cell_is(grid, ax, ay + 1, BOX),
    )
