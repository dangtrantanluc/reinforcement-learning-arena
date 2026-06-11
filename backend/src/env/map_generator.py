"""Map generation for the competitive grid.

Produces the integer grid layout (walls / rewards / dangers / goal) plus the two
agent spawn positions. Both agents share ONE map. PPO spawns top-left, Dyna-Q
spawns bottom-left, goal sits at the bottom-right corner.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# Cell type codes (shared with the env + frontend).
EMPTY, WALL, REWARD, DANGER, GOAL = 0, 1, 2, 3, 4
PPO_AGENT, DYNAQ_AGENT = 5, 6


# Fixed hand-authored layout used when randomize=False.
_FIXED_WALLS = [(1, 3), (2, 3), (3, 3), (5, 6), (6, 6), (7, 6), (4, 1), (4, 8)]
_FIXED_REWARDS = [(2, 5), (5, 2), (6, 8), (3, 7), (8, 4)]
_FIXED_DANGERS = [(1, 6), (3, 6), (6, 3), (7, 8), (5, 4)]


def generate_map(
    grid_size: int,
    n_walls: int,
    n_rewards: int,
    n_dangers: int,
    randomize: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Build a grid and the spawn/goal positions.

    Returns: (grid, ppo_spawn, dynaq_spawn, goal).
    """
    n = grid_size
    grid = np.zeros((n, n), dtype=np.int8)

    ppo_spawn = (0, 0)
    dynaq_spawn = (n - 1, 0)
    goal = (n - 1, n - 1)

    reserved = {ppo_spawn, dynaq_spawn, goal}

    if randomize:
        free = [
            (r, c)
            for r in range(n)
            for c in range(n)
            if (r, c) not in reserved
        ]
        rng.shuffle(free)

        def take(k: int) -> List[Tuple[int, int]]:
            picked = free[:k]
            del free[:k]
            return picked

        walls = take(n_walls)
        rewards = take(n_rewards)
        dangers = take(n_dangers)
    else:
        walls, rewards, dangers = _FIXED_WALLS, _FIXED_REWARDS, _FIXED_DANGERS

    grid[goal] = GOAL
    for p in walls:
        grid[p] = WALL
    for p in rewards:
        grid[p] = REWARD
    for p in dangers:
        grid[p] = DANGER

    # Keep both agents' first move open so neither is boxed in at spawn.
    for p in [(0, 1), (1, 0), (n - 1, 1), (n - 2, 0)]:
        if grid[p] == WALL:
            grid[p] = EMPTY

    return grid, ppo_spawn, dynaq_spawn, goal
