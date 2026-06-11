"""Map generation for the competitive Bomberman grid.

Produces a structured, good-looking layout each reset by picking one of several
hand-designed *templates* (pillars / rooms / cross / ring / arena) and then
sprinkling rewards, dangers and destructible boxes. Templates are wall-symmetric
so no agent gets a structurally unfair map. A connectivity pass guarantees every
agent can still reach the goal.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# Cell type codes (shared with the env + frontend).
EMPTY, WALL, REWARD, DANGER, GOAL = 0, 1, 2, 3, 4
PPO_AGENT, DYNAQ_AGENT = 5, 6
BOX = 7
# Overlay-only codes for grid_with_agents() (never stored in the base grid):
BOMB, EXPLOSION, DQN_AGENT = 8, 9, 10

TEMPLATES = ("pillars", "rooms", "cross", "ring", "arena", "diagonal")


# ── Template wall layouts (return a list of wall cells for an n×n grid) ──
def _pillars(n: int) -> List[Tuple[int, int]]:
    """Classic Bomberman: a regular lattice of pillars on even rows/cols."""
    return [(r, c) for r in range(2, n - 1, 2) for c in range(2, n - 1, 2)]


def _rooms(n: int) -> List[Tuple[int, int]]:
    """Four rooms divided by a plus-shaped corridor with doorways."""
    mid = n // 2
    walls = []
    for i in range(n):
        if i not in (1, mid, n - 2):       # leave doorways
            walls.append((mid, i))
            walls.append((i, mid))
    return walls


def _cross(n: int) -> List[Tuple[int, int]]:
    """Two diagonal barriers meeting in the middle, with gaps."""
    walls = []
    for i in range(2, n - 2):
        if i % 3 != 0:
            walls.append((i, i))
            walls.append((i, n - 1 - i))
    return walls


def _ring(n: int) -> List[Tuple[int, int]]:
    """A broken square ring around the centre."""
    lo, hi = 2, n - 3
    walls = []
    for c in range(lo, hi + 1):
        if c % 2 == 0:
            walls.append((lo, c)); walls.append((hi, c))
    for r in range(lo, hi + 1):
        if r % 2 == 0:
            walls.append((r, lo)); walls.append((r, hi))
    return walls


def _arena(n: int) -> List[Tuple[int, int]]:
    """Open arena with a few central pillars — fast, lots of movement."""
    mid = n // 2
    return [(mid, mid), (mid - 1, mid - 1), (mid + 1, mid + 1),
            (mid - 1, mid + 1), (mid + 1, mid - 1)]


def _diagonal(n: int) -> List[Tuple[int, int]]:
    """Staggered diagonal walls forming zig-zag lanes."""
    walls = []
    for r in range(1, n - 1):
        c = (r * 2) % (n - 2) + 1
        if (r, c) not in walls:
            walls.append((r, c))
            if c + 1 < n - 1:
                walls.append((r, c + 1))
    return walls


_TEMPLATE_FN = {
    "pillars": _pillars, "rooms": _rooms, "cross": _cross,
    "ring": _ring, "arena": _arena, "diagonal": _diagonal,
}


def _neighbors(p: Tuple[int, int], n: int):
    r, c = p
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            yield (nr, nc)


def _reachable(grid: np.ndarray, start: Tuple[int, int]) -> set:
    """BFS over non-wall, non-box cells from `start`."""
    n = grid.shape[0]
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in _neighbors(cur, n):
            if nb not in seen and grid[nb] not in (WALL, BOX):
                seen.add(nb)
                stack.append(nb)
    return seen


def generate_map(
    grid_size: int,
    n_walls: int,
    n_rewards: int,
    n_dangers: int,
    randomize: bool,
    rng: np.random.Generator,
    n_boxes: int = 0,
    agent_ids: Tuple[str, ...] = ("ppo", "dynaq"),
) -> Tuple[np.ndarray, dict, Tuple[int, int]]:
    """Build a structured grid plus spawn positions and the goal.

    Returns: (grid, spawns_dict, goal). Spawns: PPO top-left, Dyna-Q bottom-left,
    DQN top-right; goal at the centre for fairness with 3 corners.
    """
    n = grid_size
    grid = np.zeros((n, n), dtype=np.int8)

    corner = {"ppo": (0, 0), "dynaq": (n - 1, 0), "dqn": (0, n - 1)}
    spawns = {a: corner.get(a, (0, 0)) for a in agent_ids}
    # Goal at centre → roughly equidistant from all spawn corners.
    goal = (n // 2, n // 2)

    reserved = set(spawns.values()) | {goal}
    # Keep spawn pockets + goal surroundings clear.
    protected = set(reserved)
    for sp in list(reserved):
        protected.update(_neighbors(sp, n))

    # 1) Lay down a template's walls (skip protected cells).
    template = _TEMPLATE_FN[
        TEMPLATES[int(rng.integers(0, len(TEMPLATES)))] if randomize else 0
    ] if randomize else _pillars
    for w in template(n):
        if w not in protected:
            grid[w] = WALL

    # 2) Free-cell pool for sprinkling items.
    free = [(r, c) for r in range(n) for c in range(n)
            if grid[r, c] == EMPTY and (r, c) not in reserved]
    rng.shuffle(free)

    def take(k: int) -> List[Tuple[int, int]]:
        out, i = [], 0
        while len(out) < k and i < len(free):
            p = free[i]; i += 1
            if p not in protected:
                out.append(p)
        for p in out:
            free.remove(p)
        return out

    for p in take(n_boxes):
        grid[p] = BOX
    for p in take(n_dangers):
        grid[p] = DANGER
    for p in take(n_rewards):
        grid[p] = REWARD

    grid[goal] = GOAL

    # 3) Connectivity guarantee: every agent must reach the goal. If not,
    #    carve a straight corridor by clearing walls/boxes along the path.
    for a, sp in spawns.items():
        if goal not in _reachable(grid, sp):
            r, c = sp
            gr, gc = goal
            while (r, c) != (gr, gc):
                if grid[r, c] in (WALL, BOX):
                    grid[r, c] = EMPTY
                if r != gr:
                    r += 1 if gr > r else -1
                elif c != gc:
                    c += 1 if gc > c else -1

    # Keep each agent's immediate moves open.
    for sp in spawns.values():
        grid[sp] = EMPTY
        for nb in _neighbors(sp, n):
            if grid[nb] == WALL:
                grid[nb] = EMPTY

    return grid, spawns, goal
