"""CompetitiveGridEnv — N agents racing on ONE shared Bomberman-like map.

Supports 2 agents (PPO vs Dyna-Q) or 3 (… + DQN). Each step every agent submits
an action; the env resolves movement (with N-way collision rules), advances bombs,
applies rewards, and decides win/lose/draw.

Bomberman mechanics (B5):
    • Action 5 = PLACE_BOMB. A bomb has a fuse; on 0 it explodes in a cross
      (range R along the 4 axes, blocked by walls).
    • A blast destroys boxes (+r_break_box to the bomber) and KILLS any agent
      caught in it (-r_death; the survivor wins via r_win_death).

Observations are agent-specific:
    • Neural (PPO/DQN) → dense one-hot grid (encode_neural_obs), 10 channels.
    • Dyna-Q → discrete tuple (encode_dynaq_state) with bomb/box awareness.

`to_json()` / `grid_with_agents()` serialise state for the realtime API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is required. `pip install -r requirements.txt`."
    ) from exc

from ..config import EnvConfig
from . import reward as R
from .map_generator import (
    generate_map,
    EMPTY, WALL, REWARD, DANGER, GOAL, BOX,
    PPO_AGENT, DYNAQ_AGENT, DQN_AGENT, BOMB, EXPLOSION,
)
from .state_encoder import (
    encode_neural_obs,
    encode_dynaq_state,
    neural_obs_dim,
)

# Actions
UP, DOWN, LEFT, RIGHT, STAY, PLACE_BOMB = 0, 1, 2, 3, 4, 5
_DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1), STAY: (0, 0)}
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY", "BOMB"]

# Per-agent grid overlay code (for grid_with_agents / rendering).
AGENT_CODE = {"ppo": PPO_AGENT, "dynaq": DYNAQ_AGENT, "dqn": DQN_AGENT}


class CompetitiveGridEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
        agent_ids: Tuple[str, ...] = ("ppo", "dynaq"),
    ):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.n = self.cfg.grid_size
        self.agent_ids = tuple(agent_ids)

        self.n_actions = 6 if self.cfg.enable_bombs else 5
        self.action_space = spaces.Discrete(self.n_actions)
        obs_dim = neural_obs_dim(self.n)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Runtime state.
        self.grid: np.ndarray = np.zeros((self.n, self.n), dtype=np.int8)
        self.pos: Dict[str, Tuple[int, int]] = {}
        self.goal: Tuple[int, int] = (self.n - 1, self.n - 1)
        self.bombs: List[Dict[str, Any]] = []   # {row,col,fuse,owner}
        self.blast: np.ndarray = np.zeros((self.n, self.n), dtype=np.float32)
        self.steps = 0
        self.alive: Dict[str, bool] = {}
        self.ep_reward: Dict[str, float] = {}
        self.collected: Dict[str, int] = {}
        self.boxes_broken: Dict[str, int] = {}
        self.danger_hits: Dict[str, int] = {}
        self.last_action: Dict[str, int] = {}
        self.winner: Optional[str] = None
        self.reason: Optional[str] = None
        self._rng = np.random.default_rng(self.cfg.seed)
        self._dist: Dict[str, int] = {}

    # ──────────────────────────────────────────────────────────
    # Gym-style API (multi-agent dict form)
    # ──────────────────────────────────────────────────────────
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.steps = 0
        self.bombs = []
        self.blast = np.zeros((self.n, self.n), dtype=np.float32)
        self.winner = None
        self.reason = None
        self.alive = {a: True for a in self.agent_ids}
        self.ep_reward = {a: 0.0 for a in self.agent_ids}
        self.collected = {a: 0 for a in self.agent_ids}
        self.boxes_broken = {a: 0 for a in self.agent_ids}
        self.danger_hits = {a: 0 for a in self.agent_ids}
        self.last_action = {a: STAY for a in self.agent_ids}

        grid, spawns, goal = generate_map(
            self.n, self.cfg.n_walls, self.cfg.n_rewards, self.cfg.n_dangers,
            self.cfg.randomize, self._rng,
            n_boxes=self.cfg.n_boxes, agent_ids=self.agent_ids,
        )
        self.grid = grid
        self.goal = goal
        self.pos = dict(spawns)
        self._dist = {a: self._manhattan(self.pos[a], goal) for a in self.agent_ids}

        return self._obs_dict(), self._info(rewards=None, danger_now={})

    def step(self, actions: Dict[str, int]):
        """Advance one joint step. `actions` = {agent_id: int}."""
        self.steps += 1
        self.blast = np.zeros((self.n, self.n), dtype=np.float32)
        rewards = {a: 0.0 for a in self.agent_ids}
        danger_now = {a: False for a in self.agent_ids}
        bomb_events: List[dict] = []
        prev_dist = dict(self._dist)

        acts = {a: int(actions.get(a, STAY)) for a in self.agent_ids}
        self.last_action = acts

        # ── 1) Bomb placement (before movement) ──
        for a in self.agent_ids:
            if acts[a] == PLACE_BOMB and self.cfg.enable_bombs and self.alive[a]:
                if self._place_bomb(a):
                    bomb_events.append({"type": "place", "agent": a,
                                        "pos": list(self.pos[a])})

        # ── 2) Movement with N-way collision resolution ──
        targets = {a: self._intended_target(a, acts[a]) for a in self.agent_ids}
        blocked = self._resolve_collisions(targets)
        for a in self.agent_ids:
            if not self.alive[a]:
                continue
            if blocked[a]:
                rewards[a] += self.cfg.r_collision
            else:
                r, dnow = self._enter_cell(a, targets[a])
                rewards[a] += r
                danger_now[a] = dnow

        # ── 3) Advance bombs → explosions ──
        rewards, bomb_events = self._tick_bombs(rewards, bomb_events)

        # ── 4) Per-step shaping (alive agents only) ──
        for a in self.agent_ids:
            if not self.alive[a]:
                continue
            rewards[a] += R.step_cost(self.cfg)
            new_d = self._manhattan(self.pos[a], self.goal)
            rewards[a] += R.distance_shaping(self.cfg, prev_dist[a], new_d)
            self._dist[a] = new_d

        # ── 5) Terminal resolution ──
        terminated = self._resolve_terminal(rewards)
        truncated = False
        if not terminated and self.steps >= self.cfg.max_steps:
            truncated = True
            self._resolve_timeout(rewards)

        done_all = terminated or truncated
        dones = {a: done_all for a in self.agent_ids}
        dones["__all__"] = done_all

        for a in self.agent_ids:
            self.ep_reward[a] += rewards[a]

        return (self._obs_dict(), rewards, dones,
                self._info(rewards, danger_now, bomb_events))

    # ──────────────────────────────────────────────────────────
    # Movement + collisions
    # ──────────────────────────────────────────────────────────
    def _intended_target(self, agent: str, action: int) -> Tuple[int, int]:
        if not self.alive[agent]:
            return self.pos[agent]
        dr, dc = _DELTA.get(action, (0, 0))
        r, c = self.pos[agent]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.n and 0 <= nc < self.n):
            return (r, c)
        if self.grid[nr, nc] in (WALL, BOX):   # cannot walk through walls/boxes
            return (r, c)
        return (nr, nc)

    def _resolve_collisions(self, targets: Dict[str, Tuple[int, int]]) -> Dict[str, bool]:
        """N-way collision: if two+ agents want the same cell, or two swap, all
        involved stay put and are penalised."""
        blocked = {a: False for a in self.agent_ids}
        alive = [a for a in self.agent_ids if self.alive[a]]

        # Contested target cells (more than one mover wants it, and it's a move).
        from collections import Counter
        moving = {a: targets[a] for a in alive if targets[a] != self.pos[a]}
        counts = Counter(moving.values())
        for a, tgt in list(moving.items()):
            if counts[tgt] > 1:
                blocked[a] = True

        # Pairwise swaps.
        for i, a in enumerate(alive):
            for b in alive[i + 1:]:
                if targets[a] == self.pos[b] and targets[b] == self.pos[a] \
                        and targets[a] != self.pos[a]:
                    blocked[a] = blocked[b] = True

        for a in self.agent_ids:
            if blocked[a]:
                targets[a] = self.pos[a]
        return blocked

    def _enter_cell(self, agent: str, target: Tuple[int, int]) -> Tuple[float, bool]:
        """Commit a move; return (reward, stepped_on_danger)."""
        self.pos[agent] = target
        cell = self.grid[target]
        on_reward = cell == REWARD
        on_danger = cell == DANGER
        if on_reward:
            self.collected[agent] += 1
            self.grid[target] = EMPTY
        if on_danger:
            self.danger_hits[agent] += 1
        return R.cell_reward(self.cfg, on_reward, on_danger), on_danger

    # ──────────────────────────────────────────────────────────
    # Bombs
    # ──────────────────────────────────────────────────────────
    def _place_bomb(self, agent: str) -> bool:
        live = sum(1 for b in self.bombs if b["owner"] == agent)
        if live >= self.cfg.max_bombs_per_agent:
            return False
        r, c = self.pos[agent]
        if any(b["row"] == r and b["col"] == c for b in self.bombs):
            return False
        self.bombs.append({"row": r, "col": c, "fuse": self.cfg.bomb_fuse, "owner": agent})
        return True

    def _tick_bombs(self, rewards: Dict[str, float], events: List[dict]):
        survivors = []
        for b in self.bombs:
            b["fuse"] -= 1
            if b["fuse"] > 0:
                survivors.append(b)
            else:
                rewards = self._explode(b, rewards, events)
        self.bombs = survivors
        return rewards, events

    def _explode(self, bomb: dict, rewards: Dict[str, float], events: List[dict]):
        owner = bomb["owner"]
        center = (bomb["row"], bomb["col"])
        cells = [center]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for dist in range(1, self.cfg.bomb_range + 1):
                r, c = center[0] + dr * dist, center[1] + dc * dist
                if not (0 <= r < self.n and 0 <= c < self.n):
                    break
                if self.grid[r, c] == WALL:
                    break
                cells.append((r, c))
                if self.grid[r, c] == BOX:
                    self.grid[r, c] = EMPTY
                    self.boxes_broken[owner] += 1
                    rewards[owner] = rewards.get(owner, 0.0) + self.cfg.r_break_box
                    events.append({"type": "break_box", "agent": owner, "pos": [r, c]})
                    break   # box absorbs the blast

        for (r, c) in cells:
            self.blast[r, c] = 1.0
        events.append({"type": "explode", "agent": owner, "pos": list(center)})

        # Kill any agent standing in the blast.
        for a in self.agent_ids:
            if self.alive[a] and self.pos[a] in cells:
                self.alive[a] = False
                rewards[a] = rewards.get(a, 0.0) + self.cfg.r_death
                events.append({"type": "kill", "agent": a, "pos": list(self.pos[a])})
        return rewards

    def _danger_map(self) -> np.ndarray:
        """Cells that WILL be hit soon (bombs about to explode) — for Dyna-Q."""
        dm = self.blast.copy()
        for b in self.bombs:
            if b["fuse"] <= 1:  # imminent
                center = (b["row"], b["col"])
                dm[center] = 1.0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    for dist in range(1, self.cfg.bomb_range + 1):
                        r, c = center[0] + dr * dist, center[1] + dc * dist
                        if not (0 <= r < self.n and 0 <= c < self.n) or self.grid[r, c] == WALL:
                            break
                        dm[r, c] = 1.0
                        if self.grid[r, c] == BOX:
                            break
        return dm

    def _bomb_map(self) -> np.ndarray:
        """Per-cell bomb fuse intensity (1 = just placed → ~0 = about to blow)."""
        bm = np.zeros((self.n, self.n), dtype=np.float32)
        for b in self.bombs:
            bm[b["row"], b["col"]] = b["fuse"] / max(1, self.cfg.bomb_fuse)
        return bm

    # ──────────────────────────────────────────────────────────
    # Win / lose / draw resolution
    # ──────────────────────────────────────────────────────────
    def _resolve_terminal(self, rewards: Dict[str, float]) -> bool:
        alive = [a for a in self.agent_ids if self.alive[a]]

        # Death outcomes: if only one agent remains alive → it wins.
        if len(alive) == 1 and len(self.agent_ids) >= 2:
            self._set_winner(alive[0], "death", rewards)
            return True
        if len(alive) == 0:
            self.reason = "death"
            self.winner = "draw"
            return True

        # Goal reached by any alive agent.
        reached = [a for a in alive if self.pos[a] == self.goal]
        if reached:
            if len(reached) == 1:
                self._set_winner(reached[0], "goal", rewards)
            else:
                self._decide_by_reward(rewards)
            return True
        return False

    def _resolve_timeout(self, rewards: Dict[str, float]) -> None:
        self.reason = "timeout"
        self._decide_by_reward(rewards)

    def _decide_by_reward(self, rewards: Dict[str, float]) -> None:
        totals = {a: self.ep_reward[a] + rewards[a] for a in self.agent_ids}
        best = max(totals.values())
        winners = [a for a in self.agent_ids if abs(totals[a] - best) < 1e-6]
        if len(winners) != 1:
            self.winner = "draw"
        else:
            self._apply_win_loss(winners[0], rewards)
        if self.reason is None:
            self.reason = "goal"

    def _set_winner(self, winner: str, reason: str, rewards: Dict[str, float]) -> None:
        self.reason = reason
        if reason == "goal":
            rewards[winner] = rewards.get(winner, 0.0) + self.cfg.r_goal_first
        elif reason == "death":
            rewards[winner] = rewards.get(winner, 0.0) + self.cfg.r_win_death
        for a in self.agent_ids:
            if a != winner:
                rewards[a] = rewards.get(a, 0.0) + self.cfg.r_lose
        self.winner = winner

    def _apply_win_loss(self, winner: str, rewards: Dict[str, float]) -> None:
        for a in self.agent_ids:
            if a != winner:
                rewards[a] = rewards.get(a, 0.0) + self.cfg.r_lose
        self.winner = winner

    # ──────────────────────────────────────────────────────────
    # Observations + info
    # ──────────────────────────────────────────────────────────
    def _obs_dict(self) -> Dict[str, np.ndarray]:
        bomb_map = self._bomb_map()
        out = {}
        for a in self.agent_ids:
            opps = [self.pos[o] for o in self.agent_ids if o != a]
            out[a] = encode_neural_obs(self.grid, self.pos[a], opps, bomb_map, self.blast)
        return out

    def get_dynaq_state(self, agent_id: str = "dynaq") -> Tuple[int, ...]:
        opps = [o for o in self.agent_ids if o != agent_id]
        opp = opps[0] if opps else agent_id
        return encode_dynaq_state(
            self.grid, self.pos[agent_id], self.pos[opp], self.goal, self._danger_map()
        )

    def _info(self, rewards, danger_now=None, bomb_events=None) -> Dict[str, Any]:
        return {
            "winner": self.winner,
            "reason": self.reason,
            "positions": {a: list(self.pos[a]) for a in self.agent_ids},
            "steps": self.steps,
            "collected": dict(self.collected),
            "boxes_broken": dict(self.boxes_broken),
            "danger_hits": dict(self.danger_hits),
            "alive": dict(self.alive),
            "danger_now": danger_now or {},
            "bomb_events": bomb_events or [],
            "is_success": self.reason == "goal",
        }

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ──────────────────────────────────────────────────────────
    # Rendering + frontend bridge
    # ──────────────────────────────────────────────────────────
    def render(self):
        glyphs = {EMPTY: " ·", WALL: " █", REWARD: " ◆", DANGER: " ✸",
                  GOAL: " ⚑", BOX: " ▢"}
        label = {"ppo": " P", "dynaq": " D", "dqn": " Q"}
        bomb_cells = {(b["row"], b["col"]): b["fuse"] for b in self.bombs}
        lines = []
        for r in range(self.n):
            row = ""
            placed = False
            for a in self.agent_ids:
                if self.pos.get(a) == (r, 0):
                    placed = True
            for c in range(self.n):
                here = next((a for a in self.agent_ids if self.pos[a] == (r, c) and self.alive[a]), None)
                if here:
                    row += label.get(here, " ?")
                elif self.blast[r, c] > 0:
                    row += " *"
                elif (r, c) in bomb_cells:
                    row += f" {bomb_cells[(r, c)]}"
                else:
                    row += glyphs.get(int(self.grid[r, c]), " ·")
            lines.append(row)
        header = (f"step={self.steps} winner={self.winner or '-'} reason={self.reason or '-'}")
        text = header + "\n" + "\n".join(lines) + "\n"
        if self.render_mode == "human":
            print(text)
        return text

    def grid_with_agents(self) -> list:
        """Integer grid with bombs, blast and agents overlaid (for the UI)."""
        g = self.grid.astype(int).tolist()
        for b in self.bombs:
            g[b["row"]][b["col"]] = BOMB
        for r in range(self.n):
            for c in range(self.n):
                if self.blast[r, c] > 0 and g[r][c] in (EMPTY, BOMB):
                    g[r][c] = EXPLOSION
        for a in self.agent_ids:
            if self.alive[a]:
                r, c = self.pos[a]
                g[r][c] = AGENT_CODE[a]
        return g

    def to_json(self) -> Dict[str, Any]:
        return {
            "gridSize": self.n,
            "grid": self.grid_with_agents(),
            "goal": list(self.goal),
            "positions": {a: list(self.pos[a]) for a in self.agent_ids},
            "alive": dict(self.alive),
            "bombs": [{"row": b["row"], "col": b["col"], "fuse": b["fuse"]} for b in self.bombs],
            "steps": self.steps,
            "winner": self.winner,
            "reason": self.reason,
        }
