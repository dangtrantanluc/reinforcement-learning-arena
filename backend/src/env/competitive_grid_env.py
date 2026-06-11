"""CompetitiveGridEnv — two agents (PPO + Dyna-Q) racing on ONE shared map.

Multi-agent grid-world. Each step both agents submit an action; the env resolves
movement (with collision rules), applies rewards, and decides win/lose/draw.

Observations are agent-specific:
    • PPO  → dense one-hot grid (via state_encoder.encode_ppo_obs)
    • Dyna-Q → discrete tuple (via state_encoder.encode_dynaq_state)

The env exposes both: PPO obs is returned from reset/step, and the trainer pulls
the Dyna-Q tuple on demand via `encode_dynaq_state(env, "dynaq")`. The env also
provides `to_json()` for the realtime API / frontend.

Designed to extend toward Bomberman (bombs/boxes/explosions) later: movement,
hazards and rendering are already cell-code driven.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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
    EMPTY, WALL, REWARD, DANGER, GOAL, PPO_AGENT, DYNAQ_AGENT,
)
from .state_encoder import (
    encode_ppo_obs,
    encode_dynaq_state,
    ppo_obs_dim,
)

# Actions
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
_DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1), STAY: (0, 0)}
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

AGENTS = ("ppo", "dynaq")


class CompetitiveGridEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.n = self.cfg.grid_size

        self.action_space = spaces.Discrete(5)
        obs_dim = ppo_obs_dim(self.n)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Runtime state.
        self.grid: np.ndarray = np.zeros((self.n, self.n), dtype=np.int8)
        self.pos: Dict[str, Tuple[int, int]] = {}
        self.goal: Tuple[int, int] = (self.n - 1, self.n - 1)
        self.steps = 0
        self.alive: Dict[str, bool] = {"ppo": True, "dynaq": True}
        self.ep_reward: Dict[str, float] = {"ppo": 0.0, "dynaq": 0.0}
        self.collected: Dict[str, int] = {"ppo": 0, "dynaq": 0}
        self.danger_hits: Dict[str, int] = {"ppo": 0, "dynaq": 0}
        self.last_action: Dict[str, int] = {"ppo": STAY, "dynaq": STAY}
        self.winner: Optional[str] = None
        self.reason: Optional[str] = None
        self._rng = np.random.default_rng(self.cfg.seed)

    # ──────────────────────────────────────────────────────────
    # Gym-style API (multi-agent dict form)
    # ──────────────────────────────────────────────────────────
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.steps = 0
        self.alive = {"ppo": True, "dynaq": True}
        self.ep_reward = {"ppo": 0.0, "dynaq": 0.0}
        self.collected = {"ppo": 0, "dynaq": 0}
        self.danger_hits = {"ppo": 0, "dynaq": 0}
        self.last_action = {"ppo": STAY, "dynaq": STAY}
        self.winner = None
        self.reason = None

        grid, ppo_spawn, dynaq_spawn, goal = generate_map(
            self.n,
            self.cfg.n_walls,
            self.cfg.n_rewards,
            self.cfg.n_dangers,
            self.cfg.randomize,
            self._rng,
        )
        self.grid = grid
        self.goal = goal
        self.pos = {"ppo": ppo_spawn, "dynaq": dynaq_spawn}
        self._dist = {a: self._manhattan(self.pos[a], goal) for a in AGENTS}

        return self._obs_dict(), self._info(rewards=None)

    def step(self, actions: Dict[str, int]):
        """Advance one joint step.

        `actions` = {"ppo": int, "dynaq": int}.
        Returns next_obs, rewards, dones, infos (all dicts).
        """
        self.steps += 1
        a_ppo = int(actions.get("ppo", STAY))
        a_dyn = int(actions.get("dynaq", STAY))
        self.last_action = {"ppo": a_ppo, "dynaq": a_dyn}

        rewards = {"ppo": 0.0, "dynaq": 0.0}
        prev_dist = dict(self._dist)

        # ── 1) Resolve intended targets with collision rules ──
        targets = {
            "ppo": self._intended_target("ppo", a_ppo),
            "dynaq": self._intended_target("dynaq", a_dyn),
        }
        collided = self._resolve_collisions(targets)
        for a in AGENTS:
            if collided[a]:
                rewards[a] += self.cfg.r_collision  # -2 each on blocked move
            else:
                # Commit movement + cell effects.
                rewards[a] += self._enter_cell(a, targets[a])

        # ── 2) Per-step shaping ──
        for a in AGENTS:
            rewards[a] += R.step_cost(self.cfg)
            new_d = self._manhattan(self.pos[a], self.goal)
            rewards[a] += R.distance_shaping(self.cfg, prev_dist[a], new_d)
            self._dist[a] = new_d

        # ── 3) Terminal resolution (goal / death / timeout) ──
        terminated = self._resolve_terminal(rewards)
        truncated = False
        if not terminated and self.steps >= self.cfg.max_steps:
            truncated = True
            self._resolve_timeout(rewards)

        done_all = terminated or truncated
        dones = {
            "ppo": done_all,
            "dynaq": done_all,
            "__all__": done_all,
        }

        for a in AGENTS:
            self.ep_reward[a] += rewards[a]

        return self._obs_dict(), rewards, dones, self._info(rewards)

    # ──────────────────────────────────────────────────────────
    # Movement + collisions
    # ──────────────────────────────────────────────────────────
    def _intended_target(self, agent: str, action: int) -> Tuple[int, int]:
        """Where the agent WANTS to go (before collision arbitration)."""
        dr, dc = _DELTA.get(action, (0, 0))
        r, c = self.pos[agent]
        nr, nc = r + dr, c + dc
        # Out-of-bounds or wall → stay put.
        if not (0 <= nr < self.n and 0 <= nc < self.n):
            return (r, c)
        if self.grid[nr, nc] == WALL:
            return (r, c)
        return (nr, nc)

    def _resolve_collisions(self, targets: Dict[str, Tuple[int, int]]) -> Dict[str, bool]:
        """Apply the two collision rules. Returns which agents were blocked.

        Rule 1: both target the same cell → both stay, both penalised.
        Rule 2: agents swap cells → both stay, both penalised.
        """
        p_from, d_from = self.pos["ppo"], self.pos["dynaq"]
        p_to, d_to = targets["ppo"], targets["dynaq"]
        blocked = {"ppo": False, "dynaq": False}

        same_cell = p_to == d_to and p_to != p_from  # genuinely contested cell
        # swap: each moves into the other's previous cell
        swap = p_to == d_from and d_to == p_from and p_to != p_from

        if same_cell or swap:
            targets["ppo"] = p_from
            targets["dynaq"] = d_from
            blocked = {"ppo": True, "dynaq": True}

        return blocked

    def _enter_cell(self, agent: str, target: Tuple[int, int]) -> float:
        """Commit a (already collision-checked) move and apply cell effects."""
        self.pos[agent] = target
        cell = self.grid[target]
        on_reward = cell == REWARD
        on_danger = cell == DANGER
        if on_reward:
            self.collected[agent] += 1
            self.grid[target] = EMPTY  # consume the item
        if on_danger:
            self.danger_hits[agent] += 1
        return R.cell_reward(self.cfg, on_reward, on_danger)

    # ──────────────────────────────────────────────────────────
    # Win / lose / draw resolution
    # ──────────────────────────────────────────────────────────
    def _resolve_terminal(self, rewards: Dict[str, float]) -> bool:
        """Goal-reached or death ends the episode. Mutates rewards + winner."""
        ppo_goal = self.pos["ppo"] == self.goal
        dyn_goal = self.pos["dynaq"] == self.goal

        # Death = standing on danger this step would be too punishing; instead a
        # "death" is when an agent's danger hit drops it (we model danger as
        # lethal only when it lands exactly on a DANGER-tagged terminal cell).
        # Here danger is non-lethal shaping; death is reserved for future bombs.
        # Goal resolution:
        if ppo_goal or dyn_goal:
            if ppo_goal and dyn_goal:
                # Both reached simultaneously → compare reward, else draw.
                self._decide_by_reward(rewards)
            elif ppo_goal:
                self._set_winner("ppo", "goal", rewards)
            else:
                self._set_winner("dynaq", "goal", rewards)
            return True

        return False

    def _resolve_timeout(self, rewards: Dict[str, float]) -> None:
        self.reason = "timeout"
        self._decide_by_reward(rewards)

    def _decide_by_reward(self, rewards: Dict[str, float]) -> None:
        """Tie-break by total episode reward (incl. this step's deltas so far)."""
        p_total = self.ep_reward["ppo"] + rewards["ppo"]
        d_total = self.ep_reward["dynaq"] + rewards["dynaq"]
        if abs(p_total - d_total) < 1e-6:
            self.winner = "draw"
        elif p_total > d_total:
            self._apply_win_loss("ppo", rewards)
        else:
            self._apply_win_loss("dynaq", rewards)
        if self.reason is None:
            self.reason = "goal"

    def _set_winner(self, winner: str, reason: str, rewards: Dict[str, float]) -> None:
        self.reason = reason
        if reason == "goal":
            rewards[winner] += self.cfg.r_goal_first
        elif reason == "death":
            rewards[winner] += self.cfg.r_win_death
        loser = "dynaq" if winner == "ppo" else "ppo"
        rewards[loser] += self.cfg.r_lose
        self.winner = winner

    def _apply_win_loss(self, winner: str, rewards: Dict[str, float]) -> None:
        loser = "dynaq" if winner == "ppo" else "ppo"
        rewards[loser] += self.cfg.r_lose
        self.winner = winner

    # ──────────────────────────────────────────────────────────
    # Observations + info
    # ──────────────────────────────────────────────────────────
    def _obs_dict(self) -> Dict[str, np.ndarray]:
        return {
            "ppo": encode_ppo_obs(self.grid, self.pos["ppo"], self.pos["dynaq"]),
            "dynaq": encode_ppo_obs(self.grid, self.pos["dynaq"], self.pos["ppo"]),
        }

    def get_dynaq_state(self, agent_id: str = "dynaq") -> Tuple[int, ...]:
        """Discrete state tuple for the Q-table (perspective of `agent_id`)."""
        opp = "ppo" if agent_id == "dynaq" else "dynaq"
        return encode_dynaq_state(
            self.grid, self.pos[agent_id], self.pos[opp], self.goal
        )

    def _info(self, rewards: Optional[Dict[str, float]]) -> Dict[str, Any]:
        return {
            "winner": self.winner,
            "reason": self.reason,
            "positions": {
                "ppo": list(self.pos["ppo"]),
                "dynaq": list(self.pos["dynaq"]),
            },
            "steps": self.steps,
            "collected": dict(self.collected),
            "danger_hits": dict(self.danger_hits),
        }

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ──────────────────────────────────────────────────────────
    # Rendering + frontend bridge
    # ──────────────────────────────────────────────────────────
    def render(self):
        glyphs = {EMPTY: " ·", WALL: " █", REWARD: " ◆", DANGER: " ✸", GOAL: " ⚑"}
        lines = []
        for r in range(self.n):
            row = ""
            for c in range(self.n):
                if (r, c) == self.pos["ppo"]:
                    row += " P"
                elif (r, c) == self.pos["dynaq"]:
                    row += " D"
                else:
                    row += glyphs[int(self.grid[r, c])]
            lines.append(row)
        header = (
            f"step={self.steps} winner={self.winner or '-'} reason={self.reason or '-'} "
            f"R(ppo)={self.ep_reward['ppo']:.1f} R(dyn)={self.ep_reward['dynaq']:.1f}"
        )
        text = header + "\n" + "\n".join(lines) + "\n"
        if self.render_mode == "human":
            print(text)
        return text

    def grid_with_agents(self) -> list:
        """Integer grid with agent codes overlaid (for the API `grid` field)."""
        g = self.grid.astype(int).tolist()
        pr, pc = self.pos["ppo"]
        dr, dc = self.pos["dynaq"]
        g[pr][pc] = PPO_AGENT
        g[dr][dc] = DYNAQ_AGENT
        return g

    def to_json(self) -> Dict[str, Any]:
        """Full state snapshot for the realtime API / frontend."""
        return {
            "gridSize": self.n,
            "grid": self.grid_with_agents(),
            "goal": list(self.goal),
            "positions": {
                "ppo": list(self.pos["ppo"]),
                "dynaq": list(self.pos["dynaq"]),
            },
            "steps": self.steps,
            "winner": self.winner,
            "reason": self.reason,
        }
