"""Unit tests for CompetitiveGridEnv core logic (D13).

Run:  venv/bin/python -m pytest tests/ -q     (from backend/)
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import EnvConfig
from src.env.competitive_grid_env import (
    CompetitiveGridEnv, UP, DOWN, LEFT, RIGHT, STAY, PLACE_BOMB, EMPTY,
)
from src.env.map_generator import WALL, BOX, REWARD, DANGER
from src.env.state_encoder import neural_obs_dim, encode_dynaq_state


def make_env(**kw):
    return CompetitiveGridEnv(EnvConfig(seed=7, randomize=False, **kw))


# ── Observation / action contracts ──────────────────────────
def test_obs_shape_and_actions():
    env = make_env()
    obs, _ = env.reset(seed=7)
    assert obs["ppo"].shape == (neural_obs_dim(10),)
    assert env.n_actions == 6  # bombs enabled


def test_dynaq_state_length():
    env = make_env()
    env.reset(seed=7)
    assert len(env.get_dynaq_state("dynaq")) == 20


# ── Movement ────────────────────────────────────────────────
def test_cannot_walk_through_wall():
    env = make_env()
    env.reset(seed=7)
    env.pos["ppo"] = (5, 5)
    env.grid[5, 6] = WALL
    env._dist["ppo"] = env._manhattan((5, 5), env.goal)
    tgt = env._intended_target("ppo", RIGHT)
    assert tgt == (5, 5), "wall must block movement"


def test_cannot_walk_through_box():
    env = make_env()
    env.reset(seed=7)
    env.pos["dynaq"] = (4, 4)
    env.grid[4, 5] = BOX
    assert env._intended_target("dynaq", RIGHT) == (4, 4)


def test_reward_pickup_consumes_item():
    env = make_env()
    env.reset(seed=7)
    env.pos["ppo"] = (3, 3)
    env.grid[3, 4] = REWARD
    r, _ = env._enter_cell("ppo", (3, 4))
    assert r >= env.cfg.r_reward_item - 1e-6
    assert env.grid[3, 4] == EMPTY  # consumed
    assert env.collected["ppo"] == 1


# ── Collisions ──────────────────────────────────────────────
def test_same_cell_collision_blocks_both():
    env = make_env()
    env.reset(seed=7)
    # Use row 1 (away from the centre goal) to contest cell (1,6).
    env.pos["ppo"] = (1, 5)
    env.pos["dynaq"] = (1, 7)
    for c in (5, 6, 7):
        env.grid[1, c] = EMPTY
    env._dist = {a: env._manhattan(env.pos[a], env.goal) for a in env.agent_ids}
    _, rew, _, _ = env.step({"ppo": RIGHT, "dynaq": LEFT, "dqn": STAY})
    assert env.pos["ppo"] == (1, 5) and env.pos["dynaq"] == (1, 7)
    assert rew["ppo"] <= env.cfg.r_collision and rew["dynaq"] <= env.cfg.r_collision


def test_swap_collision_blocks_both():
    env = make_env()
    env.reset(seed=7)
    env.pos["ppo"] = (2, 2)
    env.pos["dynaq"] = (2, 3)
    env.grid[2, 2] = env.grid[2, 3] = EMPTY
    env._dist = {a: env._manhattan(env.pos[a], env.goal) for a in env.agent_ids}
    env.step({"ppo": RIGHT, "dynaq": LEFT, "dqn": STAY})
    assert env.pos["ppo"] == (2, 2) and env.pos["dynaq"] == (2, 3)


# ── Win conditions ──────────────────────────────────────────
def test_reaching_goal_wins():
    env = make_env()
    env.reset(seed=7)
    # Place PPO one cell to the LEFT of the goal, clear the approach.
    gr, gc = env.goal
    env.pos["ppo"] = (gr, gc - 1)
    env.grid[gr, gc - 1] = EMPTY
    env._dist = {a: env._manhattan(env.pos[a], env.goal) for a in env.agent_ids}
    _, rew, dones, info = env.step({"ppo": RIGHT, "dynaq": STAY, "dqn": STAY})
    assert dones["__all__"]
    assert info["winner"] == "ppo"
    assert info["reason"] == "goal"
    assert rew["ppo"] >= env.cfg.r_goal_first - 5  # +goal (minus small step costs)


# ── Bomberman ───────────────────────────────────────────────
def test_bomb_explodes_and_breaks_box():
    env = make_env(bomb_fuse=1, bomb_range=2)
    env.reset(seed=7)
    env.pos["ppo"] = (5, 5)
    # clear a line and put a box in blast range
    for c in range(5, 8):
        env.grid[5, c] = EMPTY
    env.grid[5, 6] = BOX
    env._dist = {a: env._manhattan(env.pos[a], env.goal) for a in env.agent_ids}
    # place bomb (fuse=1) then step again to detonate
    env.step({"ppo": PLACE_BOMB, "dynaq": STAY, "dqn": STAY})
    _, rew, _, info = env.step({"ppo": STAY, "dynaq": STAY, "dqn": STAY})
    assert env.boxes_broken["ppo"] >= 1
    assert env.grid[5, 6] == EMPTY


def test_bomb_kills_agent_standing_in_blast():
    env = make_env(bomb_fuse=1, bomb_range=2, max_bombs_per_agent=1)
    env.reset(seed=7)
    env.pos["dynaq"] = (3, 3)
    env.pos["ppo"] = (3, 4)   # adjacent → in blast
    for c in range(2, 6):
        env.grid[3, c] = EMPTY
    env._dist = {a: env._manhattan(env.pos[a], env.goal) for a in env.agent_ids}
    env.step({"dynaq": PLACE_BOMB, "ppo": STAY, "dqn": STAY})
    _, rew, dones, info = env.step({"dynaq": STAY, "ppo": STAY, "dqn": STAY})
    # ppo should have been caught and killed → episode resolves
    assert not env.alive["ppo"] or info["winner"] is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
