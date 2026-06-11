"""Evaluate trained PPO vs Dyna-Q head-to-head (exploration off).

  python -m src.training.evaluate_solo
  python -m src.training.evaluate_solo --episodes 100 --render

Loads both checkpoints, runs N deterministic episodes, prints a comparison table.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Config, default_config
from src.env.competitive_grid_env import CompetitiveGridEnv
from src.env.state_encoder import ppo_obs_dim
from src.agents.ppo.ppo_agent import PPOAgent
from src.agents.dynaq.dynaq_agent import DynaQAgent
from src.training.checkpoint import load_ppo, load_dynaq
from src.utils import set_seed, get_device


def evaluate(cfg: Config, episodes: int = 100, render: bool = False) -> Dict:
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)

    env = CompetitiveGridEnv(cfg.env, render_mode="human" if render else None)
    obs_dim = ppo_obs_dim(cfg.env.grid_size)

    ppo = PPOAgent(obs_dim, 5, cfg.ppo, device)
    dynaq = DynaQAgent(cfg.dynaq)

    ppo_loaded = load_ppo(ppo, cfg.train.ppo_ckpt)
    dynaq_loaded = load_dynaq(dynaq, cfg.train.dynaq_ckpt)
    print(f"[eval] ppo_loaded={ppo_loaded} dynaq_loaded={dynaq_loaded} on {device}")

    wins = {"ppo": 0, "dynaq": 0, "draw": 0}
    rewards = {"ppo": [], "dynaq": []}
    danger = {"ppo": 0, "dynaq": 0}
    lengths = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.train.seed + 1000 + ep)
        done = False
        while not done:
            ppo_state = obs["ppo"]
            dynaq_state = env.get_dynaq_state("dynaq")
            a_ppo, _, _ = ppo.select_action(ppo_state, training=False)
            a_dyn = dynaq.select_action(dynaq_state, training=False)
            obs, r, dones, infos = env.step({"ppo": a_ppo, "dynaq": a_dyn})
            done = dones["__all__"]
            if render:
                print("\033[H\033[J", end="")
                print(f"Eval episode {ep+1}/{episodes}")
                print(env.render())

        winner = infos.get("winner") or "draw"
        wins[winner if winner in wins else "draw"] += 1
        rewards["ppo"].append(env.ep_reward["ppo"])
        rewards["dynaq"].append(env.ep_reward["dynaq"])
        danger["ppo"] += infos["danger_hits"]["ppo"]
        danger["dynaq"] += infos["danger_hits"]["dynaq"]
        lengths.append(infos["steps"])

    result = {
        "episodes": episodes,
        "ppo_win_rate": wins["ppo"] / episodes,
        "dynaq_win_rate": wins["dynaq"] / episodes,
        "draw_rate": wins["draw"] / episodes,
        "ppo_avg_reward": float(np.mean(rewards["ppo"])),
        "dynaq_avg_reward": float(np.mean(rewards["dynaq"])),
        "avg_episode_length": float(np.mean(lengths)),
        "ppo_danger_hits": danger["ppo"],
        "dynaq_danger_hits": danger["dynaq"],
    }

    print("\n──────────── Evaluation: PPO vs Dyna-Q ────────────")
    print(f"Episodes            : {episodes}")
    print(f"PPO win rate        : {result['ppo_win_rate']*100:6.1f}%")
    print(f"Dyna-Q win rate     : {result['dynaq_win_rate']*100:6.1f}%")
    print(f"Draw rate           : {result['draw_rate']*100:6.1f}%")
    print(f"PPO avg reward      : {result['ppo_avg_reward']:8.2f}")
    print(f"Dyna-Q avg reward   : {result['dynaq_avg_reward']:8.2f}")
    print(f"Avg episode length  : {result['avg_episode_length']:8.1f}")
    print(f"PPO danger hits     : {result['ppo_danger_hits']}")
    print(f"Dyna-Q danger hits  : {result['dynaq_danger_hits']}")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--render", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(default_config(), episodes=args.episodes, render=args.render)
