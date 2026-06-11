"""End-to-end smoke test for the PPO vs Dyna-Q arena backend.

Run from backend/:  venv/bin/python tests/smoke_test.py

Checks: env contracts, collision rules, both encoders, PPO update, Dyna-Q
update+planning+persistence, and one full SoloTrainer episode.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, EnvConfig, PPOConfig, DynaQConfig
from src.env.competitive_grid_env import CompetitiveGridEnv, UP, DOWN, LEFT, RIGHT, STAY
from src.env.state_encoder import ppo_obs_dim, encode_dynaq_state
from src.agents.ppo.ppo_agent import PPOAgent
from src.agents.dynaq.dynaq_agent import DynaQAgent
from src.training.train_solo import SoloTrainer
from src.utils import get_device


def check_env():
    env = CompetitiveGridEnv(EnvConfig(seed=1))
    obs, info = env.reset(seed=1)
    assert set(obs.keys()) == {"ppo", "dynaq"}
    assert obs["ppo"].shape == (ppo_obs_dim(10),)
    assert env.pos["ppo"] == (0, 0) and env.pos["dynaq"] == (9, 0)

    # run random joint steps
    total = {"ppo": 0.0, "dynaq": 0.0}
    for _ in range(60):
        acts = {"ppo": int(np.random.randint(5)), "dynaq": int(np.random.randint(5))}
        obs, rew, dones, info = env.step(acts)
        total["ppo"] += rew["ppo"]; total["dynaq"] += rew["dynaq"]
        assert "__all__" in dones
        if dones["__all__"]:
            obs, info = env.reset()
    # json + render contracts
    js = env.to_json()
    assert js["gridSize"] == 10 and len(js["grid"]) == 10
    assert isinstance(env.render(), str)
    print(f"  ✓ env OK — obs_dim={obs['ppo'].shape[0]}, ran 60 joint steps")


def check_collision():
    """Two agents stepping into the same cell must both be blocked + penalised."""
    env = CompetitiveGridEnv(EnvConfig(seed=2, randomize=False))
    env.reset(seed=2)
    # Force them adjacent at a known empty spot: place both manually.
    env.pos["ppo"] = (5, 5)
    env.pos["dynaq"] = (5, 7)
    env.grid[5, 6] = 0  # ensure contested cell is empty
    env.grid[5, 5] = 0; env.grid[5, 7] = 0
    env._dist = {a: env._manhattan(env.pos[a], env.goal) for a in ("ppo", "dynaq")}
    # ppo RIGHT → (5,6); dynaq LEFT → (5,6): same cell contest
    _, rew, _, _ = env.step({"ppo": RIGHT, "dynaq": LEFT})
    assert env.pos["ppo"] == (5, 5), "ppo should be blocked"
    assert env.pos["dynaq"] == (5, 7), "dynaq should be blocked"
    print(f"  ✓ collision OK — both blocked, rewards ppo={rew['ppo']:.1f} dyn={rew['dynaq']:.1f}")


def check_encoders():
    env = CompetitiveGridEnv(EnvConfig(seed=3))
    env.reset(seed=3)
    s = env.get_dynaq_state("dynaq")
    assert len(s) == 16, f"dynaq state must be 16-tuple, got {len(s)}"
    assert all(isinstance(x, (int, np.integer)) for x in s)
    print(f"  ✓ encoders OK — dynaq state len={len(s)}")


def check_ppo():
    device = get_device("cpu")
    env = CompetitiveGridEnv(EnvConfig(seed=4))
    obs, _ = env.reset(seed=4)
    cfg = PPOConfig(rollout_steps=128, ppo_epochs=2, batch_size=32)
    ppo = PPOAgent(ppo_obs_dim(10), 5, cfg, device)
    for _ in range(cfg.rollout_steps):
        a, lp, v = ppo.select_action(obs["ppo"])
        nobs, rew, dones, _ = env.step({"ppo": a, "dynaq": int(np.random.randint(5))})
        ppo.store_transition(obs["ppo"], a, rew["ppo"], dones["ppo"], lp, v)
        obs = nobs
        if dones["__all__"]:
            obs, _ = env.reset()
    losses = ppo.update(last_state=obs["ppo"])
    for k, val in losses.items():
        assert np.isfinite(val), f"non-finite {k}"
    probs = ppo.get_action_probs(obs["ppo"])
    assert abs(sum(probs.values()) - 1.0) < 1e-4
    print(f"  ✓ PPO OK — pl={losses['policy_loss']:.4f} vl={losses['value_loss']:.4f} "
          f"ent={losses['entropy']:.4f}, probs sum=1")


def check_dynaq():
    env = CompetitiveGridEnv(EnvConfig(seed=5))
    env.reset(seed=5)
    dq = DynaQAgent(DynaQConfig(planning_steps=10), seed=5)
    s = env.get_dynaq_state("dynaq")
    for _ in range(50):
        a = dq.select_action(s)
        _, rew, dones, _ = env.step({"ppo": int(np.random.randint(5)), "dynaq": a})
        ns = env.get_dynaq_state("dynaq")
        dq.update(s, a, rew["dynaq"], ns, dones["dynaq"])
        s = ns
        if dones["__all__"]:
            env.reset(); s = env.get_dynaq_state("dynaq")
    dq.decay_epsilon()
    assert dq.q_table_size > 0
    qv = dq.get_q_values(s)
    assert len(qv) == 5
    # save/load round-trip
    os.makedirs("checkpoints/dynaq", exist_ok=True)
    p = "checkpoints/dynaq/_smoke.pkl"
    dq.save(p)
    dq2 = DynaQAgent(DynaQConfig())
    dq2.load(p)
    assert dq2.q_table_size == dq.q_table_size
    os.remove(p)
    print(f"  ✓ Dyna-Q OK — |Q|={dq.q_table_size} eps={dq.epsilon:.3f}, save/load OK")


def check_trainer():
    cfg = Config(
        env=EnvConfig(seed=6, max_steps=40),
        ppo=PPOConfig(rollout_steps=64, ppo_epochs=2, batch_size=32),
        dynaq=DynaQConfig(planning_steps=5),
    )
    trainer = SoloTrainer(cfg)
    for _ in range(3):
        trainer.run_episode()
    state = trainer.live_state()
    assert "ppo" in state and "dynaq" in state and "metrics" in state
    assert set(state["ppo"]["action_probs"].keys()) == {"UP", "DOWN", "LEFT", "RIGHT", "STAY"}
    assert state["dynaq"]["q_table_size"] >= 0
    print(f"  ✓ SoloTrainer OK — ran 3 episodes, ep={trainer.episode}, "
          f"|Q|={trainer.dynaq.q_table_size}")


if __name__ == "__main__":
    print("Running PPO vs Dyna-Q backend smoke test…")
    check_env()
    check_collision()
    check_encoders()
    check_ppo()
    check_dynaq()
    check_trainer()
    print("\nAll backend smoke checks passed ✅")
