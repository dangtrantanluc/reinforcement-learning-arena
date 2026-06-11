"""Solo training: PPO vs Dyna-Q (vs DQN) on one shared Bomberman map.

SoloTrainer drives the env + all agents one *step* or *episode* at a time so the
FastAPI runner can advance training in a background thread while streaming frames.
It owns metrics, logs, the frame buffer (A1), visit heatmaps (B8), match-replay
recording (B7) and optional TensorBoard logging (C9).

CLI:  python -m src.training.train_solo
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Config, default_config
from src.env.competitive_grid_env import CompetitiveGridEnv, ACTION_NAMES
from src.env.state_encoder import neural_obs_dim
from src.agents.ppo.ppo_agent import PPOAgent
from src.agents.dynaq.dynaq_agent import DynaQAgent
from src.agents.dqn.dqn_agent import DQNAgent
from src.training.metrics import MetricsTracker
from src.utils import set_seed, get_device, ensure_dir


class SoloTrainer:
    """Stateful N-agent trainer usable both as a CLI loop and behind the API."""

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or default_config()
        set_seed(self.cfg.train.seed)
        self.device = get_device(self.cfg.train.device)

        # Agent roster: PPO + Dyna-Q always; DQN optional (B6).
        self.agent_ids = ["ppo", "dynaq"]
        if self.cfg.train.enable_dqn:
            self.agent_ids.append("dqn")

        self.env = CompetitiveGridEnv(self.cfg.env, agent_ids=tuple(self.agent_ids))
        obs_dim = neural_obs_dim(self.cfg.env.grid_size)
        n_act = self.env.n_actions

        self.ppo = PPOAgent(obs_dim, n_act, self.cfg.ppo, self.device)
        self.dynaq = DynaQAgent(self.cfg.dynaq, seed=self.cfg.train.seed)
        self.dqn = (DQNAgent(obs_dim, n_act, self.cfg.dqn, self.device)
                    if "dqn" in self.agent_ids else None)

        self.metrics = MetricsTracker(window=100)
        self.total_wins = {a: 0 for a in self.agent_ids}
        self.total_wins["draw"] = 0
        self.logs: Deque[str] = deque(maxlen=200)

        # Frame buffer (A1) + heatmaps (B8) + replay recording (B7).
        self.frames: Deque[dict] = deque(maxlen=4000)
        self.frame_seq = 0
        n = self.cfg.env.grid_size
        self.heatmap = {a: np.zeros((n, n), dtype=np.int32) for a in self.agent_ids}
        self._cur_replay: List[dict] = []
        self.match_history: Deque[dict] = deque(maxlen=50)

        # TensorBoard (C9) — lazily created, optional.
        self._tb = None

        # Live per-step state.
        self.obs: Dict[str, np.ndarray] = {}
        self.dynaq_state = None
        self.ep_step = 0
        self.episode = 0
        self.ep_reward = {a: 0.0 for a in self.agent_ids}
        self.last_winner: Optional[str] = None
        self._start_episode()

    # ── TensorBoard ─────────────────────────────────────────
    def _tb_writer(self):
        if self._tb is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                ensure_dir(self.cfg.train.tensorboard_dir)
                self._tb = SummaryWriter(self.cfg.train.tensorboard_dir)
            except Exception:
                self._tb = False  # disabled if tensorboard missing
        return self._tb or None

    # ── Episode lifecycle ───────────────────────────────────
    def _log(self, msg: str) -> None:
        self.logs.append(msg)

    def _emit_frame(self, kind: str, events: Optional[list] = None) -> None:
        self.frame_seq += 1
        frame = {
            "seq": self.frame_seq,
            "kind": kind,
            "episode": self.episode,
            "step": self.ep_step,
            "grid": self.env.grid_with_agents(),
            "positions": {a: list(self.env.pos[a]) for a in self.agent_ids},
            "alive": {a: self.env.alive[a] for a in self.agent_ids},
            "actions": {a: ACTION_NAMES[self.env.last_action[a]] for a in self.agent_ids},
            "rewards": {a: round(self.ep_reward[a], 2) for a in self.agent_ids},
            "winner": self.last_winner,
            "events": events or [],
        }
        self.frames.append(frame)
        # Record into the in-progress replay (B7) — keep the grid so the match
        # can be re-rendered faithfully later.
        self._cur_replay.append(dict(frame))

    def get_frames_since(self, seq: int, limit: int = 240) -> list:
        return [f for f in self.frames if f["seq"] > seq][:limit]

    def _apply_curriculum(self) -> None:
        """Scale map difficulty by training progress (C10)."""
        cur = self.cfg.curriculum
        if not cur.enabled:
            return
        ep = self.episode
        if ep <= cur.warmup_episodes:
            t = 0.0
        else:
            t = min(1.0, (ep - cur.warmup_episodes) / max(1, cur.ramp_episodes))
        lerp = lambda lo, hi: int(round(lo + (hi - lo) * t))
        self.env.cfg.n_walls = lerp(*cur.walls)
        self.env.cfg.n_boxes = lerp(*cur.boxes)
        self.env.cfg.n_dangers = lerp(*cur.dangers)

    def _start_episode(self) -> None:
        self._apply_curriculum()
        self.obs, _ = self.env.reset()
        self.dynaq_state = self.env.get_dynaq_state("dynaq")
        self.ep_step = 0
        self.ep_reward = {a: 0.0 for a in self.agent_ids}
        self.episode += 1
        self.last_winner = None
        self._cur_replay = []
        self._log(f"[Episode {self.episode}] started")
        self._emit_frame("reset")

    # ── One step ────────────────────────────────────────────
    def step_once(self) -> bool:
        # 1) Each agent picks an action from its own observation.
        ppo_state = self.obs["ppo"]
        ppo_action, ppo_lp, ppo_val = self.ppo.select_action(ppo_state, training=True)
        dynaq_action = self.dynaq.select_action(self.dynaq_state, training=True)
        actions = {"ppo": ppo_action, "dynaq": dynaq_action}
        if self.dqn is not None:
            dqn_state = self.obs["dqn"]
            dqn_action = self.dqn.select_action(dqn_state, training=True)
            actions["dqn"] = dqn_action

        # 2) Step the env.
        next_obs, rewards, dones, infos = self.env.step(actions)
        next_dynaq_state = self.env.get_dynaq_state("dynaq")

        # 3) Learn — each algorithm its own way.
        self.ppo.store_transition(ppo_state, ppo_action, rewards["ppo"], dones["ppo"], ppo_lp, ppo_val)
        if self.ppo.buffer_size() >= self.cfg.ppo.rollout_steps:
            self.ppo.update(last_state=next_obs["ppo"], last_done=dones["ppo"])

        self.dynaq.update(self.dynaq_state, dynaq_action, rewards["dynaq"],
                          next_dynaq_state, dones["dynaq"])

        if self.dqn is not None:
            self.dqn.store(dqn_state, dqn_action, rewards["dqn"], next_obs["dqn"], dones["dqn"])
            self.dqn.learn()

        # 4) Bookkeeping + events.
        for a in self.agent_ids:
            self.ep_reward[a] += rewards[a]
        self.ep_step += 1
        for a in self.agent_ids:
            r, c = self.env.pos[a]
            self.heatmap[a][r, c] += 1

        events = list(infos.get("bomb_events", []))
        for a in self.agent_ids:
            if rewards[a] >= self.cfg.env.r_reward_item - 1e-6:
                events.append({"agent": a, "type": "reward"})
                self._log(f"[Reward] {a.upper()} +{int(self.cfg.env.r_reward_item)}")
            if infos.get("danger_now", {}).get(a):
                events.append({"agent": a, "type": "danger"})
        for ev in infos.get("bomb_events", []):
            if ev.get("type") == "kill":
                self._log(f"[Bomb] {ev['agent'].upper()} killed at {ev['pos']}")
            elif ev.get("type") == "break_box":
                self._log(f"[Bomb] {ev['agent'].upper()} broke a box +{int(self.cfg.env.r_break_box)}")

        self.obs = next_obs
        self.dynaq_state = next_dynaq_state

        if dones["__all__"]:
            self._finish_episode(infos)
            self._emit_frame("end", events)
            return True

        self._emit_frame("step", events)
        return False

    def _finish_episode(self, infos: dict) -> None:
        winner = infos.get("winner")
        self.last_winner = winner
        key = winner if winner in self.total_wins else "draw"
        self.total_wins[key] += 1

        wlabel = winner.upper() if winner in self.agent_ids else "Draw"
        self._log(f"[Winner] {wlabel} ({infos.get('reason')})")

        # Flush PPO remainder so short episodes still teach it.
        if self.ppo.buffer_size() > 0:
            self.ppo.update(last_state=self.obs["ppo"], last_done=True)
        self.dynaq.decay_epsilon()
        if self.dqn is not None:
            self.dqn.decay_epsilon()

        self.metrics.record_episode(
            ppo_reward=self.ep_reward["ppo"],
            dynaq_reward=self.ep_reward["dynaq"],
            winner=winner,
            length=self.ep_step,
            dynaq_epsilon=self.dynaq.epsilon,
            q_table_size=self.dynaq.q_table_size,
            ppo_losses=self.ppo.last_losses,
            dqn_reward=self.ep_reward.get("dqn"),
            dqn_epsilon=self.dqn.epsilon if self.dqn else None,
            dqn_loss=self.dqn.last_loss if self.dqn else None,
        )

        # Save the finished match for replay (B7).
        self.match_history.append({
            "episode": self.episode,
            "winner": winner or "draw",
            "reason": infos.get("reason"),
            "length": self.ep_step,
            "rewards": {a: round(self.ep_reward[a], 2) for a in self.agent_ids},
            "frames": list(self._cur_replay),
        })

        # TensorBoard scalars (C9).
        tb = self._tb_writer()
        if tb:
            tb.add_scalar("reward/ppo", self.ep_reward["ppo"], self.episode)
            tb.add_scalar("reward/dynaq", self.ep_reward["dynaq"], self.episode)
            tb.add_scalar("winrate/ppo", self.metrics.win_rate("ppo"), self.episode)
            tb.add_scalar("winrate/dynaq", self.metrics.win_rate("dynaq"), self.episode)
            tb.add_scalar("loss/ppo_policy", self.ppo.last_losses["policy_loss"], self.episode)
            tb.add_scalar("epsilon/dynaq", self.dynaq.epsilon, self.episode)
            if self.dqn is not None:
                tb.add_scalar("reward/dqn", self.ep_reward["dqn"], self.episode)
                tb.add_scalar("winrate/dqn", self.metrics.win_rate("dqn"), self.episode)
                tb.add_scalar("loss/dqn", self.dqn.last_loss, self.episode)

    def run_episode(self) -> None:
        done = False
        while not done:
            done = self.step_once()
        self._start_episode()

    # ── Snapshot for the API ────────────────────────────────
    def _agent_block(self, a: str) -> dict:
        common = {
            "position": list(self.env.pos[a]),
            "episode_reward": round(self.ep_reward[a], 2),
            "total_wins": self.total_wins[a],
            "last_action": ACTION_NAMES[self.env.last_action[a]],
            "alive": self.env.alive[a],
        }
        if a == "ppo":
            common.update({
                "action_probs": self.ppo.get_action_probs(self.obs["ppo"]),
                "policy_loss": round(self.ppo.last_losses["policy_loss"], 4),
                "value_loss": round(self.ppo.last_losses["value_loss"], 4),
                "entropy": round(self.ppo.last_losses["entropy"], 4),
            })
        elif a == "dynaq":
            common.update({
                "epsilon": round(self.dynaq.epsilon, 4),
                "q_table_size": self.dynaq.q_table_size,
                "planning_steps": self.cfg.dynaq.planning_steps,
                "q_values": self.dynaq.get_q_values(self.dynaq_state),
            })
        elif a == "dqn" and self.dqn is not None:
            common.update({
                "epsilon": round(self.dqn.epsilon, 4),
                "loss": round(self.dqn.last_loss, 4),
                "buffer": len(self.dqn.buffer),
                "q_values": self.dqn.get_q_values(self.obs["dqn"]),
            })
        return common

    def live_state(self) -> dict:
        m = self.metrics
        state = {
            "episode": self.episode,
            "step": self.ep_step,
            "grid": self.env.grid_with_agents(),
            "winner": self.last_winner,
            "agents": self.agent_ids,
            "ppo": self._agent_block("ppo"),
            "dynaq": self._agent_block("dynaq"),
            "metrics": {
                "ppo_win_rate": round(m.win_rate("ppo"), 3),
                "dynaq_win_rate": round(m.win_rate("dynaq"), 3),
                "draw_rate": round(m.win_rate("draw"), 3),
                "ppo_avg_reward": round(m.avg("ppo"), 2),
                "dynaq_avg_reward": round(m.avg("dynaq"), 2),
            },
            "history": m.history[-200:],
            "logs": list(self.logs)[-30:],
            "last_frame_seq": self.frame_seq,
            "curriculum": {
                "enabled": self.cfg.curriculum.enabled,
                "walls": self.env.cfg.n_walls,
                "boxes": self.env.cfg.n_boxes,
                "dangers": self.env.cfg.n_dangers,
            },
        }
        if self.dqn is not None:
            state["dqn"] = self._agent_block("dqn")
            state["metrics"]["dqn_win_rate"] = round(m.win_rate("dqn"), 3)
            state["metrics"]["dqn_avg_reward"] = round(m.avg("dqn"), 2)
        return state

    def heatmaps(self) -> dict:
        out = {}
        for a in self.agent_ids:
            hm = self.heatmap[a].astype(float)
            mx = hm.max() or 1.0
            out[a] = (hm / mx).round(3).tolist()
        return out

    def list_replays(self) -> list:
        """Lightweight match list for the replay browser (B7)."""
        return [{"episode": r["episode"], "winner": r["winner"], "reason": r["reason"],
                 "length": r["length"], "rewards": r["rewards"]} for r in self.match_history]

    def get_replay(self, episode: int) -> Optional[dict]:
        for r in self.match_history:
            if r["episode"] == episode:
                return r
        return None

    def save(self) -> None:
        ensure_dir(os.path.dirname(self.cfg.train.ppo_ckpt))
        ensure_dir(os.path.dirname(self.cfg.train.dynaq_ckpt))
        self.ppo.save(self.cfg.train.ppo_ckpt)
        self.dynaq.save(self.cfg.train.dynaq_ckpt)
        if self.dqn is not None:
            ensure_dir(os.path.dirname(self.cfg.train.dqn_ckpt))
            self.dqn.save(self.cfg.train.dqn_ckpt)
        ensure_dir(self.cfg.train.log_dir)
        self.metrics.save(self.cfg.train.metrics_file)

    def load_checkpoints(self) -> dict:
        from .checkpoint import load_ppo, load_dynaq
        ppo_ok = load_ppo(self.ppo, self.cfg.train.ppo_ckpt)
        dynaq_ok = load_dynaq(self.dynaq, self.cfg.train.dynaq_ckpt)
        dqn_ok = False
        if self.dqn is not None and os.path.exists(self.cfg.train.dqn_ckpt):
            self.dqn.load(self.cfg.train.dqn_ckpt)
            dqn_ok = True
        self._log(f"[Checkpoint] loaded ppo={ppo_ok} dynaq={dynaq_ok} dqn={dqn_ok}")
        return {"ppo": ppo_ok, "dynaq": dynaq_ok, "dqn": dqn_ok}


def train(cfg: Optional[Config] = None) -> None:
    """CLI training loop."""
    cfg = cfg or default_config()
    trainer = SoloTrainer(cfg)
    print(f"[setup] device={trainer.device} agents={trainer.agent_ids} "
          f"episodes={cfg.train.total_episodes}")

    for ep in range(cfg.train.total_episodes):
        trainer.run_episode()
        if (ep + 1) % 20 == 0:
            m = trainer.metrics
            extra = ""
            if trainer.dqn is not None:
                extra = f" DQN_win={m.win_rate('dqn')*100:5.1f}%"
            print(f"[ep {ep + 1:>5}/{cfg.train.total_episodes}] "
                  f"PPO_win={m.win_rate('ppo')*100:5.1f}% "
                  f"DynaQ_win={m.win_rate('dynaq')*100:5.1f}%{extra} "
                  f"draw={m.win_rate('draw')*100:5.1f}% "
                  f"len={m.avg_length:4.1f} eps_dq={trainer.dynaq.epsilon:.3f} "
                  f"|Q|={trainer.dynaq.q_table_size}")
        if (ep + 1) % cfg.train.save_every == 0:
            trainer.save()

    trainer.save()
    print("[done] checkpoints + metrics saved")


if __name__ == "__main__":
    train()
