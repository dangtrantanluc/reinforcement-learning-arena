"""Metrics tracker — accumulates per-episode results into rolling rates + history."""

from __future__ import annotations

import json
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np


class MetricsTracker:
    def __init__(self, window: int = 100) -> None:
        self.window = window
        self.episode = 0

        self._winners: Deque[str] = deque(maxlen=window)
        self._ppo_rewards: Deque[float] = deque(maxlen=window)
        self._dynaq_rewards: Deque[float] = deque(maxlen=window)
        self._dqn_rewards: Deque[float] = deque(maxlen=window)
        self._lengths: Deque[int] = deque(maxlen=window)

        self.history: List[Dict] = []  # one record per episode (for charts)

    def record_episode(
        self,
        ppo_reward: float,
        dynaq_reward: float,
        winner: Optional[str],
        length: int,
        dynaq_epsilon: float,
        q_table_size: int,
        ppo_losses: Dict[str, float],
        dqn_reward: Optional[float] = None,
        dqn_epsilon: Optional[float] = None,
        dqn_loss: Optional[float] = None,
    ) -> Dict:
        self.episode += 1
        self._winners.append(winner or "draw")
        self._ppo_rewards.append(ppo_reward)
        self._dynaq_rewards.append(dynaq_reward)
        if dqn_reward is not None:
            self._dqn_rewards.append(dqn_reward)
        self._lengths.append(length)

        record = {
            "episode": self.episode,
            "ppo_reward": round(ppo_reward, 2),
            "dynaq_reward": round(dynaq_reward, 2),
            "winner": winner or "draw",
            "ppo_win_rate": round(self.win_rate("ppo"), 3),
            "dynaq_win_rate": round(self.win_rate("dynaq"), 3),
            "draw_rate": round(self.win_rate("draw"), 3),
            "episode_length": length,
            "ppo_avg_reward": round(self.avg("ppo"), 2),
            "dynaq_avg_reward": round(self.avg("dynaq"), 2),
            "dynaq_epsilon": round(dynaq_epsilon, 4),
            "dynaq_q_table_size": q_table_size,
            "ppo_policy_loss": round(ppo_losses.get("policy_loss", 0.0), 4),
            "ppo_value_loss": round(ppo_losses.get("value_loss", 0.0), 4),
            "ppo_entropy": round(ppo_losses.get("entropy", 0.0), 4),
        }
        if dqn_reward is not None:
            record["dqn_reward"] = round(dqn_reward, 2)
            record["dqn_win_rate"] = round(self.win_rate("dqn"), 3)
            record["dqn_avg_reward"] = round(self.avg("dqn"), 2)
            record["dqn_epsilon"] = round(dqn_epsilon or 0.0, 4)
            record["dqn_loss"] = round(dqn_loss or 0.0, 4)
        self.history.append(record)
        return record

    def win_rate(self, who: str) -> float:
        if not self._winners:
            return 0.0
        return sum(1 for w in self._winners if w == who) / len(self._winners)

    def avg(self, who: str) -> float:
        buf = {"ppo": self._ppo_rewards, "dynaq": self._dynaq_rewards,
               "dqn": self._dqn_rewards}.get(who, self._ppo_rewards)
        return float(np.mean(buf)) if buf else 0.0

    @property
    def avg_length(self) -> float:
        return float(np.mean(self._lengths)) if self._lengths else 0.0

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def reset(self) -> None:
        self.__init__(self.window)
