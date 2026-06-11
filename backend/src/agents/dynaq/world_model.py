"""Deterministic world model for Dyna-Q planning.

Stores model[(state, action)] = (next_state, reward, done). During planning the
agent samples previously-seen (state, action) pairs and replays them as if they
were real experience — this is what makes Dyna-Q sample-efficient.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

StateT = Tuple[int, ...]
KeyT = Tuple[StateT, int]
ValueT = Tuple[StateT, float, bool]


class WorldModel:
    def __init__(self) -> None:
        self.model: Dict[KeyT, ValueT] = {}
        self._keys: List[KeyT] = []  # parallel list for O(1) random sampling

    def add(
        self, state: StateT, action: int, next_state: StateT, reward: float, done: bool
    ) -> None:
        key = (state, action)
        if key not in self.model:
            self._keys.append(key)
        self.model[key] = (next_state, float(reward), bool(done))

    def sample(self, rng: np.random.Generator) -> Tuple[StateT, int, StateT, float, bool]:
        """Sample a random stored transition. Caller must check `len(self) > 0`."""
        idx = int(rng.integers(0, len(self._keys)))
        state, action = self._keys[idx]
        next_state, reward, done = self.model[(state, action)]
        return state, action, next_state, reward, done

    def __len__(self) -> int:
        return len(self._keys)

    def to_serializable(self) -> dict:
        return {"model": [[list(k[0]), k[1], list(v[0]), v[1], v[2]] for k, v in self.model.items()]}

    @classmethod
    def from_serializable(cls, data: dict) -> "WorldModel":
        wm = cls()
        for s, a, ns, r, d in data["model"]:
            wm.add(tuple(s), int(a), tuple(ns), float(r), bool(d))
        return wm
