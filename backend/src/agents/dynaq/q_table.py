"""Q-table — a defaultdict-style map from discrete state → action values.

Unseen states are lazily initialised to a zero vector of length n_actions.
Backed by a plain dict (pickle-friendly for checkpointing).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class QTable:
    def __init__(self, n_actions: int = 5) -> None:
        self.n_actions = n_actions
        self.table: Dict[Tuple[int, ...], np.ndarray] = {}

    def _ensure(self, state: Tuple[int, ...]) -> np.ndarray:
        q = self.table.get(state)
        if q is None:
            q = np.zeros(self.n_actions, dtype=np.float32)
            self.table[state] = q
        return q

    def get(self, state: Tuple[int, ...]) -> np.ndarray:
        """Return the action-value vector for `state` (creates it if new)."""
        return self._ensure(state)

    def value(self, state: Tuple[int, ...], action: int) -> float:
        return float(self._ensure(state)[action])

    def set(self, state: Tuple[int, ...], action: int, value: float) -> None:
        self._ensure(state)[action] = value

    def max_value(self, state: Tuple[int, ...]) -> float:
        return float(self._ensure(state).max())

    def argmax(self, state: Tuple[int, ...]) -> int:
        return int(np.argmax(self._ensure(state)))

    def __len__(self) -> int:
        return len(self.table)

    # Persistence helpers — convert to/from plain python for pickle stability.
    def to_serializable(self) -> dict:
        return {
            "n_actions": self.n_actions,
            "table": {k: v.tolist() for k, v in self.table.items()},
        }

    @classmethod
    def from_serializable(cls, data: dict) -> "QTable":
        qt = cls(n_actions=data["n_actions"])
        qt.table = {k: np.asarray(v, dtype=np.float32) for k, v in data["table"].items()}
        return qt
