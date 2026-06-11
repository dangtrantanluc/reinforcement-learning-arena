"""Common agent interface so the trainer can treat PPO and Dyna-Q uniformly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Minimal contract shared by both algorithms.

    Note: the two agents have genuinely different learning loops (PPO is on-policy
    batch; Dyna-Q is online tabular), so only the universal methods are abstract.
    """

    name: str = "base"

    @abstractmethod
    def select_action(self, state: Any, training: bool = True):
        """Return an action (and, for PPO, extra info needed to store transitions)."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError
