"""Shared helpers: seeding, device, directory + rolling stats."""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Deque

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(pref: str = "auto"):
    import torch

    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


class RunningMean:
    """Rolling-window mean for smoothed metric logging."""

    def __init__(self, window: int = 100) -> None:
        self.buf: Deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self.buf.append(float(value))

    @property
    def mean(self) -> float:
        return float(np.mean(self.buf)) if self.buf else 0.0

    def __len__(self) -> int:
        return len(self.buf)
