"""Checkpoint helpers — thin wrappers so train/eval/API share save+load paths."""

from __future__ import annotations

import os

from ..agents.ppo.ppo_agent import PPOAgent
from ..agents.dynaq.dynaq_agent import DynaQAgent
from ..utils import ensure_dir


def save_all(ppo: PPOAgent, dynaq: DynaQAgent, ppo_path: str, dynaq_path: str) -> None:
    ensure_dir(os.path.dirname(ppo_path))
    ensure_dir(os.path.dirname(dynaq_path))
    ppo.save(ppo_path)
    dynaq.save(dynaq_path)


def load_ppo(ppo: PPOAgent, path: str) -> bool:
    if os.path.exists(path):
        ppo.load(path)
        return True
    return False


def load_dynaq(dynaq: DynaQAgent, path: str) -> bool:
    if os.path.exists(path):
        dynaq.load(path)
        return True
    return False
