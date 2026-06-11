"""Pydantic response schemas for the API (documentation + validation)."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class PPOState(BaseModel):
    position: List[int]
    episode_reward: float
    total_wins: int
    last_action: str
    action_probs: Dict[str, float]
    policy_loss: float
    value_loss: float
    entropy: float


class DynaQState(BaseModel):
    position: List[int]
    episode_reward: float
    total_wins: int
    last_action: str
    epsilon: float
    q_table_size: int
    planning_steps: int
    q_values: Dict[str, float]


class MetricsState(BaseModel):
    ppo_win_rate: float
    dynaq_win_rate: float
    draw_rate: float
    ppo_avg_reward: float
    dynaq_avg_reward: float


class StateResponse(BaseModel):
    running: bool
    episode: int
    step: int
    grid: List[List[int]]
    winner: Optional[str]
    ppo: PPOState
    dynaq: DynaQState
    metrics: MetricsState
    history: List[dict]
    logs: List[str]


class ActionResponse(BaseModel):
    ok: bool
    message: str


class EvaluateResponse(BaseModel):
    ok: bool
    result: dict
