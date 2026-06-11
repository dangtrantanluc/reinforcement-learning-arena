"""FastAPI realtime server for the PPO vs Dyna-Q arena.

Endpoints:
  GET  /api/state          → live snapshot (frontend polls this @ 500ms)
  POST /api/train/start    → start/resume background training
  POST /api/train/pause    → pause training
  POST /api/train/reset    → reset env + agents + metrics + logs
  POST /api/evaluate       → run a deterministic evaluation, return the table

Run:  uvicorn src.api.server:app --reload --port 8000   (from backend/)
"""

from __future__ import annotations

import os
import sys

# Make `src...` importable when launched as `uvicorn src.api.server:app` from backend/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import ActionResponse, EvaluateResponse
from src.api.training_runner import TrainingRunner

app = FastAPI(title="PPO vs Dyna-Q Solo Arena", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # demo — tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared runner for the process.
runner = TrainingRunner()


@app.get("/api/state")
def get_state():
    """Full live state — grid, both agents, metrics, history, logs."""
    return runner.snapshot()


@app.post("/api/train/start", response_model=ActionResponse)
def start_training():
    runner.start()
    return ActionResponse(ok=True, message="Training started")


@app.post("/api/train/pause", response_model=ActionResponse)
def pause_training():
    runner.pause()
    return ActionResponse(ok=True, message="Training paused")


@app.post("/api/train/reset", response_model=ActionResponse)
def reset_training():
    runner.reset()
    return ActionResponse(ok=True, message="Training reset")


@app.post("/api/evaluate", response_model=EvaluateResponse)
def run_evaluation(episodes: int = 50):
    result = runner.evaluate(episodes=episodes)
    return EvaluateResponse(ok=True, result=result)


@app.get("/")
def root():
    return {"service": "ppo-dynaq-solo", "docs": "/docs", "state": "/api/state"}


@app.on_event("shutdown")
def _shutdown():
    runner.shutdown()
