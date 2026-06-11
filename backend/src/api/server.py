"""FastAPI realtime server for the PPO vs Dyna-Q arena.

Endpoints:
  GET  /api/state          → live snapshot
  GET  /api/frames?since=N  → step-by-step frames for replay (A1)
  GET  /api/heatmaps        → per-agent visit heatmaps (B8)
  WS   /ws/frames           → realtime frame stream (A2)
  POST /api/train/start|pause|reset|speed
  POST /api/checkpoint/save|load   (A4)
  POST /api/evaluate

Run:  uvicorn src.api.server:app --reload --host :: --port 8001   (from backend/)
"""

from __future__ import annotations

import asyncio
import os
import sys

# Make `src...` importable when launched as `uvicorn src.api.server:app` from backend/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import ActionResponse, EvaluateResponse
from src.api.training_runner import TrainingRunner
from src.logging_setup import (
    CorrelationIdMiddleware, global_exception_handler, logger,
)

app = FastAPI(title="PPO vs Dyna-Q vs DQN Arena", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # demo — tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(CorrelationIdMiddleware)
app.add_exception_handler(Exception, global_exception_handler)
logger.info("Arena API starting")

# Single shared runner for the process.
runner = TrainingRunner()


@app.get("/api/state")
def get_state():
    """Full live state — grid, both agents, metrics, history, logs."""
    return runner.snapshot()


@app.get("/api/frames")
def get_frames(since: int = 0, limit: int = 240):
    """Frames newer than `since` for step-by-step replay (A1)."""
    return runner.frames_since(since, limit)


@app.get("/api/heatmaps")
def get_heatmaps():
    """Per-agent visit heatmaps (B8)."""
    return runner.heatmaps()


@app.get("/api/replays")
def list_replays():
    """Recent finished matches (B7)."""
    return {"replays": runner.list_replays()}


@app.get("/api/replays/{episode}")
def get_replay(episode: int):
    """Full frame track of one match for playback (B7)."""
    r = runner.get_replay(episode)
    return r or {"error": "not found"}


@app.post("/api/train/start", response_model=ActionResponse)
def start_training():
    runner.start()
    return ActionResponse(ok=True, message="Training started")


@app.post("/api/train/pause", response_model=ActionResponse)
def pause_training():
    runner.pause()
    return ActionResponse(ok=True, message="Training paused")


@app.post("/api/train/reset", response_model=ActionResponse)
def reset_training(seed: int | None = None):
    runner.reset(seed=seed)
    msg = f"Reset with seed {seed}" if seed is not None else "Training reset"
    return ActionResponse(ok=True, message=msg)


@app.post("/api/train/speed", response_model=ActionResponse)
def set_speed(delay: float = 0.06):
    """Set the env-step cadence in seconds (A3 speed slider)."""
    runner.set_speed(delay)
    return ActionResponse(ok=True, message=f"Step delay set to {delay}s")


@app.get("/api/config")
def get_config():
    """Current editable hyperparameters (C11)."""
    return runner.get_config()


@app.post("/api/config")
def update_config(patch: dict = Body(...)):
    """Patch hyperparameters and rebuild the trainer (C11). Resets training."""
    return {"ok": True, "config": runner.update_config(patch)}


@app.post("/api/checkpoint/save")
def checkpoint_save():
    return {"ok": True, "saved": runner.save_checkpoint()}


@app.post("/api/checkpoint/load")
def checkpoint_load():
    return {"ok": True, "loaded": runner.load_checkpoint()}


@app.post("/api/evaluate", response_model=EvaluateResponse)
def run_evaluation(episodes: int = 50):
    result = runner.evaluate(episodes=episodes)
    return EvaluateResponse(ok=True, result=result)


@app.websocket("/ws/frames")
async def ws_frames(ws: WebSocket):
    """Stream frames in order (A2). Client may send the last seq it has seen;
    the server pushes everything newer at a steady cadence."""
    await ws.accept()
    last_seq = 0
    try:
        while True:
            payload = runner.frames_since(last_seq, limit=240)
            if payload["frames"]:
                last_seq = payload["frames"][-1]["seq"]
            await ws.send_json(payload)
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return
    except Exception:
        # Client gone / send failed — close quietly.
        return


@app.get("/")
def root():
    return {"service": "ppo-dynaq-solo", "docs": "/docs", "state": "/api/state"}


@app.on_event("shutdown")
def _shutdown():
    runner.shutdown()
