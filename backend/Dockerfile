# Backend — PPO/Dyna-Q/DQN arena (FastAPI + PyTorch CPU)
FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; torch CPU wheels need no extras.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src/

# Checkpoints + logs are written at runtime.
RUN mkdir -p checkpoints/ppo checkpoints/dynaq checkpoints/dqn logs/replays logs/tensorboard

EXPOSE 8001
# Listen on :: so both IPv4 and IPv6 clients reach it.
CMD ["uvicorn", "src.api.server:app", "--host", "::", "--port", "8001"]
