# Backend — PPO vs Dyna-Q Arena

Python backend: custom Gymnasium environment, hai RL agent tự implement (PyTorch + NumPy),
training loop, evaluation, và FastAPI realtime server. **Không stable-baselines3.**

## Cài đặt & chạy

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# API realtime (frontend poll endpoint này)
uvicorn src.api.server:app --reload --host :: --port 8001

# hoặc train bằng CLI
python -m src.training.train_solo
python -m src.training.evaluate_solo --episodes 100
```

Smoke test toàn stack:

```bash
python tests/smoke_test.py
```

## Cấu trúc

```
src/
├── config.py                 # mọi hyperparameter (dataclass)
├── utils.py                  # seed, device, RunningMean
├── env/
│   ├── competitive_grid_env.py   # CompetitiveGridEnv (multi-agent, 1 map chung)
│   ├── map_generator.py          # sinh map (random / fixed)
│   ├── reward.py                 # reward shaping helpers
│   └── state_encoder.py          # encode_ppo_obs + encode_dynaq_state
├── agents/
│   ├── base_agent.py
│   ├── ppo/   actor_critic.py · rollout_buffer.py · ppo_agent.py
│   └── dynaq/ q_table.py · world_model.py · dynaq_agent.py
├── training/
│   ├── train_solo.py         # SoloTrainer (dùng cả CLI lẫn API) + train()
│   ├── evaluate_solo.py      # head-to-head deterministic eval
│   ├── metrics.py            # MetricsTracker (rolling rates + history)
│   └── checkpoint.py
└── api/
    ├── server.py             # FastAPI app + endpoints
    ├── schemas.py            # Pydantic response models
    └── training_runner.py    # background-thread training, shared state + Lock
```

## API Endpoints

| Method | Path | Mô tả |
|--------|------|-------|
| `GET`  | `/api/state` | Snapshot realtime: grid, 2 agent, metrics, history, logs |
| `POST` | `/api/train/start` | Bắt đầu / tiếp tục training (background thread) |
| `POST` | `/api/train/pause` | Tạm dừng |
| `POST` | `/api/train/reset` | Reset env + agents + metrics + logs |
| `POST` | `/api/evaluate?episodes=50` | Chạy evaluation, trả bảng so sánh |

Swagger UI: `http://localhost:8001/docs`.

## PPO — cách hoạt động

Actor-Critic MLP (128-128). Mỗi step lưu transition vào `RolloutBuffer`; khi đủ
`rollout_steps` thì update: tính **GAE-λ** advantage, **clipped surrogate** policy loss,
clipped value loss, entropy bonus, gradient clipping. Hyperparams trong `config.py:PPOConfig`.

## Dyna-Q — cách hoạt động

Q-table (`dict[state → 5 giá trị]`) + **world model** (`(state,action) → (next,reward,done)`).
Mỗi real step: (1) ε-greedy chọn action, (2) Q-learning update từ experience thật, (3) lưu
transition, (4) **planning**: replay `planning_steps` transition ngẫu nhiên từ model. Epsilon
decay theo episode. Hyperparams trong `config.py:DynaQConfig`.

## Kết quả training (sanity check)

Một short run (120 episodes, map cố định) cho thấy cả hai học được — PPO reward đi từ ~−53 lên
dương, win rate tăng dần, episode length giảm (đi hiệu quả hơn), Q-table Dyna-Q tăng từ vài
trăm lên vài nghìn state, epsilon giảm theo lịch. Chạy full `total_episodes` để hội tụ tốt hơn.
