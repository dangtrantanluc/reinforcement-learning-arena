# RL Arena — PPO vs Dyna-Q vs DQN (Bomberman)

**Ba** agent reinforcement learning **cạnh tranh trực tiếp trên cùng một map** Bomberman
(grid-world có bom/box/explosion). Cả ba tự implement từ đầu bằng PyTorch / NumPy —
**không dùng stable-baselines3**.

| | Thuật toán | Cách học |
|---|-----------|----------|
| 🔵 **PPO** | Proximal Policy Optimization | Neural Actor-Critic, on-policy, GAE + clipped objective |
| 🟣 **Dyna-Q** | Tabular Q-learning + planning | Q-table + world model, model-based planning |
| 🟢 **DQN** | Deep Q-Network | Neural, off-policy, experience replay + Double-DQN target |

Frontend React hiển thị **realtime** cả ba agent đua nhau qua **WebSocket** (replay từng
bước, không nhảy ô), action probabilities / Q-values, biểu đồ so sánh, heatmap, replay
browser, và bảng tinh chỉnh hyperparameter.

## Tính năng nổi bật

- 🎬 **Step-replay buffer + WebSocket**: backend đẩy từng bước, frontend phát lại ở tốc độ cố
  định → agent đi từng ô (không còn ảo giác "xuyên tường" do poll thưa).
- 💣 **Bomberman**: đặt bom (action 5) → đếm giờ → nổ phá box (+5) và **giết đối thủ** (đối
  thủ chết thì bạn thắng +20).
- 🟢 **3 thuật toán** cạnh tranh trên cùng map: PPO (on-policy) vs Dyna-Q (tabular) vs DQN (off-policy).
- 🎛️ **Speed / New Map / Seed / Load-Save checkpoint** ngay trên UI.
- 📈 **Heatmap** vết chân mỗi agent, **biểu đồ so sánh** (Recharts), **TensorBoard** scalars.
- 🎞️ **Replay browser**: xem lại từng trận đã đấu, tua tới/lui.
- ⚙️ **Hyperparameter panel**: chỉnh lr/γ/clip/planning... rồi Apply để train lại.
- 🎓 **Curriculum learning**: map dễ → khó dần theo số episode.
- ✅ **CI + unit test** (pytest cho env logic), structured JSON logging + correlationId,
  TS types sinh tự động từ OpenAPI.

```
agent-rf/
├── backend/          # Python: env + agents + training + FastAPI
│   ├── src/
│   │   ├── env/      # CompetitiveGridEnv (Bomberman), map_generator, reward, state_encoder
│   │   ├── agents/   # ppo/  dynaq/  dqn/   (+ base_agent)
│   │   ├── training/ # train_solo (SoloTrainer), evaluate_solo, metrics, checkpoint
│   │   ├── api/      # server, schemas, training_runner (frame buffer, WS, config)
│   │   └── logging_setup.py
│   ├── tests/        # smoke_test.py + test_env.py (pytest)
│   └── Dockerfile
├── frontend/         # Vite React app (+ Dockerfile, nginx.conf)
├── docker-compose.yml
├── .github/workflows/ci.yml
├── rl/               # bản single-agent CŨ (prototype, không còn dùng)
└── README.md
```

> **Port:** backend RL Arena chạy ở **8001** (port 8000 bị backend Hakuryu Docker chiếm).
> Frontend gọi **same-origin** qua **Vite proxy** (`/api`, `/ws` → `127.0.0.1:8001`) nên
> không cần lo CORS hay IPv4/IPv6. Xem `frontend/vite.config.ts`.

---

## Chạy nhanh (2 terminal)

### 1) Backend — FastAPI (port 8001)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api.server:app --reload --host 127.0.0.1 --port 8001
```

Swagger: `http://localhost:8001/docs`.

> **Lưu ý IPv6:** trên máy này `localhost` resolve về `::1`. Vite proxy gọi backend bằng
> `127.0.0.1` (server-side) nên backend chỉ cần `--host 127.0.0.1`. Nếu bạn muốn browser gọi
> 8001 *trực tiếp* (không qua proxy), chạy `--host ::` và set `VITE_API_URL`.

### 2) Frontend — React + Vite

```bash
cd frontend
npm install
npm run dev                        # http://localhost:5173
```

Mở **http://localhost:5173**, bấm **Start** → xem 3 agent train trực tiếp.

### Hoặc: Docker Compose (1 lệnh)

```bash
docker compose up --build
# Frontend: http://localhost:8088   ·   API: http://localhost:8011/docs
```

---

## Tests, CI & types

```bash
# Backend tests
cd backend && source venv/bin/activate
python -m pytest tests/ -q          # unit tests (env logic)
python tests/smoke_test.py          # end-to-end smoke

# Sinh TS types từ OpenAPI (D15)
python scripts/export_openapi.py
cd ../frontend && npx openapi-typescript ../backend/openapi.json -o src/types/api.generated.ts

# TensorBoard (C9)
tensorboard --logdir backend/logs/tensorboard
```

GitHub Actions (`.github/workflows/ci.yml`) chạy backend tests + frontend typecheck/build mỗi push.

---

## Train & Evaluate bằng CLI

```bash
cd backend
source venv/bin/activate

# Train (lưu checkpoints + logs/training_metrics.json)
python -m src.training.train_solo

# Evaluate (tắt exploration, 100 episodes, in bảng so sánh)
python -m src.training.evaluate_solo --episodes 100
# thêm --render để xem ANSI từng bước
```

---

## Luật solo (tóm tắt)

- Tới **goal trước** → thắng (+50). Đối thủ **chết** → bạn thắng (+20).
- Hết `max_steps` → so **tổng reward**; bằng nhau → **draw**.
- **Collision**: cùng vào 1 ô hoặc swap vị trí → cả hai đứng yên, mỗi bên **−2**.
- Nhặt reward **+10**, vào danger **−10**, mỗi step **−1**, tiến gần goal **+1** / xa **−1**.

## Observation

- **PPO**: flattened one-hot grid, 7 channels (empty/wall/reward/danger/goal/self/opponent)
  → `10×10×7 = 700` chiều.
- **Dyna-Q**: tuple rời rạc 16 phần tử (vị trí self/opponent, delta tới goal & reward đã clip
  ±5, hazard wall/danger 4 hướng) — gọn để Q-table không bùng nổ.

---

## Kiến trúc realtime

```
Browser (React)  ──poll GET /api/state mỗi 500ms──►  FastAPI
     ▲                                                  │
     │  Start/Pause/Reset/Evaluate (POST)               ▼
     └──────────────────────────────────────  TrainingRunner (background thread)
                                                    │ (threading.Lock)
                                                    ▼
                                            SoloTrainer.step_once()
                                            ├─ PPOAgent  (PyTorch)
                                            ├─ DynaQAgent (Q-table + planning)
                                            └─ CompetitiveGridEnv
```

Training chạy trong **background thread**, API không bị block; mọi truy cập agent/env qua
một `Lock` chung nên hai thread không bao giờ chạm tensor / Q-table đồng thời.

Chi tiết: [`backend/README.md`](backend/README.md) · [`FRONTEND.md`](FRONTEND.md).

## Mở rộng sang Bomberman

Env đã cell-code-driven (movement/hazard/render). Để thêm bomb/box/explosion: thêm code ô
mới + nhánh trong `_enter_cell` / một `_tick_bombs`, mở rộng action `PLACE_BOMB`, và thêm
channel vào `encode_ppo_obs`. Frontend chỉ cần thêm style cho code ô mới trong
`EnvironmentGrid.tsx`.
