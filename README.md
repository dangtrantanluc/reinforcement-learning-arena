# RL Arena — PPO vs Dyna-Q Solo

Hai agent reinforcement learning **cạnh tranh trực tiếp trên cùng một map** grid-world
(hướng tới Bomberman). Cả hai tự implement từ đầu bằng PyTorch / NumPy — **không dùng
stable-baselines3**.

| | Thuật toán | Cách học |
|---|-----------|----------|
| 🔵 **PPO** | Proximal Policy Optimization | Neural Actor-Critic, on-policy, GAE + clipped objective |
| 🟣 **Dyna-Q** | Tabular Q-learning + planning | Q-table + world model, model-based planning |

Frontend React hiển thị **realtime** cả hai agent đua nhau, action probabilities / Q-values,
biểu đồ so sánh, và training log — poll backend FastAPI mỗi 500ms.

```
agent-rf/
├── backend/          # Python: env + agents + training + FastAPI
│   └── src/
│       ├── env/      # CompetitiveGridEnv, map_generator, reward, state_encoder
│       ├── agents/   # ppo/  +  dynaq/   (+ base_agent)
│       ├── training/ # train_solo, evaluate_solo, metrics, checkpoint
│       └── api/      # server, schemas, training_runner
├── frontend/         # Vite React app (src/, package.json, …)
├── rl/               # bản single-agent CŨ (prototype, không còn dùng)
└── README.md         # file này
```

> **Cấu trúc:** `backend/` (Python) và `frontend/` (React) là hai folder con của `agent-rf/`.
> Frontend cũ (PPO single-agent simulator) đã được refactor thành arena này.
>
> **Port:** backend RL Arena chạy ở **8001** vì port 8000 đang được backend Hakuryu (Docker)
> dùng. Frontend mặc định trỏ `localhost:8001` (xem `frontend/.env.example`).

---

## Chạy nhanh (2 terminal)

### 1) Backend — FastAPI (port 8001)

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api.server:app --reload --host :: --port 8001
```

Backend lên ở `http://localhost:8001` — thử `http://localhost:8001/docs` để xem Swagger.

> Muốn port khác? Chạy `--port <N>` rồi đặt `VITE_API_URL=http://localhost:<N>` trong
> `frontend/.env` (xem `frontend/.env.example`).

### 2) Frontend — React + Vite

```bash
cd frontend
npm install
npm run dev                        # http://localhost:5173
```

Mở `http://localhost:5173`, bấm **Start** ở header → xem hai agent train trực tiếp.
Nếu backend chưa chạy, UI hiện hướng dẫn khởi động (poll tự kết nối lại).

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
