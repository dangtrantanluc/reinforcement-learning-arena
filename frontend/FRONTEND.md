# Frontend — RL Arena UI

React + Vite + TypeScript + TailwindCSS + Framer Motion + Recharts. Poll backend
FastAPI (`/api/state`) mỗi 500ms và hiển thị realtime hai agent PPO vs Dyna-Q.

## Chạy

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # typecheck + production build → dist/
npm run preview    # serve bản build
```

Cấu hình URL backend qua `.env` (mặc định `http://localhost:8001`):

```bash
cp .env.example .env
# sửa VITE_API_URL nếu backend chạy port khác
```

## Cấu trúc

```
src/
├── main.tsx · App.tsx · index.css · vite-env.d.ts
├── api/
│   └── client.ts          # getState / start / pause / reset / evaluate
├── hooks/
│   └── useArenaState.ts    # poll 500ms + control actions + connected flag
├── types/
│   └── index.ts            # types khớp payload backend
└── components/
    ├── Sidebar.tsx          # Arena / Training / Metrics / Settings
    ├── Header.tsx           # title + status badge + ControlPanel
    ├── ControlPanel.tsx     # Start / Pause / Reset / Evaluate
    ├── EnvironmentGrid.tsx  # grid 10×10 + 2 agent (Framer Motion)
    ├── AgentCharacter.tsx   # robot PPO (blue/P) & Dyna-Q (purple/D)
    ├── AgentStatsPanel.tsx  # stats + action bars (probs / Q-values)
    ├── ComparisonPanel.tsx  # head-to-head + kết quả evaluate
    ├── ActionPanel.tsx      # "Current Decision" của 2 agent
    ├── TrainingCharts.tsx   # Recharts: reward / win-rate / length / loss & ε
    ├── TrainingLog.tsx      # console realtime
    ├── SettingsPanel.tsx    # hyperparameters (read-only)
    └── OfflineNotice.tsx    # hướng dẫn khi backend offline
```

## Mapping màu

| | Màu |
|---|-----|
| PPO | xanh dương `#2563eb` (label **P**) |
| Dyna-Q | tím `#7c3aed` (label **D**) |
| Reward / Danger / Goal | xanh lá / đỏ / vàng |

## Animation

- Agent tween mượt giữa các ô bằng Framer Motion spring.
- Mắt LED sáng xanh khi nhặt reward, rung + mắt amber khi vào danger.
- Agent thắng → glow vàng. Idle → animation thở + chớp mắt.

## Lưu ý realtime

Backend là source-of-truth duy nhất; frontend chỉ render `ArenaState` từ `/api/state`.
Vì poll 500ms còn backend step nhanh hơn, UI hiển thị "ảnh chụp" mới nhất — agent vẫn nhảy
mượt nhờ spring animation trên vị trí.
