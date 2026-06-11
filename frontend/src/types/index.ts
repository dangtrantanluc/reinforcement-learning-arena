// ─────────────────────────────────────────────────────────────
// Types mirroring the FastAPI backend (GET /api/state).
// ─────────────────────────────────────────────────────────────

/** Integer cell codes from the backend grid (with agents/bombs overlaid). */
export const CELL = {
  EMPTY: 0,
  WALL: 1,
  REWARD: 2,
  DANGER: 3,
  GOAL: 4,
  PPO_AGENT: 5,
  DYNAQ_AGENT: 6,
  BOX: 7,
  BOMB: 8,
  EXPLOSION: 9,
  DQN_AGENT: 10,
} as const;

export type ActionName = 'UP' | 'DOWN' | 'LEFT' | 'RIGHT' | 'STAY' | 'BOMB';
export const ACTION_NAMES: ActionName[] = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY', 'BOMB'];

export type AgentId = 'ppo' | 'dynaq' | 'dqn';
export type Winner = 'ppo' | 'dynaq' | 'dqn' | 'draw' | null;

/** Agent → cell code, label, colour token. */
export const AGENT_META: Record<AgentId, { code: number; label: string; color: string; soft: string; name: string }> = {
  ppo: { code: 5, label: 'P', color: 'ppo', soft: 'ppo-soft', name: 'PPO' },
  dynaq: { code: 6, label: 'D', color: 'dynaq', soft: 'dynaq-soft', name: 'Dyna-Q' },
  dqn: { code: 10, label: 'Q', color: 'dqn-c', soft: 'dqn-soft', name: 'DQN' },
};

export interface PPOState {
  position: [number, number];
  episode_reward: number;
  total_wins: number;
  last_action: ActionName;
  alive: boolean;
  action_probs: Record<ActionName, number>;
  policy_loss: number;
  value_loss: number;
  entropy: number;
}

export interface DynaQState {
  position: [number, number];
  episode_reward: number;
  total_wins: number;
  last_action: ActionName;
  alive: boolean;
  epsilon: number;
  q_table_size: number;
  planning_steps: number;
  q_values: Record<ActionName, number>;
}

export interface DQNState {
  position: [number, number];
  episode_reward: number;
  total_wins: number;
  last_action: ActionName;
  alive: boolean;
  epsilon: number;
  loss: number;
  buffer: number;
  q_values: Record<ActionName, number>;
}

export interface MetricsState {
  ppo_win_rate: number;
  dynaq_win_rate: number;
  draw_rate: number;
  ppo_avg_reward: number;
  dynaq_avg_reward: number;
  dqn_win_rate?: number;
  dqn_avg_reward?: number;
}

/** One per-episode history record (for charts). */
export interface HistoryPoint {
  episode: number;
  ppo_reward: number;
  dynaq_reward: number;
  winner: string;
  ppo_win_rate: number;
  dynaq_win_rate: number;
  draw_rate: number;
  episode_length: number;
  ppo_avg_reward: number;
  dynaq_avg_reward: number;
  dynaq_epsilon: number;
  dynaq_q_table_size: number;
  ppo_policy_loss: number;
  ppo_value_loss: number;
  ppo_entropy: number;
}

/** Full GET /api/state payload. */
export interface ArenaState {
  running: boolean;
  episode: number;
  step: number;
  grid: number[][];
  winner: Winner;
  ppo: PPOState;
  dynaq: DynaQState;
  metrics: MetricsState;
  history: HistoryPoint[];
  logs: string[];
}

export interface EvaluateResult {
  episodes: number;
  ppo_win_rate: number;
  dynaq_win_rate: number;
  draw_rate: number;
  ppo_avg_reward: number;
  dynaq_avg_reward: number;
  avg_episode_length: number;
  ppo_danger_hits: number;
  dynaq_danger_hits: number;
}

// ── A1: per-step replay frames (N-agent) ──
export interface FrameEvent {
  agent: AgentId | 'both';
  type: 'reward' | 'danger' | 'collision' | 'place' | 'explode' | 'break_box' | 'kill' | string;
  pos?: [number, number];
}

export interface Frame {
  seq: number;
  kind: 'reset' | 'step' | 'end';
  episode: number;
  step: number;
  grid: number[][];
  positions: Record<string, [number, number]>;
  alive: Record<string, boolean>;
  actions: Record<string, ActionName>;
  rewards: Record<string, number>;
  winner: Winner;
  events: FrameEvent[];
}

/** Payload of GET /api/frames and the WebSocket stream. */
export interface FramesPayload {
  running: boolean;
  frames: Frame[];
  ppo: PPOState;
  dynaq: DynaQState;
  dqn?: DQNState;
  metrics: MetricsState;
  history: HistoryPoint[];
  logs: string[];
  last_frame_seq: number;
}

/** Per-agent normalised visit heatmaps (0..1). */
export type Heatmaps = Record<string, number[][]>;

// ── B7: match replays ──
export interface ReplaySummary {
  episode: number;
  winner: string;
  reason: string | null;
  length: number;
  rewards: Record<string, number>;
}

export interface ReplayFull extends ReplaySummary {
  frames: Frame[];
}

// ── C11: editable hyperparameters ──
export interface HyperConfig {
  ppo: { lr: number; gamma: number; gae_lambda: number; clip_eps: number; entropy_coef: number; rollout_steps: number };
  dynaq: { alpha: number; gamma: number; epsilon_decay: number; planning_steps: number };
  dqn: { lr: number; gamma: number; epsilon_decay: number; batch_size: number; target_update: number };
  env: { max_steps: number; bomb_fuse: number; bomb_range: number; enable_bombs: boolean };
  curriculum: { enabled: boolean };
  train: { enable_dqn: boolean };
}
