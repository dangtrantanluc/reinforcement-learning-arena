// ─────────────────────────────────────────────────────────────
// Types mirroring the FastAPI backend (GET /api/state).
// ─────────────────────────────────────────────────────────────

/** Integer cell codes from the backend grid (with agents overlaid). */
export const CELL = {
  EMPTY: 0,
  WALL: 1,
  REWARD: 2,
  DANGER: 3,
  GOAL: 4,
  PPO_AGENT: 5,
  DYNAQ_AGENT: 6,
} as const;

export type ActionName = 'UP' | 'DOWN' | 'LEFT' | 'RIGHT' | 'STAY';
export const ACTION_NAMES: ActionName[] = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'];

export type AgentId = 'ppo' | 'dynaq';
export type Winner = 'ppo' | 'dynaq' | 'draw' | null;

export interface PPOState {
  position: [number, number];
  episode_reward: number;
  total_wins: number;
  last_action: ActionName;
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
  epsilon: number;
  q_table_size: number;
  planning_steps: number;
  q_values: Record<ActionName, number>;
}

export interface MetricsState {
  ppo_win_rate: number;
  dynaq_win_rate: number;
  draw_rate: number;
  ppo_avg_reward: number;
  dynaq_avg_reward: number;
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
