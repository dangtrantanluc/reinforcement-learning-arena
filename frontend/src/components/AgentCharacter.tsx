import { motion } from 'framer-motion';
import type { AgentId } from '../types';

interface Props {
  agent: AgentId;
  /** 'idle' | 'moving' | 'reward' | 'danger' | 'winner' */
  mood?: 'idle' | 'moving' | 'reward' | 'danger' | 'winner';
  size?: number;
}

// PPO = blue, Dyna-Q = purple, DQN = teal. Label P / D / Q on the body.
const THEME: Record<AgentId, { body: string; label: string }> = {
  ppo: { body: '#2563eb', label: 'P' },
  dynaq: { body: '#7c3aed', label: 'D' },
  dqn: { body: '#0d9488', label: 'Q' },
};

/**
 * A tiny robot for each agent. Eyes glow green on reward, amber + shake on
 * danger, gold pulse when it wins; gentle breathing when idle.
 */
export default function AgentCharacter({ agent, mood = 'idle', size = 28 }: Props) {
  const theme = THEME[agent];
  const eyeColor =
    mood === 'danger' ? '#f59e0b' : mood === 'reward' ? '#22c55e' : '#bfdbfe';

  return (
    <motion.div
      style={{ width: size, height: size }}
      className="relative grid place-items-center"
      animate={
        mood === 'danger'
          ? { x: [0, -2, 2, -2, 2, 0] }
          : mood === 'idle'
            ? { y: [0, -1.2, 0] }
            : { x: 0, y: 0 }
      }
      transition={
        mood === 'danger'
          ? { duration: 0.4, repeat: Infinity }
          : mood === 'idle'
            ? { duration: 2.4, repeat: Infinity, ease: 'easeInOut' }
            : { duration: 0.2 }
      }
    >
      {(mood === 'reward' || mood === 'winner') && (
        <motion.span
          className="absolute inset-0 rounded-lg blur-[6px]"
          style={{ background: mood === 'winner' ? 'rgba(234,179,8,0.55)' : 'rgba(34,197,94,0.4)' }}
          animate={{ opacity: [0.2, 0.8, 0.2], scale: [0.9, 1.15, 0.9] }}
          transition={{ duration: 0.9, repeat: Infinity }}
        />
      )}

      <svg viewBox="0 0 32 32" width={size} height={size} className="relative drop-shadow-sm">
        <line x1="16" y1="3" x2="16" y2="7" stroke={theme.body} strokeWidth="1.6" />
        <circle cx="16" cy="3" r="2" fill={mood === 'winner' ? '#eab308' : theme.body} />
        <rect x="6" y="7" width="20" height="17" rx="6" fill={theme.body} />
        {/* eyes */}
        <motion.g
          animate={mood === 'idle' ? { scaleY: [1, 1, 0.1, 1] } : { scaleY: 1 }}
          transition={{ duration: 3, repeat: Infinity, times: [0, 0.85, 0.9, 1] }}
          style={{ transformOrigin: '16px 14px' }}
        >
          <circle cx="12" cy="14" r="2.2" fill={eyeColor} />
          <circle cx="20" cy="14" r="2.2" fill={eyeColor} />
        </motion.g>
        {/* label badge */}
        <text
          x="16"
          y="22"
          textAnchor="middle"
          fontSize="7"
          fontWeight="700"
          fill="#fff"
          fontFamily="monospace"
        >
          {theme.label}
        </text>
      </svg>
    </motion.div>
  );
}
