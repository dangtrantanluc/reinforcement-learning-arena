import { motion } from 'framer-motion';

export type NavKey = 'arena' | 'training' | 'metrics' | 'settings';

const Icon = ({ d }: { d: string }) => (
  <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" fill="none" stroke="currentColor"
    strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={d} /></svg>
);

const NAV: { key: NavKey; label: string; icon: JSX.Element }[] = [
  { key: 'arena', label: 'Arena', icon: <Icon d="M3 3h18v18H3zM3 9h18M9 3v18" /> },
  { key: 'training', label: 'Training', icon: <Icon d="M3 17l6-6 4 4 8-8M14 7h7v7" /> },
  { key: 'metrics', label: 'Metrics', icon: <Icon d="M4 19V5m0 14l5-5 4 4 7-9M4 19h16" /> },
  { key: 'settings', label: 'Settings', icon: <Icon d="M12 8a4 4 0 100 8 4 4 0 000-8zM2 12h2m16 0h2M12 2v2m0 16v2" /> },
];

export default function Sidebar({
  active,
  onSelect,
}: {
  active: NavKey;
  onSelect: (k: NavKey) => void;
}) {
  return (
    <aside className="flex w-[220px] shrink-0 flex-col border-r border-line bg-primary text-white/90">
      <div className="flex items-center gap-3 px-5 py-5">
        <div className="grid h-9 w-9 place-items-center rounded-lg bg-white/10 ring-1 ring-white/15">
          {/* dual-bot mark */}
          <svg viewBox="0 0 32 32" className="h-5 w-5">
            <rect x="3" y="11" width="11" height="10" rx="3" fill="#60a5fa" />
            <rect x="18" y="11" width="11" height="10" rx="3" fill="#a78bfa" />
          </svg>
        </div>
        <div className="leading-tight">
          <p className="text-sm font-bold tracking-tight text-white">RL Arena</p>
          <p className="text-[11px] text-white/50">PPO vs Dyna-Q</p>
        </div>
      </div>

      <nav className="mt-2 flex flex-col gap-1 px-3">
        {NAV.map((item) => {
          const isActive = item.key === active;
          return (
            <button
              key={item.key}
              onClick={() => onSelect(item.key)}
              className={`relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                isActive ? 'text-white' : 'text-white/60 hover:bg-white/5 hover:text-white/90'
              }`}
            >
              {isActive && (
                <motion.span
                  layoutId="nav-active"
                  className="absolute inset-0 rounded-lg bg-white/12 ring-1 ring-white/15"
                  transition={{ type: 'spring', stiffness: 400, damping: 32 }}
                />
              )}
              <span className="relative">{item.icon}</span>
              <span className="relative">{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="mt-auto px-5 py-4">
        <div className="rounded-lg bg-white/5 p-3 ring-1 ring-white/10">
          <p className="text-[11px] font-medium text-white/70">Matchup</p>
          <div className="mt-1.5 flex items-center justify-between text-xs">
            <span className="font-semibold text-[#93c5fd]">PPO</span>
            <span className="text-white/40">vs</span>
            <span className="font-semibold text-[#c4b5fd]">Dyna-Q</span>
          </div>
          <p className="mt-1 text-[10px] text-white/40">Neural policy vs tabular planning</p>
        </div>
      </div>
    </aside>
  );
}
