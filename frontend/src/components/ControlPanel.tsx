import { useState } from 'react';

interface Props {
  running: boolean;
  busy?: boolean;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
  onNewMap: () => void;
  onEvaluate: () => void;
  onSpeed: (delay: number) => void;
  onSave: () => void;
  onLoad: () => void;
}

const Play = () => (<svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>);
const Pause = () => (<svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z" /></svg>);
const Reset = () => (<svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 109-9 9 9 0 00-7 3.3M3 4v4h4" /></svg>);
const Dice = () => (<svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="4" width="16" height="16" rx="3" /><circle cx="9" cy="9" r="1" fill="currentColor" /><circle cx="15" cy="15" r="1" fill="currentColor" /></svg>);
const Eval = () => (<svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 17V9m4 8V5m4 12v-6M4 21h16" /></svg>);

// Speed presets: label → backend step delay (seconds). Smaller = faster training.
const SPEEDS: { label: string; delay: number }[] = [
  { label: '0.25×', delay: 0.24 },
  { label: '1×', delay: 0.06 },
  { label: '4×', delay: 0.015 },
  { label: 'Max', delay: 0 },
];

export default function ControlPanel(p: Props) {
  const [speedIdx, setSpeedIdx] = useState(1);
  const pickSpeed = (i: number) => {
    setSpeedIdx(i);
    p.onSpeed(SPEEDS[i].delay);
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <button className="btn-primary" onClick={p.onStart} disabled={p.running}>
        <Play /> Start
      </button>
      <button className="btn-ghost" onClick={p.onPause} disabled={!p.running}>
        <Pause /> Pause
      </button>
      <button className="btn-ghost" onClick={p.onReset}>
        <Reset /> Reset
      </button>
      <button className="btn-ghost" onClick={p.onNewMap}>
        <Dice /> New Map
      </button>

      <div className="inline-flex overflow-hidden rounded-lg border border-line" title="Speed">
        {SPEEDS.map((s, i) => (
          <button
            key={s.label}
            onClick={() => pickSpeed(i)}
            className={`px-2.5 py-2 text-xs font-medium transition-colors ${
              i === speedIdx ? 'bg-primary text-white' : 'bg-surface text-sub hover:bg-bg'
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>

      <button className="btn-ghost" onClick={p.onLoad} title="Load saved weights">Load</button>
      <button className="btn-ghost" onClick={p.onSave} title="Save weights">Save</button>
      <button className="btn-ghost" onClick={p.onEvaluate} disabled={p.busy}>
        <Eval /> {p.busy ? 'Evaluating…' : 'Evaluate'}
      </button>
    </div>
  );
}
