interface Props {
  running: boolean;
  busy?: boolean;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
  onEvaluate: () => void;
}

const Play = () => (<svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>);
const Pause = () => (<svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z" /></svg>);
const Reset = () => (<svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 109-9 9 9 0 00-7 3.3M3 4v4h4" /></svg>);
const Eval = () => (<svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 17V9m4 8V5m4 12v-6M4 21h16" /></svg>);

export default function ControlPanel({ running, busy, onStart, onPause, onReset, onEvaluate }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button className="btn-primary" onClick={onStart} disabled={running}>
        <Play /> Start
      </button>
      <button className="btn-ghost" onClick={onPause} disabled={!running}>
        <Pause /> Pause
      </button>
      <button className="btn-ghost" onClick={onReset}>
        <Reset /> Reset
      </button>
      <button className="btn-ghost" onClick={onEvaluate} disabled={busy}>
        <Eval /> {busy ? 'Evaluating…' : 'Evaluate'}
      </button>
    </div>
  );
}
