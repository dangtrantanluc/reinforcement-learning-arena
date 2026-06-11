import { API_BASE } from '../api/client';

/** Shown when the backend isn't reachable — explains how to start it. */
export default function OfflineNotice() {
  return (
    <div className="card border-warning/30 bg-warning/5 p-5">
      <div className="flex items-start gap-3">
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-warning/15 text-warning">
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2"
            strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 9v4m0 4h.01M10.3 3.9L1.8 18a2 2 0 001.7 3h16.9a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z" />
          </svg>
        </div>
        <div>
          <p className="text-sm font-semibold text-ink">Backend offline</p>
          <p className="mt-1 text-xs text-sub">
            Can't reach the API at <span className="font-mono">{API_BASE || '/api (via Vite proxy → :8001)'}</span>. Start the backend with:
          </p>
          <pre className="mt-2 overflow-x-auto rounded-lg bg-[#1f2535] p-3 font-mono text-[12px] leading-relaxed text-white/80">
{`cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.server:app --reload --host :: --port 8001`}
          </pre>
          <p className="mt-2 text-[11px] text-sub">Polling every 500ms — this will connect automatically once the server is up.</p>
        </div>
      </div>
    </div>
  );
}
