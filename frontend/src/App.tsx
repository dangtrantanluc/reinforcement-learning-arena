import { useEffect, useRef, useState } from 'react';
import Sidebar, { type NavKey } from './components/Sidebar';
import Header from './components/Header';
import EnvironmentGrid from './components/EnvironmentGrid';
import AgentStatsPanel from './components/AgentStatsPanel';
import ComparisonPanel from './components/ComparisonPanel';
import ActionPanel from './components/ActionPanel';
import TrainingCharts from './components/TrainingCharts';
import TrainingLog from './components/TrainingLog';
import SettingsPanel from './components/SettingsPanel';
import OfflineNotice from './components/OfflineNotice';
import { useArenaState } from './hooks/useArenaState';
import { CELL, type Winner } from './types';

type Mood = 'idle' | 'moving' | 'reward' | 'danger' | 'winner';

/** Derive an agent's mood from the cell it stands on + win state. */
function moodFor(
  prevCellCode: number | undefined,
  running: boolean,
  isWinner: boolean,
): Mood {
  if (isWinner) return 'winner';
  if (!running) return 'idle';
  if (prevCellCode === CELL.REWARD) return 'reward';
  if (prevCellCode === CELL.DANGER) return 'danger';
  return 'moving';
}

function Section({
  id,
  title,
  innerRef,
  children,
}: {
  id: string;
  title?: string;
  innerRef: (el: HTMLElement | null) => void;
  children: React.ReactNode;
}) {
  return (
    <section id={id} ref={innerRef} className="scroll-mt-4">
      {title && <h2 className="mb-3 text-sm font-bold uppercase tracking-wider text-sub">{title}</h2>}
      {children}
    </section>
  );
}

export default function App() {
  const { state, connected, busy, evalResult, start, pause, reset, evaluate } = useArenaState();
  const [active, setActive] = useState<NavKey>('arena');

  const sectionRefs = useRef<Record<string, HTMLElement | null>>({});
  const register = (key: string) => (el: HTMLElement | null) => {
    sectionRefs.current[key] = el;
  };
  const scrollRoot = useRef<HTMLDivElement | null>(null);

  const handleNav = (key: NavKey) => {
    setActive(key);
    sectionRefs.current[key]?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  useEffect(() => {
    const root = scrollRoot.current;
    if (!root) return;
    const obs = new IntersectionObserver(
      (entries) => {
        const v = entries.filter((e) => e.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (v) setActive(v.target.id as NavKey);
      },
      { root, threshold: 0.4 },
    );
    Object.values(sectionRefs.current).forEach((el) => el && obs.observe(el));
    return () => obs.disconnect();
  }, [state !== null]);

  const running = state?.running ?? false;

  // Mood derivation: peek at the cell each agent sits on in the integer grid.
  const grid = state?.grid;
  const cellUnder = (pos?: [number, number]): number | undefined =>
    grid && pos ? grid[pos[0]]?.[pos[1]] : undefined;

  const winner: Winner = state?.winner ?? null;
  const ppoMood = moodFor(cellUnder(state?.ppo.position), running, winner === 'ppo');
  const dynaqMood = moodFor(cellUnder(state?.dynaq.position), running, winner === 'dynaq');

  return (
    <div className="flex h-screen overflow-hidden bg-bg">
      <Sidebar active={active} onSelect={handleNav} />

      <div className="flex min-w-0 flex-1 flex-col">
        <Header
          running={running}
          connected={connected}
          busy={busy}
          onStart={start}
          onPause={pause}
          onReset={reset}
          onEvaluate={evaluate}
        />

        <main ref={scrollRoot} className="scroll-thin flex-1 overflow-y-auto px-6 py-6">
          <div className="mx-auto flex max-w-[1500px] flex-col gap-8">
            {!connected && <OfflineNotice />}

            {state && (
              <>
                {/* ARENA: PPO panel | grid | Dyna-Q panel */}
                <Section id="arena" innerRef={register('arena')}>
                  <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)_minmax(0,1fr)]">
                    <AgentStatsPanel agent="ppo" ppo={state.ppo} winRate={state.metrics.ppo_win_rate} />
                    <div className="flex flex-col gap-4">
                      <EnvironmentGrid
                        grid={state.grid}
                        ppoPos={state.ppo.position}
                        dynaqPos={state.dynaq.position}
                        ppoMood={ppoMood}
                        dynaqMood={dynaqMood}
                        winner={winner}
                      />
                      <ActionPanel ppo={state.ppo} dynaq={state.dynaq} />
                    </div>
                    <AgentStatsPanel agent="dynaq" dynaq={state.dynaq} winRate={state.metrics.dynaq_win_rate} />
                  </div>
                </Section>

                {/* TRAINING: comparison + log */}
                <Section id="training" title="Training" innerRef={register('training')}>
                  <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
                    <ComparisonPanel
                      metrics={state.metrics}
                      ppoWins={state.ppo.total_wins}
                      dynaqWins={state.dynaq.total_wins}
                      evalResult={evalResult}
                    />
                    <TrainingLog logs={state.logs} />
                  </div>
                </Section>

                {/* METRICS: charts */}
                <Section id="metrics" title="Comparison Charts" innerRef={register('metrics')}>
                  <TrainingCharts history={state.history} />
                </Section>

                {/* SETTINGS */}
                <Section id="settings" title="Settings" innerRef={register('settings')}>
                  <SettingsPanel
                    epsilon={state.dynaq.epsilon}
                    planningSteps={state.dynaq.planning_steps}
                    qTableSize={state.dynaq.q_table_size}
                  />
                </Section>
              </>
            )}

            <footer className="pb-2 pt-4 text-center text-xs text-sub">
              RL Arena · PPO (neural) vs Dyna-Q (tabular) · live competitive training over FastAPI
            </footer>
          </div>
        </main>
      </div>
    </div>
  );
}
