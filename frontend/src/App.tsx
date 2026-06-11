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
import HeatmapPanel from './components/HeatmapPanel';
import ReplayBrowser from './components/ReplayBrowser';
import OfflineNotice from './components/OfflineNotice';
import { useArenaStream } from './hooks/useArenaStream';
import { useTheme } from './hooks/useTheme';
import type { Winner } from './types';

function Section({
  id, title, innerRef, children,
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
  const arena = useArenaStream();
  const {
    current, ppo, dynaq, dqn, metrics, history, logs, running, connected, queueLen,
    busy, evalResult, heatmaps,
    start, pause, reset, setSpeed, saveCheckpoint, loadCheckpoint, evaluate, fetchHeatmaps,
  } = arena;

  const { theme, toggle: toggleTheme } = useTheme();
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

  // Refresh heatmaps when the Metrics section is viewed.
  useEffect(() => {
    if (active === 'metrics') fetchHeatmaps();
  }, [active, fetchHeatmaps]);

  const ready = connected && current && ppo && dynaq && metrics;
  const winner: Winner = current?.winner ?? null;

  return (
    <div className="flex h-screen overflow-hidden bg-bg">
      <Sidebar active={active} onSelect={handleNav} />

      <div className="flex min-w-0 flex-1 flex-col">
        <Header
          running={running}
          connected={connected}
          busy={busy}
          episode={current?.episode ?? 0}
          step={current?.step ?? 0}
          queueLen={queueLen}
          theme={theme}
          onToggleTheme={toggleTheme}
          onStart={start}
          onPause={pause}
          onReset={() => reset()}
          onNewMap={() => reset(Math.floor(Math.random() * 100000))}
          onEvaluate={evaluate}
          onSpeed={setSpeed}
          onSave={saveCheckpoint}
          onLoad={loadCheckpoint}
        />

        <main ref={scrollRoot} className="scroll-thin flex-1 overflow-y-auto px-6 py-6">
          <div className="mx-auto flex max-w-[1500px] flex-col gap-8">
            {!connected && <OfflineNotice />}

            {ready && (
              <>
                {/* ARENA */}
                <Section id="arena" innerRef={register('arena')}>
                  <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)_minmax(0,1fr)]">
                    <AgentStatsPanel agent="ppo" ppo={ppo!} winRate={metrics!.ppo_win_rate} />
                    <div className="flex flex-col gap-4">
                      <EnvironmentGrid frame={current!} winner={winner} />
                      <ActionPanel ppo={ppo!} dynaq={dynaq!} dqn={dqn} />
                    </div>
                    <AgentStatsPanel agent="dynaq" dynaq={dynaq!} winRate={metrics!.dynaq_win_rate} />
                  </div>
                  {dqn && (
                    <div className="mt-4">
                      <AgentStatsPanel agent="dqn" dqn={dqn} winRate={metrics!.dqn_win_rate ?? 0} />
                    </div>
                  )}
                </Section>

                {/* TRAINING */}
                <Section id="training" title="Training" innerRef={register('training')}>
                  <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
                    <ComparisonPanel
                      metrics={metrics!}
                      ppoWins={ppo!.total_wins}
                      dynaqWins={dynaq!.total_wins}
                      dqnWins={dqn?.total_wins}
                      evalResult={evalResult}
                    />
                    <TrainingLog logs={logs} />
                  </div>
                </Section>

                {/* METRICS */}
                <Section id="metrics" title="Comparison Charts" innerRef={register('metrics')}>
                  <div className="flex flex-col gap-4">
                    <TrainingCharts history={history} hasDqn={!!dqn} />
                    <HeatmapPanel heatmaps={heatmaps} onRefresh={fetchHeatmaps} />
                  </div>
                </Section>

                {/* REPLAYS */}
                <Section id="replays" title="Match Replays" innerRef={register('replays')}>
                  <ReplayBrowser />
                </Section>

                {/* SETTINGS */}
                <Section id="settings" title="Settings" innerRef={register('settings')}>
                  <SettingsPanel
                    epsilon={dynaq!.epsilon}
                    planningSteps={dynaq!.planning_steps}
                    qTableSize={dynaq!.q_table_size}
                  />
                </Section>
              </>
            )}

            <footer className="pb-2 pt-4 text-center text-xs text-sub">
              RL Arena · PPO vs Dyna-Q vs DQN · Bomberman · realtime step replay over WebSocket
            </footer>
          </div>
        </main>
      </div>
    </div>
  );
}
