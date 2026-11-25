// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// The praxis-1 page: pick a rig and a sim task, choose seeds, run the episodes,
// then watch every episode (the sim cameras next to the video the model imagined)
// with a pass/fail badge per seed. Drives /generate -> /status -> /result and
// /result/<id>/summary on the cartridge's endpoint.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, Bot, CheckCircle2, Download, Loader2, Play, XCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { getConfig } from '@/services/config';
import { useApiToken } from '@/services/apiToken';
import { PRAXIS_GROUPS, PRAXIS_MAX_SEEDS, PRAXIS_TASKS, PraxisRig } from '@/data/praxis-tasks';

type ServerStatus = 'unknown' | 'loading' | 'ready' | 'offline';

interface Episode { seed: number; success: boolean; video?: string }
interface Summary { rig: string; task: string; prompt: string | null; episodes: number; successes: number; per_episode: Episode[] }

const RIGS: { id: PraxisRig; label: string; sub: string }[] = [
  { id: 'droid', label: 'Single arm', sub: 'Franka · praxis-1-droid' },
  { id: 'yam', label: 'Two arms', sub: 'YAM · praxis-1-yam' },
];

const Panel = ({ title, right, children }: { title?: string; right?: React.ReactNode; children: React.ReactNode }) => (
  <div className="rounded-xl bg-game-surface/70 backdrop-blur-sm border border-game-border p-5 md:p-6">
    {(title || right) && (
      <div className="flex items-center justify-between mb-4">
        {title && <h2 className="text-[11px] font-bold tracking-[0.14em] uppercase text-white/40">{title}</h2>}
        {right}
      </div>
    )}
    {children}
  </div>
);

const StatusPill = ({ status }: { status: ServerStatus }) => {
  const map: Record<ServerStatus, { text: string; cls: string; Icon: typeof Loader2 }> = {
    unknown: { text: 'checking…', cls: 'text-white/40 border-white/10', Icon: Loader2 },
    loading: { text: 'model loading', cls: 'text-amber-300 border-amber-300/40', Icon: Loader2 },
    ready: { text: 'ready', cls: 'text-emerald-300 border-emerald-400/40', Icon: CheckCircle2 },
    offline: { text: 'offline', cls: 'text-red-300 border-red-400/40', Icon: XCircle },
  };
  const { text, cls, Icon } = map[status];
  return (
    <span className={`inline-flex items-center gap-2 text-[11px] font-medium tracking-wider uppercase px-3 py-1 rounded-full border ${cls}`}>
      <Icon className={`h-3 w-3 ${status === 'unknown' || status === 'loading' ? 'animate-spin' : ''}`} />
      {text}
    </span>
  );
};

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;

export const PraxisRunner = ({ endpointUrl }: { endpointUrl?: string }) => {
  const apiUrl = endpointUrl || getConfig().lingbotApiUrl;
  const apiToken = useApiToken();
  const authHeaders = useMemo<Record<string, string>>(() => {
    const h: Record<string, string> = {};
    if (apiToken) h.Authorization = `Bearer ${apiToken}`;
    return h;
  }, [apiToken]);

  const [serverStatus, setServerStatus] = useState<ServerStatus>('unknown');
  const [rig, setRig] = useState<PraxisRig>('droid');
  const [taskId, setTaskId] = useState('lift_cube');
  const [prompt, setPrompt] = useState('');
  const [seeds, setSeeds] = useState<number[]>([0, 1]);

  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  const tickRef = useRef<number | null>(null);

  const task = PRAXIS_TASKS.find(t => t.id === taskId)!;
  const tasks = PRAXIS_TASKS.filter(t => t.rig === rig);

  useEffect(() => {
    const probe = () => fetch(`${apiUrl}/health`, { cache: 'no-store', headers: authHeaders })
      .then(r => r.json())
      .then(d => setServerStatus(d.model_loaded ? 'ready' : 'loading'))
      .catch(() => setServerStatus('offline'));
    probe();
    const h = window.setInterval(probe, 15_000);
    return () => window.clearInterval(h);
  }, [apiUrl, authHeaders]);

  const stopTimers = () => {
    if (pollRef.current) window.clearInterval(pollRef.current);
    if (tickRef.current) window.clearInterval(tickRef.current);
    pollRef.current = tickRef.current = null;
  };
  useEffect(() => stopTimers, []);

  const pickRig = (r: PraxisRig) => {
    setRig(r);
    setTaskId(PRAXIS_TASKS.find(t => t.rig === r)!.id);
  };
  const toggleSeed = (s: number) =>
    setSeeds(cur => (cur.includes(s) ? (cur.length > 1 ? cur.filter(x => x !== s) : cur) : [...cur, s].sort((a, b) => a - b)));

  const poll = useCallback(async (jobId: string) => {
    try {
      const data = await (await fetch(`${apiUrl}/status/${jobId}`, { headers: authHeaders })).json();
      if (data.status === 'complete') {
        stopTimers();
        const blob = await (await fetch(`${apiUrl}/result/${jobId}`, { headers: authHeaders })).blob();
        setVideoUrl(URL.createObjectURL(blob));
        const s = await fetch(`${apiUrl}/result/${jobId}/summary`, { headers: authHeaders });
        if (s.ok) setSummary(await s.json());
        setRunning(false);
      } else if (data.status === 'failed') {
        stopTimers();
        setError(data.error || 'The rollout failed');
        setRunning(false);
      }
    } catch { /* keep polling */ }
  }, [apiUrl, authHeaders]);

  const run = useCallback(async () => {
    setRunning(true); setError(null); setVideoUrl(null); setSummary(null); setElapsed(0);
    try {
      const form = new FormData();
      form.append('rig', rig);
      form.append('task', taskId);
      form.append('seeds', seeds.join(','));
      if (prompt.trim()) form.append('prompt', prompt.trim());
      const resp = await fetch(`${apiUrl}/generate`, { method: 'POST', body: form, headers: authHeaders });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || 'Request failed');
      }
      const { job_id } = await resp.json();
      pollRef.current = window.setInterval(() => poll(job_id), 3000);
      tickRef.current = window.setInterval(() => setElapsed(s => s + 1), 1000);
    } catch (e: any) {
      setError(e.message || 'Request failed');
      setRunning(false);
    }
  }, [apiUrl, authHeaders, rig, taskId, seeds, prompt, poll]);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="text-[11px] font-bold tracking-[0.18em] uppercase text-emerald-300/80">Runway · world action model</div>
          <h1 className="text-3xl md:text-4xl font-extrabold text-white mt-1">Praxis-1</h1>
          <p className="text-white/55 text-sm mt-2 max-w-[720px]">
            Give it a task in plain words. From the robot's cameras it imagines the next few seconds as video,
            and turns that imagined future into arm movements — here in closed loop in a MuJoCo simulation.
          </p>
        </div>
        <StatusPill status={serverStatus} />
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
        <div className="space-y-5">
          <Panel title="1 · Robot">
            <div className="grid grid-cols-2 gap-3">
              {RIGS.map(r => (
                <button key={r.id} type="button" onClick={() => pickRig(r.id)} disabled={running}
                  className={`text-left rounded-lg border px-4 py-3 transition ${rig === r.id ? 'border-game-accent bg-game-accent/10' : 'border-game-border hover:border-white/25'}`}>
                  <div className="flex items-center gap-2 text-white font-semibold"><Bot className="h-4 w-4" />{r.label}</div>
                  <div className="text-[11px] text-white/40 mt-1">{r.sub}</div>
                </button>
              ))}
            </div>
          </Panel>

          <Panel title="2 · Task">
            <div className="space-y-4 max-h-[420px] overflow-y-auto pr-1">
              {PRAXIS_GROUPS.map(g => {
                const items = tasks.filter(t => t.group === g);
                if (!items.length) return null;
                return (
                  <div key={g}>
                    <div className="text-[10px] font-bold tracking-[0.16em] uppercase text-white/30 mb-2">{g}</div>
                    <div className="grid sm:grid-cols-2 gap-2">
                      {items.map(t => (
                        <button key={t.id} type="button" onClick={() => setTaskId(t.id)} disabled={running} data-task={t.id}
                          className={`text-left rounded-lg border px-3 py-2.5 transition ${taskId === t.id ? 'border-game-accent bg-game-accent/10' : 'border-game-border hover:border-white/25'}`}>
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-sm text-white font-semibold">{t.label}</span>
                            {t.note && <span className="text-[10px] text-white/35 whitespace-nowrap">{t.note}</span>}
                          </div>
                          <div className="text-[12px] text-white/50 mt-1 leading-snug">“{t.prompt}”</div>
                        </button>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </Panel>

          <Panel title="3 · Instruction and seeds">
            <label className="block">
              <span className="block text-[11px] font-bold tracking-[0.14em] uppercase text-white/50 mb-2">
                Instruction<span className="ml-2 text-white/25 tracking-wide normal-case font-normal">— optional; empty uses the task's own</span>
              </span>
              <input value={prompt} onChange={e => setPrompt(e.target.value)} disabled={running} maxLength={500}
                placeholder={task.prompt}
                className="w-full bg-black/30 border border-game-border rounded-lg px-3 py-2 text-sm text-white placeholder:text-white/25 outline-none focus:border-game-accent focus:ring-2 focus:ring-game-accent/30" />
              <span className="block text-[11px] text-white/30 mt-1.5">The success check is always the task's, so a different instruction can count as a miss.</span>
            </label>
            <div className="mt-4">
              <span className="block text-[11px] font-bold tracking-[0.14em] uppercase text-white/50 mb-2">
                Seeds<span className="ml-2 text-white/25 tracking-wide normal-case font-normal">— one episode each; the seed places the objects</span>
              </span>
              <div className="flex flex-wrap gap-2">
                {Array.from({ length: PRAXIS_MAX_SEEDS }, (_, s) => (
                  <button key={s} type="button" onClick={() => toggleSeed(s)} disabled={running}
                    className={`h-9 w-9 rounded-full border text-sm font-mono transition ${seeds.includes(s) ? 'border-game-accent bg-game-accent/20 text-white' : 'border-game-border text-white/40 hover:border-white/25'}`}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
            <Button variant="primary" size="lg" className="w-full mt-5" onClick={run} disabled={running || serverStatus === 'offline'}>
              {running ? <><Loader2 className="h-4 w-4 animate-spin mr-2" />Running {seeds.length} episode{seeds.length > 1 ? 's' : ''} · {fmt(elapsed)}</>
                : <><Play className="h-4 w-4 mr-2" />Run {seeds.length} episode{seeds.length > 1 ? 's' : ''}</>}
            </Button>
            {running && <p className="text-[11px] text-white/35 mt-2 text-center">The first job compiles kernels and captures CUDA graphs, so it takes a few minutes longer.</p>}
          </Panel>
        </div>

        <Panel title="Result" right={summary && (
          <span className="text-[11px] font-bold tracking-[0.14em] uppercase text-white/50">
            {summary.successes} of {summary.episodes} succeeded
          </span>
        )}>
          {error && (
            <div className="flex items-start gap-2 rounded-lg border border-red-400/40 bg-red-500/10 px-3 py-2 text-sm text-red-200 mb-4">
              <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />{error}
            </div>
          )}
          {videoUrl ? (
            <div className="space-y-4">
              <div className="rounded-lg bg-black/40 border border-game-border px-4 py-3 text-white text-lg font-semibold">
                “{summary?.prompt || task.prompt}”
              </div>
              <video src={videoUrl} controls autoPlay loop muted playsInline className="w-full rounded-lg border border-game-border bg-black" />
              <p className="text-[12px] text-white/40">
                {rig === 'droid'
                  ? 'Top row: the simulator and the cameras the model sees. Bottom row: the video Praxis-1 imagined. Episodes play back to back, in seed order.'
                  : 'The top camera and both wrist cameras of the two-arm rig. Episodes play back to back, in seed order.'}
              </p>
              {summary && (
                <div className="flex flex-wrap gap-2">
                  {summary.per_episode.map(e => (
                    <span key={e.seed} className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[12px] font-mono ${e.success ? 'border-emerald-400/40 text-emerald-300' : 'border-amber-300/40 text-amber-300'}`}>
                      {e.success ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
                      seed {e.seed} · {e.success ? 'success' : 'missed'}
                    </span>
                  ))}
                </div>
              )}
              <a href={videoUrl} download={`praxis-1-${summary?.task ?? taskId}.mp4`}
                className="inline-flex items-center gap-2 text-sm text-game-accent hover:underline">
                <Download className="h-4 w-4" />Download the MP4
              </a>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center text-center min-h-[360px] text-white/35 gap-3">
              {running ? <Loader2 className="h-8 w-8 animate-spin text-game-accent" /> : <Bot className="h-8 w-8" />}
              <p className="text-sm max-w-[360px]">
                {running ? 'Praxis-1 is imagining and acting, one episode per seed…'
                  : 'Pick a task and run it. You get every episode back, with what the model imagined beside what the simulator did.'}
              </p>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
};
