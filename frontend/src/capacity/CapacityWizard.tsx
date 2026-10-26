// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, AlertTriangle, ArrowUpRight, Check, Clock, Copy, Cpu, Eye, Gauge, Layers, Lock, MapPin, Radar,
  RefreshCw, Rocket, ShieldCheck, Sparkles, Terminal, X,
} from 'lucide-react';
import { bootstrap, scan, type Block, type Bootstrap, type Cell, type Instance, type Model, type RegionInfo, type ScanEvent, type Status } from './api';

const REGION_NAMES: Record<string, string> = {
  'us-east-1': 'N. Virginia', 'us-east-2': 'Ohio', 'us-west-1': 'N. California', 'us-west-2': 'Oregon',
  'ca-central-1': 'Canada', 'eu-west-1': 'Ireland', 'eu-west-2': 'London', 'eu-west-3': 'Paris',
  'eu-central-1': 'Frankfurt', 'eu-central-2': 'Zurich', 'eu-north-1': 'Stockholm', 'eu-south-1': 'Milan',
  'eu-south-2': 'Spain', 'ap-northeast-1': 'Tokyo', 'ap-northeast-2': 'Seoul', 'ap-northeast-3': 'Osaka',
  'ap-southeast-1': 'Singapore', 'ap-southeast-2': 'Sydney', 'ap-south-1': 'Mumbai', 'sa-east-1': 'São Paulo',
  'me-central-1': 'UAE', 'il-central-1': 'Tel Aviv',
};

const STATUS: Record<Status, { label: string; color: string; bg: string; ring: string }> = {
  available: { label: 'Available now', color: 'text-ok', bg: 'bg-ok', ring: 'ring-ok/50' },
  offered: { label: 'Offered, not probed', color: 'text-accent', bg: 'bg-accent/70', ring: 'ring-accent/40' },
  none: { label: 'No capacity', color: 'text-bad', bg: 'bg-bad/80', ring: 'ring-bad/40' },
  quota: { label: 'Over quota', color: 'text-warn', bg: 'bg-warn/80', ring: 'ring-warn/40' },
  unsupported: { label: 'Not offered', color: 'text-dim', bg: 'bg-white/[0.06]', ring: 'ring-white/10' },
  error: { label: 'Error', color: 'text-warn', bg: 'bg-warn/40', ring: 'ring-warn/30' },
};

const WATCH_SECONDS = 120;
const key = (r: string, az: string, t: string) => `${r}|${az}|${t}`;
const money = (n: number | null | undefined) => (n == null ? '—' : `$${n < 10 ? n.toFixed(2) : n.toFixed(1)}`);

type Phase = 'setup' | 'scanning' | 'done';

interface Option {
  tier: 1 | 2 | 3;
  region: string;
  instanceType: string;
  azs: string[];
  price: number | null;
  launch?: string;
  deploy?: string;
  hold?: string;
}

export function CapacityWizard() {
  const [boot, setBoot] = useState<Bootstrap | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const [modelId, setModelId] = useState<string>('');
  const [regions, setRegions] = useState<string[]>([]);
  const [probe, setProbe] = useState(true);
  const [blocksOn, setBlocksOn] = useState(true);
  const [phase, setPhase] = useState<Phase>('setup');
  const [candidates, setCandidates] = useState<Instance[]>([]);
  const [regionInfo, setRegionInfo] = useState<Record<string, RegionInfo>>({});
  const [cells, setCells] = useState<Record<string, Cell>>({});
  const [blocks, setBlocks] = useState<Block[]>([]);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [doneRegions, setDoneRegions] = useState<string[]>([]);
  const [watch, setWatch] = useState<{ on: boolean; next: number; rounds: number }>({ on: false, next: 0, rounds: 0 });
  const [now, setNow] = useState(Date.now());
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    bootstrap().then((b) => {
      setBoot(b);
      setModelId(b.models.find((m) => m.id === 'waypoint-1-5')?.id ?? b.models[0]?.id ?? '');
      setRegions(b.defaultRegions);
    }).catch((e) => setBootError(String(e.message ?? e)));
  }, []);

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, []);

  const model: Model | undefined = boot?.models.find((m) => m.id === modelId);

  const onEvent = useCallback((e: ScanEvent) => {
    switch (e.type) {
      case 'start': setCandidates(e.candidates); break;
      case 'region': setRegionInfo((s) => ({ ...s, [e.region]: e })); break;
      case 'cell':
        setCells((s) => ({ ...s, [key(e.region, e.az, e.instanceType)]: e }));
        if (e.leak) {
          const msg = `Probe reservation ${e.leak} (${e.instanceType}, ${e.az}) could not be cancelled and is billing. Run: ${e.cancel}`;
          setErrors((s) => ({ ...s, [e.region]: s[e.region] ? `${s[e.region]}\n${msg}` : msg }));
        }
        break;
      case 'block': setBlocks((s) => [...s, e]); break;
      case 'region-error': setErrors((s) => ({ ...s, [e.region]: e.error })); break;
      case 'region-done': setDoneRegions((s) => [...s, e.region]); break;
      case 'done': setPhase('done'); break;
    }
  }, []);

  const runScan = useCallback(async (opts?: { shortlist?: string[]; keep?: boolean }) => {
    if (!model) return;
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    if (!opts?.keep) {
      setRegionInfo({}); setCells({}); setBlocks([]); setErrors({});
    }
    setDoneRegions([]);
    setPhase('scanning');
    try {
      await scan({ model: model.id, regions, probe, blocks: blocksOn && !opts?.shortlist, instanceTypes: opts?.shortlist }, onEvent, ac.signal);
    } catch (err) {
      if (!ac.signal.aborted) setErrors((s) => ({ ...s, scan: String(err) }));
      setPhase('done');
    }
  }, [model, regions, probe, blocksOn, onEvent]);

  // ---- derived -------------------------------------------------------------
  const all = Object.values(cells);
  const counts = useMemo(() => {
    const c: Record<Status, number> = { available: 0, offered: 0, none: 0, quota: 0, unsupported: 0, error: 0 };
    for (const x of all) c[x.status] += 1;
    return c;
  }, [all]);

  const options: Option[] = useMemo(() => {
    const byKey = new Map<string, Option>();
    for (const c of all) {
      if (c.status !== 'available' && c.status !== 'offered') continue;
      const verified = c.status === 'available';
      const tier: Option['tier'] = verified && c.inStackAzs && c.launch ? 1 : verified && c.inStackAzs ? 2 : 3;
      const k = `${c.region}|${c.instanceType}|${verified}`;
      const price = regionInfo[c.region]?.prices[c.instanceType] ?? null;
      const prev = byKey.get(k);
      if (prev) {
        prev.azs.push(c.az);
        prev.tier = Math.min(prev.tier, tier) as Option['tier'];
        prev.launch ??= c.launch; prev.deploy ??= c.deploy; prev.hold ??= c.hold;
      } else {
        byKey.set(k, { tier: verified ? tier : 3, region: c.region, instanceType: c.instanceType, azs: [c.az], price, launch: c.launch, deploy: c.deploy, hold: c.hold });
      }
    }
    return [...byKey.values()]
      .filter((o) => probe ? o.deploy : true)
      .sort((a, b) => a.tier - b.tier || (a.price ?? 1e9) - (b.price ?? 1e9) || (a.instanceType === model?.defaultInstance ? -1 : 1));
  }, [all, regionInfo, probe, model]);

  const sortedBlocks = useMemo(() => [...blocks].sort((a, b) => a.start.localeCompare(b.start) || a.fee - b.fee), [blocks]);
  const nothingNow = phase === 'done' && probe && counts.available === 0;

  // ---- watch mode: re-probe the offered shortlist until something turns green --
  useEffect(() => {
    if (!watch.on || phase !== 'done') return;
    if (counts.available > 0) {
      setWatch((w) => ({ ...w, on: false }));
      document.title = '● Capacity found · GPU Capacity Wizard';
      if ('Notification' in window && Notification.permission === 'granted') {
        new Notification('GPU capacity found', { body: `${model?.name}: ${options[0]?.instanceType} in ${options[0]?.region}` });
      }
      return;
    }
    if (!watch.next) { setWatch((w) => ({ ...w, next: Date.now() + WATCH_SECONDS * 1000 })); return; }
    if (now >= watch.next) {
      const shortlist = [...new Set(all.filter((c) => c.status !== 'unsupported').map((c) => c.instanceType))];
      setWatch((w) => ({ ...w, next: 0, rounds: w.rounds + 1 }));
      void runScan({ shortlist, keep: true });
    }
  }, [watch, phase, now, counts.available, all, runScan, model, options]);

  const startWatch = () => {
    if ('Notification' in window && Notification.permission === 'default') void Notification.requestPermission();
    setWatch({ on: true, next: Date.now() + WATCH_SECONDS * 1000, rounds: 0 });
  };

  // ---- render --------------------------------------------------------------
  if (bootError) return <Shell><Fatal message={bootError} /></Shell>;
  if (!boot) return <Shell><div className="grid place-items-center h-[70vh]"><RadarDial scanning blips={[]} size={220} /></div></Shell>;

  const progress = regions.length ? doneRegions.length / regions.length : 0;

  return (
    <Shell identity={boot.identity} launchStack={boot.launchStack}>
      {/* Hero */}
      <section className="grid grid-cols-[1fr_auto] gap-10 items-center pt-10 pb-8">
        <div className="cw-in">
          <div className="cw-kicker flex items-center gap-2"><Radar className="h-3.5 w-3.5 text-accent" /> GPU capacity wizard</div>
          <h1 className="cw-display text-[54px] leading-[1.02] font-bold mt-4 text-white">
            Find the GPU <span className="text-accent">before</span><br />you deploy.
          </h1>
          <p className="mt-5 max-w-[620px] text-[17px] leading-relaxed text-mute">
            Pick a cartridge and the Regions you can use. The wizard checks every Availability Zone for the GPUs that
            cartridge runs on: offered, within your quota, free right now, or bookable as a Capacity Block.
            Then it hands you a pre-filled Launch Stack.
          </p>
        </div>
        <RadarDial scanning={phase === 'scanning'} size={250}
          blips={all.filter((c) => c.status === 'available').map((c) => key(c.region, c.az, c.instanceType))} />
      </section>

      {/* Step 1: cartridge */}
      <Step n={1} title="Cartridge" hint="Requirements come from each cartridge's endpoint.yaml">
        <div className="grid grid-cols-4 gap-3">
          {boot.models.map((m, i) => (
            <button key={m.id} onClick={() => setModelId(m.id)} style={{ animationDelay: `${i * 40}ms` }}
              className={`cw-in group text-left rounded-xl border px-4 py-3.5 transition-all duration-200 cursor-pointer ${
                m.id === modelId
                  ? 'border-accent/70 bg-accent/[0.08] shadow-[0_0_0_1px_rgba(83,159,229,0.35),0_10px_40px_-12px_rgba(83,159,229,0.55)]'
                  : 'border-line bg-panel/60 hover:border-line-strong hover:bg-panel-2'}`}>
              <div className="flex items-start justify-between gap-2">
                <span className="cw-display text-[16px] font-semibold text-white leading-tight">{m.name}</span>
                {m.restricted && <span title="Licence-restricted: needs AcceptModelLicense" className="shrink-0 text-warn"><Lock className="h-3.5 w-3.5" /></span>}
              </div>
              <div className="mt-2 flex items-center gap-2 text-[12px] cw-num text-mute">
                <Cpu className="h-3.5 w-3.5" />{m.gpuCount}× GPU{m.gpuMemoryGb ? ` ≥ ${m.gpuMemoryGb} GB` : ''}
              </div>
              <div className="mt-1 text-[11px] cw-num text-dim">default {m.defaultInstance}</div>
            </button>
          ))}
        </div>
      </Step>

      {/* Step 2: where */}
      <Step n={2} title="Regions" hint={`${regions.length} selected`}>
        <div className="flex flex-wrap gap-2">
          {boot.regions.filter((r) => REGION_NAMES[r]).map((r) => {
            const on = regions.includes(r);
            return (
              <button key={r} onClick={() => setRegions((s) => (on ? s.filter((x) => x !== r) : [...s, r]))}
                className={`rounded-full border px-3.5 py-1.5 text-[13px] transition-all cursor-pointer ${
                  on ? 'border-accent/70 bg-accent/15 text-white' : 'border-line text-mute hover:text-white hover:border-line-strong'}`}>
                <span className="font-medium">{REGION_NAMES[r]}</span>
                <span className="cw-num text-[11px] ml-2 opacity-60">{r}</span>
              </button>
            );
          })}
        </div>
        <div className="mt-5 grid grid-cols-2 gap-3">
          <Toggle on={probe} onChange={setProbe} icon={<Activity className="h-4 w-4" />} title="Live capacity probe"
            body="Creates a 1-instance Capacity Reservation in each AZ and cancels it at once. That's the only way to know there's capacity right now; it is billed for seconds at most." />
          <Toggle on={blocksOn} onChange={setBlocksOn} icon={<Layers className="h-4 w-4" />} title="Capacity Blocks"
            body="Lists bookable 24-hour Capacity Blocks for P-series types over the next 7 days, with the upfront fee. Nothing is bought." />
        </div>
      </Step>

      {/* Scan */}
      <div className="flex items-center gap-5 mt-2 mb-10">
        <button disabled={!model || !regions.length || phase === 'scanning'} onClick={() => { setWatch({ on: false, next: 0, rounds: 0 }); void runScan(); }}
          className="relative overflow-hidden cw-display text-[17px] font-semibold tracking-wide rounded-xl px-8 py-4 bg-accent text-ink disabled:opacity-60 cursor-pointer hover:brightness-110 transition shadow-[0_12px_40px_-10px_rgba(83,159,229,0.8)]">
          <span className="relative z-10 flex items-center gap-2.5">
            {phase === 'scanning' ? <RefreshCw className="h-5 w-5 animate-spin" /> : <Radar className="h-5 w-5" />}
            {phase === 'scanning' ? 'Scanning…' : phase === 'done' ? 'Scan again' : 'Scan for capacity'}
          </span>
          {phase === 'scanning' && <span className="absolute inset-y-0 left-0 w-1/3 bg-white/25 cw-scan-bar" />}
        </button>
        {model && (
          <div className="text-[13px] text-mute">
            <span className="text-white font-medium">{model.name}</span> · {model.candidates.length} GPU types × {regions.length} Regions
            {phase !== 'setup' && <span className="cw-num ml-3 text-accent">{Math.round(progress * 100)}%</span>}
          </div>
        )}
      </div>

      {phase !== 'setup' && (
        <>
          <Summary counts={counts} blocks={blocks.length} probe={probe} />
          <div className="grid grid-cols-[1fr_400px] gap-6 mt-6 items-start">
            <div className="space-y-4">
              {regions.map((r) => (
                <RegionCard key={r} region={r} info={regionInfo[r]} candidates={candidates} cells={cells} error={errors[r]}
                  defaultInstance={model?.defaultInstance ?? null} model={model} />
              ))}
            </div>
            <aside className="sticky top-6 space-y-4">
              <Recommendations options={options} blocks={sortedBlocks} phase={phase} probe={probe}
                nothingNow={nothingNow} watch={watch} now={now} onWatch={startWatch}
                onStopWatch={() => setWatch({ on: false, next: 0, rounds: 0 })} launchConfigured={boot.launchStack} />
            </aside>
          </div>
        </>
      )}
    </Shell>
  );
}

// --------------------------------------------------------------------------- //

function Shell({ children, identity, launchStack }: { children: React.ReactNode; identity?: Bootstrap['identity']; launchStack?: boolean }) {
  return (
    <div className="min-h-screen">
      <div className="cw-backdrop" />
      <header className="border-b border-line bg-ink/60 backdrop-blur-md sticky top-0 z-20">
        <div className="max-w-[1440px] mx-auto px-10 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-7 w-7 rounded-md bg-accent/15 border border-accent/40 grid place-items-center"><Gauge className="h-4 w-4 text-accent" /></div>
            <span className="cw-display font-semibold text-white">World Model Inference</span>
            <span className="text-dim">/</span>
            <span className="text-mute text-[14px]">Capacity</span>
          </div>
          {identity && (
            <div className="flex items-center gap-4 text-[12px]">
              <span className={`flex items-center gap-1.5 ${launchStack ? 'text-ok' : 'text-dim'}`}>
                <Rocket className="h-3.5 w-3.5" />{launchStack ? 'Launch Stack linked' : 'Launch Stack not configured'}
              </span>
              <span className="flex items-center gap-1.5 rounded-full border border-line px-3 py-1 cw-num text-mute">
                <ShieldCheck className="h-3.5 w-3.5 text-accent" />account {identity.account}
              </span>
            </div>
          )}
        </div>
      </header>
      <main className="max-w-[1440px] mx-auto px-10 pb-24">{children}</main>
    </div>
  );
}

function Fatal({ message }: { message: string }) {
  return (
    <div className="cw-panel mt-20 p-8 max-w-xl">
      <div className="flex items-center gap-2 text-warn"><AlertTriangle className="h-5 w-5" /> Can't reach the wizard server</div>
      <p className="mt-3 text-mute">{message}</p>
      <pre className="mt-4 cw-num text-[13px] text-white bg-black/40 rounded-lg p-3">python scripts/capacity.py</pre>
    </div>
  );
}

function RadarDial({ scanning, blips, size }: { scanning: boolean; blips: string[]; size: number }) {
  const pos = (s: string) => {
    let h = 2166136261;
    for (const ch of s) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
    const a = ((h >>> 0) % 360) * (Math.PI / 180);
    const r = 0.25 + (((h >>> 9) % 1000) / 1000) * 0.62;
    return { left: `${50 + Math.cos(a) * r * 50}%`, top: `${50 + Math.sin(a) * r * 50}%` };
  };
  return (
    <div className="relative shrink-0 rounded-full border border-accent/25 bg-[radial-gradient(circle,rgba(83,159,229,0.10),transparent_70%)]"
      style={{ width: size, height: size }}>
      {[0.33, 0.66].map((f) => (
        <div key={f} className="absolute rounded-full border border-accent/15" style={{ inset: `${(1 - f) * 50}%` }} />
      ))}
      <div className="absolute inset-0 grid place-items-center"><div className="h-px w-full bg-accent/10" /></div>
      <div className="absolute inset-0 grid place-items-center"><div className="w-px h-full bg-accent/10" /></div>
      <div className={`absolute inset-0 cw-sweep ${scanning ? '' : 'idle'}`} />
      {blips.slice(0, 40).map((b) => (
        <span key={b} className="absolute h-2.5 w-2.5 -ml-[5px] -mt-[5px] rounded-full bg-ok text-ok cw-ping cw-pop" style={pos(b)} />
      ))}
      <div className="absolute inset-[46%] rounded-full bg-accent shadow-[0_0_20px_rgba(83,159,229,0.9)]" />
    </div>
  );
}

function Step({ n, title, hint, children }: { n: number; title: string; hint?: string; children: React.ReactNode }) {
  return (
    <section className="cw-panel p-6 mb-5 cw-in">
      <div className="flex items-baseline gap-3 mb-4">
        <span className="cw-num text-[12px] text-accent border border-accent/40 rounded-md px-1.5 py-0.5">0{n}</span>
        <h2 className="cw-display text-[20px] font-semibold text-white">{title}</h2>
        {hint && <span className="text-[12px] text-dim">{hint}</span>}
      </div>
      {children}
    </section>
  );
}

function Toggle({ on, onChange, icon, title, body }: { on: boolean; onChange: (v: boolean) => void; icon: React.ReactNode; title: string; body: string }) {
  return (
    <button onClick={() => onChange(!on)}
      className={`text-left rounded-xl border p-4 flex gap-4 transition cursor-pointer ${on ? 'border-accent/50 bg-accent/[0.06]' : 'border-line hover:border-line-strong'}`}>
      <span className={`mt-0.5 relative h-6 w-11 shrink-0 rounded-full transition ${on ? 'bg-accent' : 'bg-white/10'}`}>
        <span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-all ${on ? 'left-[22px]' : 'left-0.5'}`} />
      </span>
      <span>
        <span className="flex items-center gap-2 text-white font-medium text-[14px]">{icon}{title}</span>
        <span className="block mt-1 text-[12.5px] leading-relaxed text-mute">{body}</span>
      </span>
    </button>
  );
}

function Summary({ counts, blocks, probe }: { counts: Record<Status, number>; blocks: number; probe: boolean }) {
  const items: [string, number, string, React.ReactNode][] = [
    probe ? ['Available now', counts.available, 'text-ok', <Sparkles key="a" className="h-4 w-4" />]
      : ['Offered', counts.offered, 'text-accent', <Eye key="a" className="h-4 w-4" />],
    ['No capacity', counts.none, 'text-bad', <X key="b" className="h-4 w-4" />],
    ['Over quota', counts.quota, 'text-warn', <AlertTriangle key="c" className="h-4 w-4" />],
    ['Capacity Blocks', blocks, 'text-block', <Layers key="d" className="h-4 w-4" />],
  ];
  return (
    <div className="grid grid-cols-4 gap-3">
      {items.map(([label, n, color, icon], i) => (
        <div key={label} className="cw-panel px-5 py-4 cw-in" style={{ animationDelay: `${i * 60}ms` }}>
          <div className={`flex items-center gap-2 text-[12px] ${color}`}>{icon}{label}</div>
          <div className={`cw-num text-[34px] font-medium mt-1 ${n ? color : 'text-dim'}`}>{n}</div>
        </div>
      ))}
    </div>
  );
}

function RegionCard({ region, info, candidates, cells, error, defaultInstance, model }: {
  region: string; info?: RegionInfo; candidates: Instance[]; cells: Record<string, Cell>; error?: string;
  defaultInstance: string | null; model?: Model;
}) {
  const quotaShort = (fam: 'G' | 'P', vcpus: number | null) => {
    const q = info?.quota[fam];
    return q && q.limit != null && vcpus != null && q.limit - q.used < vcpus;
  };
  return (
    <div className="cw-panel p-5 cw-in">
      <div className="flex items-center justify-between">
        <div className="flex items-baseline gap-3">
          <MapPin className="h-4 w-4 text-accent self-center" />
          <span className="cw-display text-[18px] font-semibold text-white">{REGION_NAMES[region] ?? region}</span>
          <span className="cw-num text-[12px] text-dim">{region}</span>
        </div>
        <div className="flex items-center gap-3 text-[11.5px] cw-num text-mute">
          {info && (['G', 'P'] as const).map((f) => (
            <span key={f} title={`On-Demand ${f === 'G' ? 'G and VT' : 'P'} instances quota (vCPUs)`}>
              {f}: {info.quota[f].used}/{info.quota[f].limit ?? '?'} vCPU
            </span>
          ))}
          {info?.templatePublished && <span className="text-ok flex items-center gap-1"><Rocket className="h-3 w-3" />template</span>}
          {!info && !error && <RefreshCw className="h-3.5 w-3.5 animate-spin text-accent" />}
        </div>
      </div>
      {error && <div className="mt-3 text-[13px] text-warn flex gap-2"><AlertTriangle className="h-4 w-4 shrink-0" /><span className="whitespace-pre-line break-words">{error}</span></div>}
      {info && (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full border-separate border-spacing-x-1.5 border-spacing-y-1.5">
            <thead>
              <tr>
                <th className="w-[120px]" />
                {candidates.map((c) => (
                  <th key={c.type} className="text-left align-bottom font-normal pb-1">
                    <div className={`cw-num text-[12px] ${c.type === defaultInstance ? 'text-white' : 'text-mute'}`}>
                      {c.type}{c.type === defaultInstance && <span className="text-accent"> ★</span>}
                    </div>
                    <div className="text-[10.5px] text-dim cw-num">{c.gpus}× {c.gpu} · {money(info.prices[c.type])}/h</div>
                    {quotaShort(c.family, c.vcpus) && <div className="text-[10.5px] text-warn">quota too low</div>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {info.azs.map((az) => (
                <tr key={az}>
                  <td className="cw-num text-[12px] text-mute pr-2 whitespace-nowrap">
                    {az}
                    {info.stackAzs.includes(az) && <span className="ml-2 text-[9.5px] tracking-wider text-accent border border-accent/40 rounded px-1 py-px">STACK</span>}
                  </td>
                  {candidates.map((c) => {
                    const cell = cells[key(region, az, c.type)];
                    return <td key={c.type}><Led cell={cell} model={model} /></td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Led({ cell }: { cell?: Cell; model?: Model }) {
  if (!cell) return <div className="h-9 rounded-lg bg-white/[0.03] border border-dashed border-white/10 animate-pulse" />;
  const s = STATUS[cell.status];
  return (
    <div title={`${s.label}${cell.detail ? ` — ${cell.detail}` : ''}`}
      className={`cw-pop h-9 rounded-lg ring-1 ${s.ring} flex items-center gap-2 px-2.5 ${cell.status === 'unsupported' ? 'bg-white/[0.02]' : 'bg-black/30'}`}>
      <span className={`h-2.5 w-2.5 rounded-full ${s.bg} ${cell.status === 'available' ? 'cw-led-ok' : ''}`} />
      <span className={`text-[11.5px] ${s.color} truncate`}>{cell.status === 'unsupported' ? '—' : s.label.replace(', not probed', '')}</span>
    </div>
  );
}

function Recommendations({ options, blocks, phase, probe, nothingNow, watch, now, onWatch, onStopWatch, launchConfigured }: {
  options: Option[]; blocks: Block[]; phase: Phase; probe: boolean; nothingNow: boolean;
  watch: { on: boolean; next: number; rounds: number }; now: number; onWatch: () => void; onStopWatch: () => void; launchConfigured: boolean;
}) {
  const top = options.slice(0, 3);
  return (
    <>
      <div className="cw-panel p-5">
        <div className="cw-kicker flex items-center gap-2 mb-4"><Sparkles className="h-3.5 w-3.5 text-ok" /> Best options</div>
        {phase === 'scanning' && !top.length && <div className="text-mute text-[14px] py-6 text-center">Listening for capacity…</div>}
        {!top.length && phase === 'done' && (
          <div className="text-[14px] text-mute">{probe ? 'No GPU type the cartridge runs on is free in these Regions right now.' : 'Nothing offered within quota.'}</div>
        )}
        <div className="space-y-3">
          {top.map((o, i) => <OptionCard key={`${o.region}${o.instanceType}`} o={o} rank={i + 1} probe={probe} launchConfigured={launchConfigured} />)}
        </div>
        {nothingNow && (
          <div className="mt-4 rounded-xl border border-accent/30 bg-accent/[0.06] p-4">
            <div className="flex items-center gap-2 text-white font-medium text-[14px]"><Eye className="h-4 w-4 text-accent" /> Watch for capacity</div>
            <p className="text-[12.5px] text-mute mt-1">Re-probes the offered types every {WATCH_SECONDS / 60} minutes and notifies you when one frees up.</p>
            {watch.on ? (
              <div className="mt-3 flex items-center justify-between">
                <span className="cw-num text-[13px] text-accent flex items-center gap-2"><Clock className="h-4 w-4" />
                  {phase === 'scanning' ? 'probing…' : `next check in ${Math.max(0, Math.ceil((watch.next - now) / 1000))}s`} · round {watch.rounds}</span>
                <button onClick={onStopWatch} className="text-[12px] text-mute hover:text-white cursor-pointer">Stop</button>
              </div>
            ) : (
              <button onClick={onWatch} className="mt-3 w-full rounded-lg bg-accent/20 border border-accent/50 text-white py-2 text-[13px] font-medium hover:bg-accent/30 cursor-pointer">Start watching</button>
            )}
          </div>
        )}
      </div>
      {blocks.length > 0 && (
        <div className="cw-panel p-5">
          <div className="cw-kicker flex items-center gap-2 mb-3"><Layers className="h-3.5 w-3.5 text-block" /> Capacity Blocks · 24 h</div>
          <div className="space-y-2.5">
            {blocks.slice(0, 4).map((b) => <BlockRow key={b.offeringId} b={b} />)}
          </div>
          <p className="text-[11.5px] text-dim mt-3 leading-relaxed">Upfront and non-refundable. Pass the reservation ID as <span className="cw-num">CapacityBlockId</span>; the instance is pinned to the block's AZ.</p>
        </div>
      )}
    </>
  );
}

function OptionCard({ o, rank, probe, launchConfigured }: { o: Option; rank: number; probe: boolean; launchConfigured: boolean }) {
  const verified = probe;
  return (
    <div className={`rounded-xl border p-4 cw-in ${rank === 1 ? 'border-ok/50 bg-ok/[0.05] shadow-[0_10px_40px_-18px_rgba(74,222,128,0.6)]' : 'border-line bg-black/20'}`}
      style={{ animationDelay: `${rank * 80}ms` }}>
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="cw-num text-[11px] text-dim">#{rank}</span>
            <span className="cw-num text-[15px] text-white font-medium">{o.instanceType}</span>
          </div>
          <div className="text-[12.5px] text-mute mt-0.5">{REGION_NAMES[o.region] ?? o.region} · {o.azs.sort().join(', ')}</div>
        </div>
        <div className="text-right">
          <div className="cw-num text-[15px] text-white">{money(o.price)}<span className="text-dim text-[11px]">/h</span></div>
          <div className={`text-[11px] mt-0.5 ${verified ? 'text-ok' : 'text-accent'}`}>{verified ? '● free now' : '○ unverified'}</div>
        </div>
      </div>
      {o.tier === 3 && verified && <div className="mt-2 text-[11.5px] text-warn">Outside the Launch Stack's two AZs: use ./deploy.sh</div>}
      <div className="mt-3 flex flex-wrap gap-2">
        {o.launch ? (
          <a href={o.launch} target="_blank" rel="noreferrer"
            className="inline-flex items-center gap-1.5 rounded-lg bg-ok text-ink px-3 py-1.5 text-[12.5px] font-semibold hover:brightness-110">
            <Rocket className="h-3.5 w-3.5" /> Launch Stack <ArrowUpRight className="h-3.5 w-3.5" />
          </a>
        ) : verified && o.tier < 3 && !launchConfigured ? (
          <span className="text-[11.5px] text-dim self-center">Start with --launch-stack for a pre-filled link</span>
        ) : null}
        {o.deploy && <CopyButton text={o.deploy} label="deploy.sh" icon={<Terminal className="h-3.5 w-3.5" />} />}
        {o.hold && <CopyButton text={o.hold} label="Hold it" icon={<Lock className="h-3.5 w-3.5" />} />}
      </div>
    </div>
  );
}

function BlockRow({ b }: { b: Block }) {
  const start = new Date(b.start);
  return (
    <div className="rounded-lg border border-block/25 bg-block/[0.05] p-3">
      <div className="flex items-center justify-between">
        <span className="cw-num text-[13px] text-white">{b.instanceType}</span>
        <span className="cw-num text-[13px] text-block">${b.fee.toFixed(0)}</span>
      </div>
      <div className="flex items-center justify-between mt-1 text-[11.5px] text-mute">
        <span>{REGION_NAMES[b.region] ?? b.region} · {b.az}{!b.inStackAzs && <span className="text-warn"> · not a stack AZ</span>}</span>
        <span className="cw-num">{start.toLocaleString(undefined, { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}</span>
      </div>
      <div className="mt-2"><CopyButton text={b.purchase} label="Purchase command" icon={<Terminal className="h-3.5 w-3.5" />} /></div>
    </div>
  );
}

function CopyButton({ text, label, icon }: { text: string; label: string; icon: React.ReactNode }) {
  const [done, setDone] = useState(false);
  return (
    <button title={text} onClick={() => { void navigator.clipboard.writeText(text); setDone(true); setTimeout(() => setDone(false), 1400); }}
      className="inline-flex items-center gap-1.5 rounded-lg border border-line-strong px-3 py-1.5 text-[12.5px] text-white hover:bg-white/[0.06] cursor-pointer">
      {done ? <Check className="h-3.5 w-3.5 text-ok" /> : icon}{done ? 'Copied' : label}{!done && <Copy className="h-3 w-3 opacity-50" />}
    </button>
  );
}
