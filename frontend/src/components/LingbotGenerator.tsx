// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertCircle, CheckCircle2, ChevronDown, Download, Film, Loader2,
  Sparkles, Upload, XCircle,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { getConfig } from '@/services/config';
import { BundledExample } from '@/data/lingbot-examples';
import { profileFor } from '@/data/gen-profiles';

const SIZE_OPTIONS = [
  { label: '480×832',   value: '480*832' },
  { label: '832×480',   value: '832*480' },
  { label: '720×1280',  value: '720*1280' },
  { label: '1280×720',  value: '1280*720' },
];

type ServerStatus = 'unknown' | 'loading' | 'ready' | 'offline';

const StatusPill = ({ status }: { status: ServerStatus }) => {
  const map: Record<ServerStatus, { text: string; cls: string; Icon: any }> = {
    unknown: { text: 'checking…', cls: 'text-white/40 border-white/10', Icon: Loader2 },
    loading: { text: 'model loading', cls: 'text-amber-300 border-amber-300/40', Icon: Loader2 },
    ready:   { text: 'ready',       cls: 'text-emerald-300 border-emerald-400/40', Icon: CheckCircle2 },
    offline: { text: 'offline',     cls: 'text-red-300 border-red-400/40', Icon: XCircle },
  };
  const { text, cls, Icon } = map[status];
  const spinning = status === 'unknown' || status === 'loading';
  return (
    <span className={`inline-flex items-center gap-2 text-[11px] font-medium tracking-wider uppercase px-3 py-1 rounded-full border ${cls}`}>
      <Icon className={`h-3 w-3 ${spinning ? 'animate-spin' : ''}`} />
      {text}
    </span>
  );
};

// Card / panel shell used throughout
const Panel = ({ title, right, children }: { title?: string; right?: React.ReactNode; children: React.ReactNode }) => (
  <div className="rounded-xl bg-game-surface/70 backdrop-blur-sm border border-game-border p-5 md:p-6">
    {(title || right) && (
      <div className="flex items-center justify-between mb-4">
        {title && (
          <h2 className="text-[11px] font-bold tracking-[0.14em] uppercase text-white/40">
            {title}
          </h2>
        )}
        {right}
      </div>
    )}
    {children}
  </div>
);

// Labelled field
const Field = ({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) => (
  <label className="block">
    <span className="block text-[11px] font-bold tracking-[0.14em] uppercase text-white/50 mb-2">
      {label}{hint && <span className="ml-2 text-white/25 tracking-wide normal-case font-normal">— {hint}</span>}
    </span>
    {children}
  </label>
);

const inputCls =
  'w-full bg-black/30 border border-game-border rounded-lg px-3 py-2 text-sm text-white ' +
  'placeholder:text-white/25 outline-none focus:border-game-accent focus:ring-2 focus:ring-game-accent/30 ' +
  'disabled:opacity-50';

export const LingbotGenerator = ({
  cartridgeId,
  endpointUrl,
}: {
  /** Which cartridge this generator drives — selects the control profile. */
  cartridgeId?: string;
  /** Endpoint override; falls back to config.lingbotApiUrl (lingbot-fast). */
  endpointUrl?: string;
} = {}) => {
  const config = getConfig();
  const apiUrl = endpointUrl || config.lingbotApiUrl;
  // R1: present the shared API token the backend validates. fetch() calls send
  // it as a Bearer header; <img> src URLs (which can't set headers) append it as
  // a ?token= query param, which the backend AuthMiddleware also accepts.
  const apiToken = config.apiToken;
  const authHeaders: Record<string, string> = apiToken ? { Authorization: `Bearer ${apiToken}` } : {};
  const withToken = (u: string) => (apiToken ? `${u}${u.includes('?') ? '&' : '?'}token=${encodeURIComponent(apiToken)}` : u);
  const profile = profileFor(cartridgeId);
  const firstExample = profile.examples[0] ?? null;

  // Core input state
  const [prompt, setPrompt] = useState(firstExample?.prompt ?? profile.defaultPrompt);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(
    firstExample ? withToken(`${apiUrl}/examples/${firstExample.id}/image`) : null,
  );
  const [selectedExampleId, setSelectedExampleId] = useState<string | null>(firstExample?.id ?? null);
  const [examples] = useState<BundledExample[]>(profile.examples);
  const [serverStatus, setServerStatus] = useState<ServerStatus>('unknown');

  // Advanced settings — seeded from the cartridge's profile.
  const [frameNum, setFrameNum] = useState(profile.defaults.frameNum);
  const [size, setSize] = useState(profile.defaults.size);
  const [samplingSteps, setSamplingSteps] = useState(profile.defaults.samplingSteps);
  const [guideScale, setGuideScale] = useState(profile.defaults.guideScale);
  const [seed, setSeed] = useState(profile.defaults.seed);
  const [useDmd, setUseDmd] = useState(false);

  // Camera (only Upload .npy is wired on the server today)
  const [cameraMode, setCameraMode] = useState<'none' | 'upload'>('none');
  const [posesFile, setPosesFile] = useState<File | null>(null);
  const [intrinsicsFile, setIntrinsicsFile] = useState<File | null>(null);

  // Generation lifecycle
  const [isGenerating, setIsGenerating] = useState(false);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  const elapsedRef = useRef<number | null>(null);

  const imageInputRef = useRef<HTMLInputElement>(null);
  const posesInputRef = useRef<HTMLInputElement>(null);
  const intrinsicsInputRef = useRef<HTMLInputElement>(null);

  // ─── Health probe on mount (examples are hardcoded, no fetch needed) ───
  useEffect(() => {
    const probeHealth = () => fetch(`${apiUrl}/health`, { cache: 'no-store', headers: authHeaders })
      .then(r => r.json())
      .then(d => setServerStatus(d.model_loaded ? 'ready' : 'loading'))
      .catch(() => setServerStatus('offline'));
    probeHealth();
    const h = window.setInterval(probeHealth, 15_000);
    return () => window.clearInterval(h);
  }, [apiUrl]);

  // ─── Handlers ───────────────────────────────────────────────────
  const handleImageSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    setSelectedExampleId(null);
    const reader = new FileReader();
    reader.onload = (ev) => setImagePreview(ev.target?.result as string);
    reader.readAsDataURL(file);
  }, []);

  const handleExampleClick = useCallback((ex: BundledExample) => {
    setSelectedExampleId(ex.id);
    setPrompt(ex.prompt);
    setImagePreview(withToken(`${apiUrl}/examples/${ex.id}/image`));
    setImageFile(null);
    if (ex.has_poses) setCameraMode('none');
  }, [apiUrl]);

  const pollJobStatus = useCallback(async (jobId: string) => {
    try {
      const resp = await fetch(`${apiUrl}/status/${jobId}`, { headers: authHeaders });
      const data = await resp.json();
      if (data.status === 'complete') {
        const videoResp = await fetch(`${apiUrl}/result/${jobId}`, { headers: authHeaders });
        const blob = await videoResp.blob();
        setVideoUrl(URL.createObjectURL(blob));
        setIsGenerating(false);
        if (pollRef.current) window.clearInterval(pollRef.current);
        if (elapsedRef.current) window.clearInterval(elapsedRef.current);
      } else if (data.status === 'failed') {
        setError(data.error || 'Generation failed');
        setIsGenerating(false);
        if (pollRef.current) window.clearInterval(pollRef.current);
        if (elapsedRef.current) window.clearInterval(elapsedRef.current);
      }
    } catch { /* keep polling */ }
  }, [apiUrl]);

  const handleGenerate = useCallback(async () => {
    // Cartridges that ship bundled examples (lingbot, lyra) are image-to-video,
    // so an image or example is required. Text-to-video cartridges (cosmos3)
    // have no examples and generate from the prompt alone — don't block them.
    const requiresImage = profile.examples.length > 0;
    if (requiresImage && !imageFile && !selectedExampleId) {
      setError('Please upload an image or select a bundled example.');
      return;
    }
    setIsGenerating(true);
    setError(null);
    setVideoUrl(null);
    setElapsedSec(0);

    try {
      const formData = new FormData();
      formData.append('prompt', prompt);
      // Only send controls this cartridge actually honours — otherwise a knob
      // silently no-ops on the runner. `sampling_steps` + `seed` are shared.
      formData.append('sampling_steps', samplingSteps);
      formData.append('seed', seed);
      if (profile.show.frames) formData.append('frame_num', frameNum);
      if (profile.show.size) formData.append('size', size);
      if (profile.show.guidance) formData.append('guide_scale', guideScale);
      if (profile.show.dmd) formData.append('use_dmd', String(useDmd));
      if (imageFile) formData.append('image', imageFile);
      else if (selectedExampleId) formData.append('example_id', selectedExampleId);
      if (profile.show.camera && cameraMode === 'upload' && posesFile && intrinsicsFile) {
        formData.append('poses', posesFile);
        formData.append('intrinsics', intrinsicsFile);
      }

      const resp = await fetch(`${apiUrl}/generate`, { method: 'POST', body: formData, headers: authHeaders });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || 'Generation request failed');
      }
      const { job_id } = await resp.json();
      pollRef.current = window.setInterval(() => pollJobStatus(job_id), 3000);
      elapsedRef.current = window.setInterval(() => setElapsedSec(s => s + 1), 1000);
    } catch (err: any) {
      setError(err.message || 'Generation failed');
      setIsGenerating(false);
    }
  }, [apiUrl, imageFile, selectedExampleId, prompt, frameNum, size, samplingSteps, guideScale, seed, useDmd, profile, cameraMode, posesFile, intrinsicsFile, pollJobStatus]);

  useEffect(() => () => {
    if (pollRef.current) window.clearInterval(pollRef.current);
    if (elapsedRef.current) window.clearInterval(elapsedRef.current);
  }, []);

  // With bundled examples, either an upload or a selection works; without them
  // (e.g. lyra-2) an uploaded image is required.
  const hasInput = examples.length ? (imageFile || selectedExampleId) : imageFile;
  const canGenerate = serverStatus === 'ready' && !isGenerating && !!hasInput;

  // ─── Render ─────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-6">
      {/* Examples row — hardcoded list so it's resilient to server hiccups.
          Cartridges without bundled examples (e.g. lyra-2) skip this and show a
          standalone status pill in the Input panel instead. */}
      {profile.show.examples && (
      <Panel title="Pick a bundled scene — click to load prompt + image" right={<StatusPill status={serverStatus} />}>
        <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3">
          {examples.map(ex => {
            const selected = selectedExampleId === ex.id;
            const truncated = ex.prompt.length > 110 ? ex.prompt.slice(0, 107) + '…' : ex.prompt;
            // Inline gradient fallback for when the server is down / unreachable
            const hue = 180 + Number(ex.id) * 40;
            return (
              <button
                key={ex.id}
                onClick={() => handleExampleClick(ex)}
                disabled={isGenerating}
                className={`group relative text-left bg-black/20 rounded-lg border-2 overflow-hidden transition-all duration-200 cursor-pointer
                  ${selected
                    ? 'border-game-accent shadow-[0_0_24px_rgba(83,159,229,0.3)]'
                    : 'border-transparent hover:border-white/20 hover:-translate-y-0.5'}
                  disabled:opacity-60 disabled:cursor-not-allowed`}
              >
                <div
                  className="aspect-video overflow-hidden relative"
                  style={{
                    background: `
                      radial-gradient(at 30% 30%, hsl(${hue} 55% 40% / 0.9), transparent 60%),
                      radial-gradient(at 70% 70%, hsl(${(hue + 40) % 360} 50% 30% / 0.8), transparent 65%),
                      linear-gradient(135deg, hsl(${hue} 30% 10%), hsl(${(hue + 60) % 360} 30% 8%))`,
                  }}
                >
                  <img
                    src={withToken(`${apiUrl}/examples/${ex.id}/image`)}
                    alt={`Example ${ex.id}`}
                    className="w-full h-full object-cover transition-all duration-300 group-hover:brightness-110"
                    onError={e => { (e.currentTarget as HTMLImageElement).style.opacity = '0'; }}
                    loading="lazy"
                  />
                </div>
                <div className="p-2.5">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[11px] font-bold tracking-wider text-white/80">
                      {ex.id} · {ex.label}
                    </span>
                    {ex.has_poses && <span className="text-[10px]" title="Camera poses bundled">📷</span>}
                  </div>
                  <p className="text-[11px] leading-snug text-white/50 line-clamp-2">
                    {truncated}
                  </p>
                </div>
              </button>
            );
          })}
        </div>
      </Panel>
      )}

      {/* Input: image + prompt + camera */}
      <Panel
        title="Input"
        right={!profile.show.examples ? <StatusPill status={serverStatus} /> : undefined}
      >
        <div className="flex flex-col gap-5">
          <Field
            label="Seed image"
            hint={profile.show.examples ? 'Upload your own or pick a bundled scene above' : 'Upload a real photo'}
          >
            <div className="flex items-center gap-3">
              <Button
                type="button"
                variant="default"
                size="sm"
                onClick={() => imageInputRef.current?.click()}
                disabled={isGenerating}
              >
                <Upload className="h-3.5 w-3.5" />
                Upload image
              </Button>
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleImageSelect}
              />
              {(imageFile || selectedExampleId) && (
                <span className="text-xs text-white/60">
                  {imageFile ? imageFile.name : `Bundled example ${selectedExampleId}`}
                </span>
              )}
            </div>
            {imagePreview && (
              <div className="mt-3">
                <img
                  src={imagePreview}
                  alt="Seed preview"
                  className="max-h-40 rounded-lg border border-game-border"
                />
              </div>
            )}
          </Field>

          <Field label="Prompt">
            <textarea
              className={`${inputCls} resize-none leading-relaxed`}
              rows={3}
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              disabled={isGenerating}
              placeholder="Describe the video you want to generate…"
            />
          </Field>

          {/* Camera control — simple toggle + optional upload */}
          {profile.show.camera && (
          <Field label="Camera control" hint="Bundled examples with 📷 carry their own poses">
            <div className="flex gap-2">
              {(['none', 'upload'] as const).map(m => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setCameraMode(m)}
                  disabled={isGenerating}
                  className={`px-3 py-1.5 text-xs font-semibold uppercase tracking-wider rounded-md border transition-colors
                    ${cameraMode === m
                      ? 'bg-game-accent/20 border-game-accent text-white'
                      : 'bg-transparent border-white/10 text-white/50 hover:text-white hover:border-white/25'}`}
                >
                  {m === 'none' ? 'No camera' : 'Upload .npy'}
                </button>
              ))}
            </div>
            {cameraMode === 'upload' && (
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
                {[
                  { label: 'poses.npy', hint: '[T, 4, 4]', file: posesFile, setFile: setPosesFile, ref: posesInputRef },
                  { label: 'intrinsics.npy', hint: '[T, 4]', file: intrinsicsFile, setFile: setIntrinsicsFile, ref: intrinsicsInputRef },
                ].map(f => (
                  <div key={f.label} className="flex items-center gap-2">
                    <Button
                      type="button"
                      variant="default"
                      size="sm"
                      onClick={() => f.ref.current?.click()}
                      disabled={isGenerating}
                    >
                      <Upload className="h-3.5 w-3.5" />
                      {f.file ? f.file.name : f.label}
                    </Button>
                    <span className="text-[11px] text-white/30">{f.hint}</span>
                    <input
                      ref={f.ref}
                      type="file"
                      accept=".npy"
                      className="hidden"
                      onChange={e => f.setFile(e.target.files?.[0] || null)}
                    />
                  </div>
                ))}
              </div>
            )}
          </Field>
          )}

          {/* DMD "fast mode" — lyra-2 only. Swaps to the 4-step distilled
              sampler (~11× faster, slight quality trade-off). When on, the
              sampling-steps control is ignored by the runner, so disable it. */}
          {profile.show.dmd && (
          <Field label="Fast mode (DMD)" hint="4-step distilled sampler · ~11× faster">
            <button
              type="button"
              onClick={() => setUseDmd(v => !v)}
              disabled={isGenerating}
              className={`px-3 py-1.5 text-xs font-semibold uppercase tracking-wider rounded-md border transition-colors
                ${useDmd
                  ? 'bg-game-accent/20 border-game-accent text-white'
                  : 'bg-transparent border-white/10 text-white/50 hover:text-white hover:border-white/25'}`}
            >
              {useDmd ? 'Fast mode: on' : 'Fast mode: off'}
            </button>
          </Field>
          )}

          {/* Advanced collapsible */}
          <details className="group rounded-lg bg-black/20 border border-game-border">
            <summary className="flex items-center gap-2 cursor-pointer select-none px-4 py-2.5 text-[11px] font-bold tracking-[0.14em] uppercase text-white/40 hover:text-white/70">
              <ChevronDown className="h-3.5 w-3.5 transition-transform group-open:rotate-180" />
              Advanced settings
            </summary>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 px-4 pb-4">
              {profile.show.frames && (
              <Field label="Frames" hint={profile.framesHint}>
                <input type="number" className={inputCls} value={frameNum}
                  onChange={e => setFrameNum(e.target.value)} disabled={isGenerating} />
              </Field>
              )}
              {profile.show.size && (
              <Field label="Resolution">
                <select className={inputCls} value={size}
                  onChange={e => setSize(e.target.value)} disabled={isGenerating}>
                  {SIZE_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </Field>
              )}
              <Field label="Sampling steps" hint={useDmd ? 'ignored in fast mode' : undefined}>
                <input type="number" className={inputCls} value={samplingSteps}
                  onChange={e => setSamplingSteps(e.target.value)} disabled={isGenerating || useDmd} />
              </Field>
              {profile.show.guidance && (
              <Field label="Guidance">
                <input type="number" step="0.1" className={inputCls} value={guideScale}
                  onChange={e => setGuideScale(e.target.value)} disabled={isGenerating} />
              </Field>
              )}
              <Field label="Seed" hint="-1 = random">
                <input type="number" className={inputCls} value={seed}
                  onChange={e => setSeed(e.target.value)} disabled={isGenerating} />
              </Field>
            </div>
          </details>

          {/* Generate button + errors */}
          <div className="flex items-center gap-4">
            <Button variant="glow" size="lg" onClick={handleGenerate} disabled={!canGenerate}>
              {isGenerating
                ? <><Loader2 className="h-5 w-5 animate-spin" />Generating…</>
                : <><Sparkles className="h-5 w-5" />Generate video</>}
            </Button>
            {error && (
              <span className="flex items-center gap-2 text-sm text-red-300">
                <AlertCircle className="h-4 w-4" />
                {error}
              </span>
            )}
          </div>
        </div>
      </Panel>

      {/* Output */}
      {(isGenerating || videoUrl) && (
        <Panel
          title="Output"
          right={videoUrl ? (
            <a
              href={videoUrl}
              download="lingbot_output.mp4"
              className="inline-flex items-center gap-1.5 text-xs font-semibold tracking-wider uppercase text-white/60 hover:text-game-accent transition-colors"
            >
              <Download className="h-3.5 w-3.5" /> Download
            </a>
          ) : null}
        >
          {isGenerating && !videoUrl && (
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2 text-sm text-white/70">
                <Loader2 className="h-4 w-4 animate-spin text-game-accent" />
                Generating — elapsed {elapsedSec}s
                <span className="text-white/25">{profile.progressHint}</span>
              </div>
              <div className="h-1 bg-white/10 rounded overflow-hidden">
                <div className="h-full w-1/3 bg-game-accent animate-[slide_2s_ease-in-out_infinite_alternate]" />
              </div>
            </div>
          )}
          {videoUrl && (
            <div className="flex flex-col items-center gap-3">
              <Film className="h-4 w-4 text-white/30 hidden" />
              <video
                src={videoUrl}
                controls
                autoPlay
                loop
                className="w-full max-h-[500px] rounded-lg border border-game-border bg-black"
              />
            </div>
          )}
        </Panel>
      )}
    </div>
  );
};
