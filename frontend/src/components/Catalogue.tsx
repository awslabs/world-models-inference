// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from 'react';
import { ArrowLeft, ExternalLink, Sparkles } from 'lucide-react';
import { LingbotGenerator } from '@/components/LingbotGenerator';
import {
  CARTRIDGES,
  CARTRIDGE_TYPES,
  Cartridge,
  CartridgeType,
  TYPE_LABEL,
} from '@/data/cartridges';

// ─── Card preview (hue-tinted gradient, no procedural canvas) ──────
const Preview = ({ hue }: { hue: number }) => (
  <div
    className="relative h-[170px] overflow-hidden"
    style={{
      background: `
        radial-gradient(at 30% 20%, hsl(${hue} 65% 45% / 0.9) 0%, transparent 60%),
        radial-gradient(at 70% 80%, hsl(${(hue + 35) % 360} 55% 35% / 0.8) 0%, transparent 65%),
        linear-gradient(135deg, hsl(${hue} 30% 12%), hsl(${(hue + 60) % 360} 30% 8%))
      `,
    }}
  >
    {/* subtle animated diagonal shimmer */}
    <div
      className="absolute inset-0 opacity-30 mix-blend-overlay"
      style={{
        backgroundImage: 'linear-gradient(135deg, transparent 40%, rgba(255,255,255,0.12) 50%, transparent 60%)',
        backgroundSize: '200% 200%',
        animation: 'lobby-bg-drift 20s ease-in-out infinite alternate',
      }}
    />
  </div>
);

// ─── Small status/type badges ──────────────────────────────────────
const StatusBadge = ({ status }: { status: Cartridge['status'] }) => {
  const cls = status === 'ready'
    ? 'text-emerald-300 bg-emerald-400/10 border-emerald-400/30'
    : status === 'warming'
    ? 'text-amber-300 bg-amber-400/10 border-amber-400/30'
    : 'text-slate-300 bg-slate-400/10 border-slate-400/30';
  const label = status === 'ready' ? 'ready' : status === 'warming' ? 'warming' : 'cold';
  return (
    <span className={`text-[10px] font-bold tracking-[0.12em] uppercase px-2 py-0.5 rounded-full border ${cls}`}>
      {label}
    </span>
  );
};

const TypeBadge = ({ type }: { type: CartridgeType }) => {
  const cls: Record<CartridgeType, string> = {
    'real-time':        'bg-orange-500/80 text-orange-50',
    'video-generation': 'bg-blue-500/80 text-blue-50',
    'representation':   'bg-indigo-500/80 text-indigo-50',
    '3dgs':             'bg-purple-500/80 text-purple-50',
  };
  return (
    <span className={`text-[10px] font-bold tracking-[0.12em] uppercase px-2 py-0.5 rounded ${cls[type]}`}>
      {TYPE_LABEL[type]}
    </span>
  );
};

// ─── License line ──────────────────────────────────────────────────
// Renders the cartridge's weights license; restricted weights get an amber
// warning tint. Links to the license/model card when a URL is provided.
const LicenseLine = ({ cart }: { cart: Cartridge }) => {
  const restricted = !!cart.restricted;
  const cls = restricted ? 'text-amber-300/80' : 'text-white/40';
  const label = `${restricted ? '⚠ ' : ''}${cart.license}${restricted ? ' — restricted' : ''}`;
  return (
    <div className={`text-[11px] leading-snug ${cls}`}>
      {cart.licenseUrl ? (
        <a
          href={cart.licenseUrl}
          target="_blank"
          rel="noreferrer"
          onClick={e => e.stopPropagation()}
          className="inline-flex items-center gap-1 hover:underline"
        >
          {label} <ExternalLink className="h-3 w-3" />
        </a>
      ) : (
        <span>{label}</span>
      )}
    </div>
  );
};

// ─── Single cartridge card ─────────────────────────────────────────
const Card = ({ cart, onClick }: { cart: Cartridge; onClick: () => void }) => (
  <button
    onClick={onClick}
    className="text-left bg-game-surface/60 backdrop-blur-sm border border-game-border rounded-xl overflow-hidden transition-all duration-200 hover:-translate-y-1 hover:border-game-accent/50 hover:shadow-[0_12px_40px_-12px_rgba(83,159,229,0.4)] flex flex-col"
  >
    <div className="relative">
      <Preview hue={cart.hue} />
      <div className="absolute top-2 left-2"><TypeBadge type={cart.type} /></div>
      <div className="absolute top-2 right-2"><StatusBadge status={cart.status} /></div>
    </div>
    <div className="p-4 flex flex-col gap-3 flex-1">
      <div>
        <h3 className="text-white font-semibold tracking-wide">{cart.name}</h3>
        <p className="text-[13px] text-white/50 mt-1 leading-relaxed line-clamp-2">{cart.tagline}</p>
      </div>
      <div className="flex flex-wrap gap-1.5 text-[11px] font-medium">
        {[cart.instance, cart.gpu, cart.fps ? `~${cart.fps} fps` : 'async', cart.model].map(s => (
          <span key={s} className="px-2 py-0.5 rounded bg-white/5 text-white/60 border border-white/5">
            {s}
          </span>
        ))}
      </div>
      {cart.license && <LicenseLine cart={cart} />}
      <div className="flex items-center justify-between mt-auto pt-2 border-t border-white/5">
        <div className="flex gap-1.5">
          {cart.deploy.map(t => (
            <span
              key={t}
              className={`text-[10px] font-bold tracking-[0.12em] uppercase px-2 py-0.5 rounded-full border
                ${t === 'ec2'
                  ? 'text-green-300/80 border-green-400/20'
                  : 'text-blue-300/80 border-blue-400/20'}`}
            >
              {t === 'ec2' ? 'EC2' : 'SageMaker'}
            </span>
          ))}
        </div>
        <span className="text-[13px] text-white/50 font-mono">${cart.price.toFixed(2)}/s</span>
      </div>
    </div>
  </button>
);

// ─── Full-page detail view for non-wired cartridges ────────────────
const ComingSoon = ({ cart, onBack }: { cart: Cartridge; onBack: () => void }) => (
  <div className="max-w-2xl mx-auto flex flex-col gap-5">
    <button
      onClick={onBack}
      className="self-start flex items-center gap-1.5 text-sm text-white/50 hover:text-white transition-colors"
    >
      <ArrowLeft className="h-4 w-4" /> Back to catalogue
    </button>
    <div className="rounded-xl bg-game-surface/70 border border-game-border overflow-hidden">
      <Preview hue={cart.hue} />
      <div className="p-6 flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold text-white">{cart.name}</h1>
          <StatusBadge status={cart.status} />
          <TypeBadge type={cart.type} />
        </div>
        <p className="text-white/60 leading-relaxed">{cart.tagline}</p>
        <div className="rounded-lg bg-amber-400/5 border border-amber-400/20 p-4 text-[13px] text-amber-100/80 leading-relaxed">
          <strong className="text-amber-200">Not wired up yet.</strong>
          {' '}{cart.notes ?? 'Scaffold exists in inference/models/; inference handler not implemented.'}
        </div>
        <div className="flex flex-wrap items-center gap-3 text-[13px] text-white/50">
          <span>Instance: <span className="text-white/80">{cart.instance}</span></span>
          <span>GPU: <span className="text-white/80">{cart.gpu}</span></span>
          <span>Model: <span className="text-white/80">{cart.model}</span></span>
        </div>
        {cart.upstream && (
          <a
            href={cart.upstream}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 text-sm text-game-accent hover:text-game-accent/80"
          >
            <ExternalLink className="h-4 w-4" /> Upstream repo
          </a>
        )}
        <div className="pt-2 border-t border-white/5 text-[12px] text-white/40">
          To wire this up, use <code className="text-white/60 font-mono">lingbot-fast</code> as a reference:
          its handler is at <code className="text-white/60 font-mono">inference/models/lingbot-fast/runner.py</code>.
        </div>
      </div>
    </div>
  </div>
);

// ─── Catalogue root ────────────────────────────────────────────────
export const Catalogue = ({ endpointUrl }: { endpointUrl?: string }) => {
  const [filter, setFilter] = useState<CartridgeType | 'all'>('all');
  const [selected, setSelected] = useState<Cartridge | null>(null);

  const filtered = useMemo(
    () => filter === 'all' ? CARTRIDGES : CARTRIDGES.filter(c => c.type === filter),
    [filter],
  );

  // Detail view: a ready async cartridge → route to the batch generator UI
  // (LingbotGenerator drives the /generate → /status → /result flow and adapts
  // its controls per-cartridge via gen-profiles); otherwise → ComingSoon.
  // fps === 0 marks the async batch cartridges; real-time ones (fps > 0) have
  // their own WebSocket UI and aren't served here.
  if (selected) {
    if (selected.status === 'ready' && selected.fps === 0) {
      return (
        <div className="flex flex-col gap-5">
          <button
            onClick={() => setSelected(null)}
            className="self-start flex items-center gap-1.5 text-sm text-white/50 hover:text-white transition-colors"
          >
            <ArrowLeft className="h-4 w-4" /> Back to catalogue
          </button>
          <LingbotGenerator cartridgeId={selected.id} endpointUrl={endpointUrl} />
        </div>
      );
    }
    return <ComingSoon cart={selected} onBack={() => setSelected(null)} />;
  }

  return (
    <div className="flex flex-col gap-10">
      {/* Top nav bar */}
      <div className="flex items-center justify-between pb-3 border-b border-white/5">
        <div className="flex items-baseline gap-8">
          <span className="font-bold text-white tracking-tight">World Foundry</span>
          <nav className="flex items-center gap-5">
            <span className="text-sm text-white border-b-2 border-game-accent pb-3 -mb-3">Catalogue</span>
            <span className="text-sm text-white/40 cursor-not-allowed">Docs</span>
          </nav>
        </div>
        {endpointUrl && (
          <span className="hidden md:inline text-[11px] text-white/30 font-mono">{endpointUrl}</span>
        )}
      </div>

      {/* Hero + filters */}
      <div>
        <h1 className="text-[clamp(26px,3.5vw,40px)] font-extrabold tracking-tight text-white">
          Any world model. One platform.
        </h1>
        <p className="text-white/50 mt-2">
          Pick a cartridge. We handle the GPUs, the streaming, the scaling.
        </p>
        <div className="flex flex-wrap gap-2 mt-5">
          {CARTRIDGE_TYPES.map(t => (
            <button
              key={t.value}
              onClick={() => setFilter(t.value)}
              className={`px-4 py-1.5 rounded-full text-[13px] border transition-all cursor-pointer
                ${filter === t.value
                  ? 'bg-game-accent/15 border-game-accent text-white'
                  : 'bg-transparent border-white/10 text-white/50 hover:text-white hover:border-white/30'}`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-[repeat(auto-fill,minmax(290px,1fr))] gap-5">
        {filtered.map(cart => (
          <Card key={cart.id} cart={cart} onClick={() => setSelected(cart)} />
        ))}
      </div>

      {/* Footer note */}
      <div className="text-center text-[12px] text-white/25 pt-4">
        <span className="inline-flex items-center gap-1.5">
          <Sparkles className="h-3 w-3" />
          Only <span className="text-emerald-300/70 font-semibold">ready</span> cartridges are wired end-to-end. Others are scaffolded — see <code className="font-mono text-white/40">inference/models/{"<id>"}/endpoint.yaml</code>.
        </span>
      </div>
    </div>
  );
};
