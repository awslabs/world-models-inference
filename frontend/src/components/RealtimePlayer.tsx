// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Play view for real-time cartridges (Matrix Game 3.0). Speaks the endpoint's
// actual session protocol, which is deliberately minimal:
//
//   client -> server : one 12-byte binary action per tick
//                      <uint32 LE key bitflags><float32 pitch><float32 yaw>
//                      (bits 0..3 = W,S,A,D — see inference/models/matrix-game-3/actions.py)
//   server -> client : one binary message per frame, each a bare JPEG
//
// Binary only, on purpose: lib/app.py reads the socket with receive_bytes(),
// so this component does NOT reuse useWebSocket — that hook sends a JSON text
// ping every second. It connects to its own origin (/ws); `./deploy.sh ui`
// starts the vite dev proxy, which forwards the socket to the ALB and injects
// the bearer token server-side, so no token ever reaches the browser.

import { useEffect, useRef, useState } from 'react';
import { Play, X } from 'lucide-react';
import { useFrameBuffer } from '@/hooks/useFrameBuffer';
import type { Cartridge } from '@/data/cartridges';

// Native output is 832x480 (HEIGHT*WIDTH "480*832" on the server).
const FRAME_W = 832, FRAME_H = 480;

// Matches CAM_VALUE in actions.py — one camera step per axis per tick. The
// server clamps to this anyway; sending the exact value keeps motion honest.
const CAM_STEP = 0.1;

const KEY_BITS: Record<string, number> = { w: 1 << 0, s: 1 << 1, a: 1 << 2, d: 1 << 3 };

function packAction(keys: Set<string>, pitch: number, yaw: number): ArrayBuffer {
  const buf = new ArrayBuffer(12);
  const view = new DataView(buf);
  let flags = 0;
  for (const [k, bit] of Object.entries(KEY_BITS)) if (keys.has(k)) flags |= bit;
  view.setUint32(0, flags, true);
  view.setFloat32(4, pitch, true);
  view.setFloat32(8, yaw, true);
  return buf;
}

export const RealtimePlayer = ({ cart, onBack }: { cart: Cartridge; onBack: () => void }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const keysRef = useRef<Set<string>>(new Set());
  const frameCountRef = useRef(0);

  const [fps, setFps] = useState(0);
  const [frames, setFrames] = useState(0);
  const [phase, setPhase] = useState<'idle' | 'connecting' | 'warming' | 'live' | 'ended'>('idle');
  // Mirrors keysRef for the keypad highlight. The ref alone cannot drive it: it
  // is mutated in place, so React never re-renders on a key press.
  const [held, setHeld] = useState<Set<string>>(new Set());

  // The socket opens on a click, never on mount. A session drives the whole GPU
  // cluster and only one can exist, but React StrictMode mounts twice in dev, so
  // connecting on mount opened two sockets milliseconds apart — the second
  // preempted the first and both died at 0 frames. One click is one session.
  const [started, setStarted] = useState(false);

  const { pushFrame, startPlayback, stopPlayback } = useFrameBuffer({ canvasRef, onFpsUpdate: setFps });

  useEffect(() => {
    if (!started) return;
    setPhase('connecting');
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${window.location.host}/ws`);
    ws.binaryType = 'blob';
    wsRef.current = ws;

    const sendAction = () => {
      if (ws.readyState !== WebSocket.OPEN) return;
      const k = keysRef.current;
      const pitch = (k.has('i') ? CAM_STEP : 0) + (k.has('k') ? -CAM_STEP : 0);
      const yaw = (k.has('l') ? CAM_STEP : 0) + (k.has('j') ? -CAM_STEP : 0);
      ws.send(packAction(k, pitch, yaw));
    };

    ws.onopen = () => {
      setPhase('warming');
      startPlayback();
      // Send one action straight away. The server reads the action buffer as
      // soon as the session opens; if nothing has arrived yet it reads an empty
      // action, which ends the session before the first chunk.
      sendAction();
      // Heartbeat only. The real input path is sendAction() straight off the
      // key event: the server keeps just the latest action, so a tap that began
      // and ended between two ticks would otherwise never be sent at all.
      const tick = window.setInterval(sendAction, 100);
      ws.addEventListener('close', () => clearInterval(tick));
    };

    ws.onmessage = (event) => {
      if (!(event.data instanceof Blob)) return;
      frameCountRef.current += 1;
      setFrames(frameCountRef.current);
      if (frameCountRef.current === 1) setPhase('live');
      pushFrame(event.data);
    };

    ws.onclose = () => { stopPlayback(); setPhase('ended'); };
    ws.onerror = () => { stopPlayback(); setPhase('ended'); };

    const isTyping = () => {
      const t = document.activeElement?.tagName;
      return t === 'INPUT' || t === 'TEXTAREA';
    };
    const down = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { ws.close(); onBack(); return; }
      if (isTyping()) return;
      const k = e.key.toLowerCase();
      if ('wasdijkl'.includes(k)) {
        e.preventDefault();
        if (keysRef.current.has(k)) return; // key repeat — nothing changed
        keysRef.current.add(k);
        setHeld(new Set(keysRef.current));
        sendAction();
      }
    };
    const up = (e: KeyboardEvent) => {
      if (keysRef.current.delete(e.key.toLowerCase())) {
        setHeld(new Set(keysRef.current));
        sendAction();
      }
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);

    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
      stopPlayback();
      ws.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [started]);

  return (
    <div className="fixed inset-0 bg-black flex flex-col z-50">
      {/* HUD */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-5 py-3 bg-gradient-to-b from-black/70 to-transparent">
        <button
          onClick={() => { wsRef.current?.close(); onBack(); }}
          className="bg-white/8 border border-white/15 text-white/70 px-4 py-1.5 rounded-md text-xs font-semibold tracking-wider cursor-pointer hover:bg-red-500/15 hover:border-red-500/40 hover:text-white transition-all flex items-center gap-1.5"
        >
          <X className="h-3.5 w-3.5" /> EXIT
        </button>
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-white/80">{cart.name}</span>
          {phase === 'connecting' && <span className="text-xs font-semibold tracking-wider text-white/60 bg-white/5 px-3 py-1 rounded-full">Connecting…</span>}
          {phase === 'warming' && <span className="text-xs font-semibold tracking-wider text-amber-300 bg-amber-500/10 px-3 py-1 rounded-full animate-pulse">Generating first frame…</span>}
          {phase === 'live' && <span className="text-xs font-semibold tracking-wider text-emerald-300 bg-emerald-500/10 px-3 py-1 rounded-full">● LIVE</span>}
          {phase === 'ended' && <span className="text-xs font-semibold tracking-wider text-red-300 bg-red-500/10 px-3 py-1 rounded-full">■ ENDED</span>}
        </div>
        <div className="flex items-center gap-3 text-xs text-white/40 tabular-nums">
          <span>{fps} FPS</span>
          <span>{frames} frames</span>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 flex items-center justify-center overflow-hidden relative">
        <canvas
          ref={canvasRef}
          width={FRAME_W}
          height={FRAME_H}
          className="block max-w-[100vw] max-h-[calc(100vh-90px)] w-auto h-auto bg-black"
        />

        {(phase === 'idle' || phase === 'ended') && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-5 bg-black/80">
            <button
              onClick={() => { frameCountRef.current = 0; setFrames(0); setStarted(false); setTimeout(() => setStarted(true), 0); }}
              className="bg-emerald-500 hover:bg-emerald-400 text-black font-bold tracking-wider px-10 py-4 rounded-lg text-base cursor-pointer transition-colors flex items-center gap-2"
            >
              <Play className="h-5 w-5" /> {phase === 'ended' ? 'PLAY AGAIN' : 'START'}
            </button>
            <p className="text-xs text-white/45 max-w-sm text-center leading-relaxed">
              {phase === 'ended'
                ? 'Session ended. Only one session can run at a time — if someone else is playing, starting here takes over.'
                : 'Takes ~15s to generate the first frame. Then steer with WASD and IJKL.'}
            </p>
          </div>
        )}

        {phase === 'connecting' && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/70 text-sm text-white/60">
            Connecting…
          </div>
        )}

        {phase === 'warming' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/70">
            <div className="h-8 w-8 rounded-full border-2 border-emerald-400/30 border-t-emerald-400 animate-spin" />
            <p className="text-sm text-white/60">Generating first frame…</p>
          </div>
        )}
      </div>

      {/* Directional keypads — right-hand side, laid out like the keys on the
          keyboard (inverted T) so each key's direction is visible at a glance. */}
      <div className="absolute right-5 bottom-24 z-10 flex flex-col gap-5 items-center pointer-events-none">
        <KeyPad
          label="MOVE"
          held={held}
          accent="border-emerald-400 bg-emerald-500/20 text-emerald-200"
          keys={[
            { k: 'w', arrow: '↑' },
            { k: 'a', arrow: '←' },
            { k: 's', arrow: '↓' },
            { k: 'd', arrow: '→' },
          ]}
        />
        <KeyPad
          label="LOOK"
          held={held}
          accent="border-pink-400 bg-pink-500/20 text-pink-200"
          keys={[
            { k: 'i', arrow: '↑' },
            { k: 'j', arrow: '←' },
            { k: 'k', arrow: '↓' },
            { k: 'l', arrow: '→' },
          ]}
        />
      </div>

      {/* Footer note */}
      <div className="flex items-center justify-center py-3 bg-gradient-to-t from-black/60 to-transparent text-[12px] text-white/35">
        Actions apply at chunk boundaries — steer, don't twitch. ESC to exit.
      </div>
    </div>
  );
};

/* ── Directional keypad (inverted T, like the physical keys) ─────────── */

interface PadKey { k: string; arrow: string }

const KeyCap = ({ pk, held, accent }: { pk: PadKey; held: boolean; accent: string }) => (
  <kbd
    className={`flex flex-col items-center justify-center w-11 h-11 gap-0.5 rounded-md border transition-colors leading-none
      ${held ? accent : 'border-white/20 bg-black/40 text-white/70'}`}
  >
    <span className="text-[13px] font-bold uppercase">{pk.k}</span>
    <span className="text-[13px] leading-none">{pk.arrow}</span>
  </kbd>
);

const KeyPad = ({ label, keys, held, accent }: {
  label: string;
  keys: [PadKey, PadKey, PadKey, PadKey]; // [up, left, down, right]
  held: Set<string>;
  accent: string;
}) => (
  <div className="flex flex-col items-center gap-1 rounded-xl bg-black/45 border border-white/10 px-3 pt-3 pb-2 backdrop-blur-sm">
    <KeyCap pk={keys[0]} held={held.has(keys[0].k)} accent={accent} />
    <div className="flex gap-1">
      <KeyCap pk={keys[1]} held={held.has(keys[1].k)} accent={accent} />
      <KeyCap pk={keys[2]} held={held.has(keys[2].k)} accent={accent} />
      <KeyCap pk={keys[3]} held={held.has(keys[3].k)} accent={accent} />
    </div>
    <span className="text-[10px] font-semibold tracking-[0.18em] text-white/35 mt-1">{label}</span>
  </div>
);
