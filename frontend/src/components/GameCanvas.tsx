// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';
import { X, Gamepad2 } from 'lucide-react';
import { getConfig } from '@/services/config';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useFrameBuffer } from '@/hooks/useFrameBuffer';
import { useGamepad } from '@/hooks/useGamepad';
import type { PendingStart } from '@/App';

const FRAME_W = 1280, FRAME_H = 720, AXIS_T = 0.4;
const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
const pick = (a: number, b: number) => (Math.abs(a) > Math.abs(b) ? a : b);

// Inject keyframes
if (typeof document !== 'undefined' && !document.getElementById('placeholder-zoom-style')) {
  const style = document.createElement('style');
  style.id = 'placeholder-zoom-style';
  style.textContent = `@keyframes placeholder-zoom {
    0% { transform: scale(1.0); filter: grayscale(0.4) brightness(0.65); }
    100% { transform: scale(1.6); filter: grayscale(0) brightness(1.0); }
  }`;
  document.head.appendChild(style);
}

interface GameCanvasProps {
  pendingStart: PendingStart;
  onBack: () => void;
  addSession: (session: { sceneName: string; thumbnailUrl: string; seedImageS3Key: string }) => Promise<string>;
}

export const GameCanvas = ({ pendingStart, onBack, addSession }: GameCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const keysRef = useRef<Set<string>>(new Set());
  const startRef = useRef(true);

  const [playing, setPlaying] = useState(false);
  const [waiting, setWaiting] = useState(true);
  const waitingRef = useRef(true);
  const [fps, setFps] = useState(0);
  const [lStick, setLStick] = useState<[number, number]>([0, 0]);
  const [rStick, setRStick] = useState<[number, number]>([0, 0]);
  const [secs, setSecs] = useState(0);

  const pendingImageDataRef = useRef<string | null>(pendingStart.imageData || null);
  const pendingImageS3KeyRef = useRef<string | null>(pendingStart.imageS3Key || null);
  const activeSessionIdRef = useRef<string | null>(null);
  const predictionRef = useRef({ scaleX: 1, scaleY: 1, tx: 0, ty: 0 });
  const lastFrameTimeRef = useRef(0);
  const frameCountRef = useRef(0);
  const [showingPlaceholder, setShowingPlaceholder] = useState(false);
  const [sessionEnded, setSessionEnded] = useState(false);

  const cfg = getConfig();
  const { pushFrame, startPlayback, stopPlayback } = useFrameBuffer({ canvasRef, onFpsUpdate: setFps });

  /** Handle canvas zoom animation on first few frames, then render via Blob. */
  const handleFrame = (blob: Blob) => {
    lastFrameTimeRef.current = performance.now();
    frameCountRef.current += 1;
    if (frameCountRef.current === 1 && canvasRef.current) {
      canvasRef.current.style.animation = 'placeholder-zoom 30s ease-out forwards';
      setShowingPlaceholder(true);
    } else if (frameCountRef.current === 2 && canvasRef.current) {
      const cs = getComputedStyle(canvasRef.current);
      canvasRef.current.style.animation = 'none';
      canvasRef.current.style.filter = 'none';
      canvasRef.current.style.transform = cs.transform;
      canvasRef.current.style.transition = 'transform 0.5s ease-in, filter 0.3s linear';
      canvasRef.current.offsetHeight; // eslint-disable-line
      canvasRef.current.style.transform = 'scale(1)';
      setShowingPlaceholder(false);
    } else if (frameCountRef.current >= 3 && canvasRef.current?.style.transition) {
      canvasRef.current.style.transition = '';
    }
    predictionRef.current = { scaleX: 1, scaleY: 1, tx: 0, ty: 0 };
    if (waitingRef.current) { waitingRef.current = false; setWaiting(false); setSecs(0); }
    pushFrame(blob);
  };

  const { gamepadConnected, pollGamepad } = useGamepad();
  const { isConnected, latency, connect, disconnect, sendMessage } = useWebSocket({
    url: cfg.websocketUrl,
    // Binary frames = raw JPEG from server (33% smaller than base64 JSON)
    onBinaryMessage: handleFrame,
    onMessage: (m) => {
      if (m.type === 'connected' && startRef.current) {
        const msg: any = { type: 'start' };
        if (pendingImageS3KeyRef.current) { msg.image_s3_key = pendingImageS3KeyRef.current; pendingImageS3KeyRef.current = null; }
        else if (pendingImageDataRef.current) { msg.image_data = pendingImageDataRef.current; pendingImageDataRef.current = null; }
        sendMessage(msg);
        startRef.current = false;
      } else if (m.type === 'started') { setPlaying(true); startPlayback(); }
      else if (m.type === 'error') console.error('Server:', m.message);
    },
    onDisconnect: () => {
      stopPlayback(); setPlaying(false); waitingRef.current = false; setWaiting(false);
    },
  });

  // Auto-connect
  useEffect(() => {
    const init = async () => {
      const sid = await addSession({ sceneName: pendingStart.sceneName, thumbnailUrl: pendingStart.thumbnailUrl, seedImageS3Key: pendingStart.seedImageS3Key });
      activeSessionIdRef.current = sid;
      await connect();
    };
    init();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const doStop = () => {
    stopPlayback(); disconnect(); setPlaying(false); waitingRef.current = false; setWaiting(false);
    onBack();
  };

  // Session timer (display only, not persisted)
  useEffect(() => {
    if (!playing || waiting) return;
    const id = setInterval(() => setSecs(s => s + 1), 1000);
    return () => clearInterval(id);
  }, [playing, waiting]);

  // Frame starvation detection — if no frame for 3s while playing, session ended
  useEffect(() => {
    if (!playing || waiting || sessionEnded) return;
    const id = setInterval(() => {
      if (frameCountRef.current >= 10 && lastFrameTimeRef.current > 0 && performance.now() - lastFrameTimeRef.current > 3000) {
        setSessionEnded(true);
        stopPlayback();
        // Dim the canvas
        if (canvasRef.current) {
          canvasRef.current.style.transition = 'filter 1s ease';
          canvasRef.current.style.filter = 'brightness(0.3) grayscale(0.5)';
          canvasRef.current.style.transform = 'scale(1)';
        }
      }
    }, 500);
    return () => clearInterval(id);
  }, [playing, waiting, sessionEnded, stopPlayback]);

  // Keyboard
  useEffect(() => {
    const isTyping = () => { const t = document.activeElement?.tagName; return t === 'INPUT' || t === 'TEXTAREA'; };
    const dn = (e: KeyboardEvent) => { if (e.key === 'Escape') { doStop(); return; } if (isTyping()) return; if ('wasdjikl'.includes(e.key.toLowerCase())) e.preventDefault(); keysRef.current.add(e.key.toLowerCase()); };
    const up = (e: KeyboardEvent) => { if (isTyping()) { keysRef.current.clear(); return; } keysRef.current.delete(e.key.toLowerCase()); };
    window.addEventListener('keydown', dn); window.addEventListener('keyup', up);
    return () => { window.removeEventListener('keydown', dn); window.removeEventListener('keyup', up); };
  });

  // Input loop
  useEffect(() => {
    const id = setInterval(() => {
      const k = keysRef.current, g = pollGamepad();
      const lx = pick(g.leftStick[0], (k.has('d') ? 1 : 0) - (k.has('a') ? 1 : 0));
      const ly = pick(g.leftStick[1], (k.has('s') ? 1 : 0) - (k.has('w') ? 1 : 0));
      const rx = pick(g.rightStick[0], (k.has('l') ? 1 : 0) - (k.has('j') ? 1 : 0));
      const ry = pick(g.rightStick[1], (k.has('k') ? 1 : 0) - (k.has('i') ? 1 : 0));
      setLStick([lx, ly]); setRStick([rx, ry]);
      const canControl = playing && !sessionEnded && frameCountRef.current >= 2;
      if (canControl) {
        const buttons: string[] = [];
        if (ly < -AXIS_T) buttons.push('W'); if (ly > AXIS_T) buttons.push('S');
        if (lx < -AXIS_T) buttons.push('A'); if (lx > AXIS_T) buttons.push('D');
        sendMessage({ type: 'control', buttons, mouse_dx: Math.round(rx * 100), mouse_dy: Math.round(ry * 100) });
        const dt = performance.now() - lastFrameTimeRef.current;
        if (dt > 80 && canvasRef.current) {
          const p = predictionRef.current;
          if (ly < -AXIS_T) { p.scaleX += 0.002; p.scaleY += 0.002; }
          if (ly > AXIS_T) p.ty += 0.75;
          if (lx < -AXIS_T) p.tx += 1.5; if (lx > AXIS_T) p.tx -= 1.5;
          if (Math.abs(rx) > 0.1) p.tx -= rx * 2; if (Math.abs(ry) > 0.1) p.ty -= ry;
          p.scaleX = Math.max(1, Math.min(1.1, p.scaleX)); p.scaleY = Math.max(1, Math.min(1.1, p.scaleY));
          p.tx = Math.max(-30, Math.min(30, p.tx)); p.ty = Math.max(-15, Math.min(15, p.ty));
          if (Math.abs(lx) > AXIS_T || Math.abs(ly) > AXIS_T || Math.abs(rx) > 0.1 || Math.abs(ry) > 0.1) {
            canvasRef.current.style.filter = 'blur(0.7px)';
            canvasRef.current.style.transform = `scale(${p.scaleX},${p.scaleY}) translate(${p.tx}px,${p.ty}px)`;
          }
        }
      }
    }, 33);
    return () => clearInterval(id);
  }, [playing, sendMessage, pollGamepad]);

  return (
    <div className="fixed inset-0 bg-black flex flex-col z-50">
      {/* HUD */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-5 py-3 bg-gradient-to-b from-black/70 to-transparent pointer-events-none">
        <button
          onClick={doStop}
          className="pointer-events-auto bg-white/8 border border-white/15 text-white/70 px-4 py-1.5 rounded-md text-xs font-semibold tracking-wider cursor-pointer hover:bg-red-500/15 hover:border-red-500/40 hover:text-white transition-all flex items-center gap-1.5"
        >
          <X className="h-3.5 w-3.5" /> EXIT
        </button>
        <div className="flex items-center gap-2 pointer-events-auto">
          {waiting && <span className="text-xs font-semibold tracking-wider text-white/60 bg-white/5 px-3 py-1 rounded-full animate-[hud-pulse_1.5s_ease-in-out_infinite]">Connecting…</span>}
          {sessionEnded && <span className="text-xs font-semibold tracking-wider text-game-red bg-game-red/10 px-3 py-1 rounded-full">■ ENDED</span>}
          {playing && !waiting && !sessionEnded && <span className="text-xs font-semibold tracking-wider text-game-green bg-game-green/10 px-3 py-1 rounded-full">● LIVE</span>}
        </div>
        <div className="flex items-center gap-3 text-xs text-white/40 tabular-nums pointer-events-auto">
          {playing && <span>{fmt(secs)}</span>}
          <span>{fps} FPS</span>
          <span>{latency}ms</span>
          {gamepadConnected && <Gamepad2 className="h-4 w-4 text-game-green" />}
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-game-green shadow-[0_0_6px_theme(colors.game-green)]' : 'bg-game-red shadow-[0_0_6px_theme(colors.game-red)] animate-[hud-pulse_1.2s_ease-in-out_infinite]'}`} />
        </div>
      </div>

      {/* Canvas — overflow hidden clips the zoom animation */}
      <div className="flex-1 flex items-center justify-center overflow-hidden relative">
        <div className="overflow-hidden" style={{ lineHeight: 0 }}>
          <canvas ref={canvasRef} width={FRAME_W} height={FRAME_H}
            className="block max-w-[100vw] max-h-[calc(100vh-100px)] w-auto h-auto bg-black"
            style={{
              transition: 'filter 0.3s linear',
              ...(showingPlaceholder ? { animation: 'placeholder-zoom 30s ease-out forwards' } : {}),
            }}
          />
        </div>

        {/* Session ended overlay */}
        {sessionEnded && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-6 animate-[fadeIn_0.8s_ease-in] z-20">
            <div className="text-center">
              <p className="text-2xl font-bold text-white tracking-wide">Session Ended</p>
              <p className="text-sm text-white/40 mt-2">World generation limit reached</p>
            </div>
            <button
              onClick={doStop}
              className="bg-white/10 border border-white/20 text-white px-6 py-2.5 rounded-lg text-sm font-semibold tracking-wide cursor-pointer hover:bg-white/20 hover:border-white/30 transition-all"
            >
              ← Back to Lobby
            </button>
          </div>
        )}
      </div>

      {/* Sticks */}
      <div className="flex items-center justify-center py-2 gap-0 bg-gradient-to-t from-black/50 to-transparent">
        <Stick x={lStick[0]} y={lStick[1]} color="#818cf8" label="WASD" />
        <div className="w-16" />
        <Stick x={rStick[0]} y={rStick[1]} color="#f472b6" label="IJKL" />
      </div>
    </div>
  );
};

/* ── Analog stick SVG ──────────────────────────────────────────── */
const S = 100, R = 40, D = 12, M = 26;
const Stick = ({ x, y, color, label }: { x: number; y: number; color: string; label: string }) => {
  const cx = S / 2, cy = S / 2, dx = cx + x * M, dy = cy + y * M;
  const on = Math.abs(x) > 0.01 || Math.abs(y) > 0.01;
  return (
    <div className="flex flex-col items-center gap-0.5">
      <svg width={S} height={S} viewBox={`0 0 ${S} ${S}`}>
        <circle cx={cx} cy={cy} r={R} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth={2} />
        <line x1={cx} y1={cy - R + 6} x2={cx} y2={cy + R - 6} stroke="rgba(255,255,255,0.05)" strokeWidth={1} />
        <line x1={cx - R + 6} y1={cy} x2={cx + R - 6} y2={cy} stroke="rgba(255,255,255,0.05)" strokeWidth={1} />
        {on && <circle cx={dx} cy={dy} r={D + 5} fill="none" stroke={color} strokeWidth={1} opacity={0.3} />}
        <circle cx={dx} cy={dy} r={D} fill={on ? color : 'rgba(255,255,255,0.12)'} stroke={on ? color : 'rgba(255,255,255,0.2)'} strokeWidth={on ? 2 : 1} />
      </svg>
      <span className="text-[10px] font-semibold tracking-[0.1em] text-white/20 uppercase">{label}</span>
    </div>
  );
};
