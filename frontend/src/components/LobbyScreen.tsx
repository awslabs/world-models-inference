// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';
import { Sparkles, LogOut } from 'lucide-react';
import { DEFAULT_SCENES, Scene } from '@/data/scenes';
import { Button } from '@/components/ui/button';
import { SceneGenerator } from '@/components/SceneGenerator';
import { SessionRecord } from '@/hooks/useSessionHistory';

interface LobbyScreenProps {
  user: { username: string; email?: string } | null;
  onStartScene: (scene: Scene) => void;
  onStartNovaScene: (s3Key: string, url: string, title?: string) => void;
  onSignOut: () => void;
  sessions: SessionRecord[];
  onClearHistory: () => void;
}

const fmtDate = (iso: string) => {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
    ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
};

export const LobbyScreen = ({
  user, onStartScene, onStartNovaScene, onSignOut, sessions, onClearHistory,
}: LobbyScreenProps) => {
  const [generatorVisible, setGeneratorVisible] = useState(false);

  return (
    <div className="min-h-screen relative overflow-hidden flex flex-col">
      {/* Animated blurred background */}
      <div
        className="fixed -inset-10 z-0 bg-cover bg-center blur-[30px] brightness-[0.25] saturate-[0.6] animate-[lobby-bg-drift_30s_ease-in-out_infinite_alternate]"
        style={{ backgroundImage: `url(${DEFAULT_SCENES[0]?.imageUrl})` }}
      />

      {/* Top bar */}
      <div className="relative z-2 flex justify-end items-center gap-4 px-6 py-3 shrink-0">
        <span className="text-xs text-white/50 tracking-wide">
          {user?.email || user?.username || 'Local Dev'}
        </span>
        <button
          className="bg-transparent border border-white/15 text-white/50 px-4 py-1.5 rounded-md text-xs cursor-pointer hover:border-white/30 hover:text-white transition-all"
          onClick={onSignOut}
        >
          <LogOut className="h-3 w-3 inline mr-1.5" />
          Sign Out
        </button>
      </div>

      {/* Main content — vertically centered */}
      <div className="relative z-1 flex-1 flex flex-col justify-center max-w-[1100px] w-full mx-auto px-8 py-8">
        {/* Hero */}
        <div className="text-center mb-12">
          <h1
            className="text-[clamp(32px,5vw,56px)] font-extrabold tracking-[0.08em] uppercase leading-tight text-white m-0"
            style={{ textShadow: '0 0 40px rgba(83,159,229,0.4), 0 0 80px rgba(83,159,229,0.15)' }}
          >
            INTERACTIVE<br />WORLD BUILDER
          </h1>
          <p className="text-[15px] text-white/45 tracking-wide mt-2">
            Generate & explore AI-powered game worlds in real-time
          </p>
        </div>

        {/* Section: Select a World */}
        <div className="text-[11px] font-bold tracking-[0.12em] uppercase text-white/30 mb-4">
          SELECT A WORLD
        </div>

        {/* Scene tiles */}
        <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-5 mb-8">
          {DEFAULT_SCENES.map((scene) => (
            <button
              key={scene.id}
              className="relative border-none p-0 cursor-pointer rounded-xl overflow-hidden bg-game-surface aspect-video transition-all duration-250 hover:-translate-y-1 hover:scale-[1.02] hover:shadow-[0_12px_40px_rgba(83,159,229,0.2),0_0_0_2px_rgba(83,159,229,0.4)] active:-translate-y-0.5 active:scale-[1.005] group"
              onClick={() => onStartScene(scene)}
            >
              <img
                src={scene.imageUrl}
                alt={scene.name}
                className="w-full h-full object-cover block transition-all duration-300 group-hover:brightness-[0.7] group-hover:scale-[1.08]"
              />
              <div className="absolute inset-0 flex flex-col justify-end p-4 bg-gradient-to-t from-black/85 via-transparent to-transparent">
                <span className="text-white text-base font-bold tracking-wide drop-shadow-md">
                  {scene.name}
                </span>
                <span className="text-white/50 text-[13px] font-semibold tracking-[0.08em] mt-1 opacity-0 translate-y-1 transition-all duration-200 group-hover:opacity-100 group-hover:translate-y-0">
                  ▶ PLAY
                </span>
              </div>
            </button>
          ))}
        </div>

        {/* Generate button */}
        <div className="text-center mb-10">
          <Button variant="glow" size="lg" onClick={() => setGeneratorVisible(true)}>
            <Sparkles className="h-5 w-5" />
            Generate New World with AI
          </Button>
        </div>

        {/* Session History — fade in */}
        {sessions.length > 0 && (
          <div className="mt-2 animate-[fadeIn_0.5s_ease-in]">
            <div className="text-[11px] font-bold tracking-[0.12em] uppercase text-white/30 mb-4 flex items-center gap-3">
              RECENT SESSIONS
              <button
                className="bg-transparent border-none text-white/25 text-[11px] cursor-pointer tracking-wide uppercase hover:text-white/50"
                onClick={onClearHistory}
              >
                Clear
              </button>
            </div>
            <div className="flex flex-col gap-0.5">
              {sessions.slice(0, 5).map((s) => (
                <div
                  key={s.id}
                  className="flex items-center gap-3 px-3 py-2 rounded-lg bg-white/[0.03] hover:bg-white/[0.06] transition-colors"
                >
                  <div className="w-12 h-7 rounded overflow-hidden bg-game-surface shrink-0">
                    <img src={s.thumbnailUrl} alt="" className="w-full h-full object-cover block" />
                  </div>
                  <span className="flex-1 text-[13px] text-white/70">{s.sceneName}</span>
                  <span className="text-xs text-white/25">{fmtDate(s.startedAt)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Nova Canvas Generator Modal */}
      <SceneGenerator
        visible={generatorVisible}
        onDismiss={() => setGeneratorVisible(false)}
        onSelectImage={onStartNovaScene}
      />
    </div>
  );
};
