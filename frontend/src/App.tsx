// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState, useCallback } from 'react';
import { Amplify } from 'aws-amplify';
import { signInWithRedirect, signOut, getCurrentUser, fetchAuthSession } from 'aws-amplify/auth';
import { Hub } from 'aws-amplify/utils';
import { Loader2 } from 'lucide-react';

import { getConfig } from '@/services/config';
import { useSessionHistory } from '@/hooks/useSessionHistory';
import { Scene } from '@/data/scenes';
import { DEFAULT_SCENES } from '@/data/scenes';
import { Button } from '@/components/ui/button';
import { LobbyScreen } from '@/components/LobbyScreen';
import { GameCanvas } from '@/components/GameCanvas';
import { Catalogue } from '@/components/Catalogue';

import '@/styles/App.css';

interface AuthUser { username: string; email?: string; }

export interface PendingStart {
  sceneName: string;
  thumbnailUrl: string;
  seedImageS3Key: string;
  imageData?: string;
  imageS3Key?: string;
}

type AppScreen = 'lobby' | 'game';

function App() {
  const [isConfigured, setIsConfigured] = useState(false);
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [screen, setScreen] = useState<AppScreen>('lobby');
  const [pendingStart, setPendingStart] = useState<PendingStart | null>(null);
  const isLocalDev = typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');

  // Demo mode: skip auth entirely so the UI can be previewed on CloudFront
  const isDemoMode = (() => { try { return getConfig().demoMode === true; } catch { return false; } })();

  // `./deploy ui` sets this to 'lingbot' → render the World Foundry catalogue
  // pointing at a live EC2 endpoint. No auth, no lobby, no WebSocket.
  const uiMode = (() => { try { return getConfig().ui || 'worlds'; } catch { return 'worlds'; } })();
  const apiUrl = (() => { try { return getConfig().lingbotApiUrl; } catch { return ''; } })();
  if (uiMode === 'lingbot') {
    return (
      <div className="min-h-screen relative overflow-hidden flex flex-col bg-game-bg">
        {/* Animated blurred atmospheric background (matches LobbyScreen) */}
        <div
          className="fixed -inset-10 z-0 bg-cover bg-center blur-[30px] brightness-[0.2] saturate-[0.6] animate-[lobby-bg-drift_30s_ease-in-out_infinite_alternate]"
          style={{ backgroundImage: `url(${DEFAULT_SCENES[0]?.imageUrl})` }}
        />
        {/* Content */}
        <div className="relative z-1 max-w-[1200px] w-full mx-auto px-6 py-6 flex-1">
          <Catalogue endpointUrl={apiUrl} />
        </div>
      </div>
    );
  }

  const authReady = !!user || isLocalDev || isDemoMode;
  const { sessions, addSession, clearHistory } = useSessionHistory(authReady);

  // ── Amplify config ────────────────────────────────────────────
  useEffect(() => {
    if (isDemoMode) { setIsConfigured(true); setIsLoading(false); return; }
    try {
      const config = getConfig();
      Amplify.configure({
        Auth: {
          Cognito: {
            userPoolId: config.userPoolId,
            userPoolClientId: config.userPoolClientId,
            loginWith: {
              oauth: {
                domain: config.cognitoDomain.replace('https://', ''),
                scopes: ['email', 'openid', 'phone', 'profile'],
                redirectSignIn: [config.redirectSignIn],
                redirectSignOut: [config.redirectSignOut],
                responseType: 'code',
                providers: [{ custom: 'Federate' }],
              },
            },
          },
        },
      });
      setIsConfigured(true);
    } catch (e) { console.error('Amplify config failed:', e); }
  }, [isDemoMode]);

  const checkUser = useCallback(async () => {
    if (isDemoMode) { setUser({ username: 'Demo User' }); setIsLoading(false); return; }
    try {
      const cu = await getCurrentUser();
      const s = await fetchAuthSession();
      setUser({ username: cu.username, email: s.tokens?.idToken?.payload?.email as string | undefined });
    } catch { setUser(null); } finally { setIsLoading(false); }
  }, [isDemoMode]);

  useEffect(() => {
    if (!isConfigured) return;
    if (isDemoMode) { setUser({ username: 'Demo User' }); return; }
    checkUser();
    const unsub = Hub.listen('auth', ({ payload }) => {
      if (payload.event === 'signInWithRedirect') checkUser();
      else if (payload.event === 'signInWithRedirect_failure' || payload.event === 'signedOut') { setUser(null); setIsLoading(false); }
    });
    return unsub;
  }, [isConfigured, isDemoMode, checkUser]);

  // ── Auth gates ────────────────────────────────────────────────
  if (!isConfigured || isLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-game-bg gap-4">
        <Loader2 className="h-8 w-8 animate-spin text-game-accent" />
        <p className="text-white/40 text-sm">{!isConfigured ? 'Loading...' : 'Authenticating...'}</p>
      </div>
    );
  }

  if (!user && !isLocalDev && !isDemoMode) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-game-bg gap-8">
        <h1 className="text-5xl font-extrabold tracking-wider text-white text-center" style={{ textShadow: '0 0 40px rgba(83,159,229,0.4)' }}>
          INTERACTIVE<br/>WORLD BUILDER
        </h1>
        <Button variant="primary" size="lg" onClick={() => signInWithRedirect({ provider: { custom: 'Federate' } })}>
          Sign in with Midway
        </Button>
      </div>
    );
  }

  // ── Helpers ───────────────────────────────────────────────────
  const imageUrlToBase64 = async (url: string): Promise<string> => {
    const res = await fetch(url);
    const blob = await res.blob();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  const handleStartScene = async (scene: Scene) => {
    try {
      const base64 = await imageUrlToBase64(scene.imageUrl);
      setPendingStart({ sceneName: scene.name, thumbnailUrl: scene.imageUrl, seedImageS3Key: '', imageData: base64 });
      setScreen('game');
    } catch (e) { console.error('Failed to load scene:', e); }
  };

  const handleStartNovaScene = (s3Key: string, url: string, title?: string) => {
    setPendingStart({ sceneName: title || '✨ Custom Scene', thumbnailUrl: url, seedImageS3Key: s3Key, imageS3Key: s3Key });
    setScreen('game');
  };

  const handleBackToLobby = () => { setScreen('lobby'); setPendingStart(null); };

  // ── Render ────────────────────────────────────────────────────
  if (screen === 'game' && pendingStart) {
    return <GameCanvas pendingStart={pendingStart} onBack={handleBackToLobby} addSession={addSession} />;
  }

  return (
    <LobbyScreen
      user={user}
      onStartScene={handleStartScene}
      onStartNovaScene={handleStartNovaScene}
      onSignOut={() => signOut()}
      sessions={sessions}
      onClearHistory={clearHistory}
    />
  );
}

export default App;
