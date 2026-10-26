// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useCallback, useEffect, useState } from 'react';
import { Amplify } from 'aws-amplify';
import { fetchAuthSession, signInWithRedirect, signOut } from 'aws-amplify/auth';
import { Hub } from 'aws-amplify/utils';
import { Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { getConfig } from '@/services/config';
import { ApiTokenContext } from '@/services/apiToken';

// Access tokens live an hour. fetchAuthSession() refreshes one that has expired
// (with the refresh token), so polling well inside the hour keeps it current.
const REFRESH_MS = 5 * 60 * 1000;

/**
 * Cognito Hosted UI sign-in for the Launch Stack's hosted UI (config.auth ===
 * 'hosted'). Renders `children` once signed in, with the user's access token in
 * ApiTokenContext.
 *
 * The scopes are exactly the two the deployment's app client allows: `openid`,
 * and the resource-server scope the endpoint requires of every token.
 */
export const HostedSignIn = ({ children }: { children: ReactNode }) => {
  const [token, setToken] = useState<string | null | undefined>(undefined);  // undefined = checking

  const refresh = useCallback(async () => {
    try {
      const session = await fetchAuthSession();
      setToken(session.tokens?.accessToken?.toString() ?? null);
    } catch {
      setToken(null);
    }
  }, []);

  useEffect(() => {
    const config = getConfig();
    Amplify.configure({
      Auth: {
        Cognito: {
          userPoolId: config.userPoolId,
          userPoolClientId: config.userPoolClientId,
          loginWith: {
            oauth: {
              domain: config.cognitoDomain.replace('https://', ''),
              scopes: ['openid', 'world-model/invoke'],
              redirectSignIn: [config.redirectSignIn],
              redirectSignOut: [config.redirectSignOut],
              responseType: 'code',
            },
          },
        },
      },
    });
    const unsubscribe = Hub.listen('auth', ({ payload }) => {
      if (payload.event === 'signInWithRedirect' || payload.event === 'tokenRefresh') void refresh();
      else if (payload.event === 'signedOut' || payload.event === 'signInWithRedirect_failure') setToken(null);
    });
    void refresh();
    const timer = window.setInterval(() => void refresh(), REFRESH_MS);
    return () => { unsubscribe(); window.clearInterval(timer); };
  }, [refresh]);

  if (token === undefined) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-game-bg gap-4">
        <Loader2 className="h-8 w-8 animate-spin text-game-accent" />
        <p className="text-white/40 text-sm">Authenticating...</p>
      </div>
    );
  }

  if (token === null) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-game-bg gap-8">
        <h1 className="text-4xl font-extrabold tracking-tight text-white text-center">World Model Inference</h1>
        <Button variant="primary" size="lg" onClick={() => void signInWithRedirect()}>
          Sign in
        </Button>
      </div>
    );
  }

  return (
    <ApiTokenContext.Provider value={token}>
      {children}
      <div className="fixed bottom-3 right-4 z-10">
        <button
          onClick={() => void signOut()}
          className="text-[11px] text-white/30 hover:text-white/70 transition-colors"
        >
          Sign out
        </button>
      </div>
    </ApiTokenContext.Provider>
  );
};
