// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, useCallback, useEffect } from 'react';
import { fetchAuthSession } from 'aws-amplify/auth';
import { getConfig } from '../services/config';

export interface SessionRecord {
  id: string;
  sceneName: string;
  thumbnailUrl: string;
  seedImageS3Key: string;
  startedAt: string;
}

async function getAuthHeaders(): Promise<Record<string, string>> {
  const session = await fetchAuthSession();
  const token = session.tokens?.idToken?.toString();
  return {
    'Content-Type': 'application/json',
    ...(token ? { 'Authorization': token } : {}),
  };
}

function getApiBase(): string {
  const cfg = getConfig();
  const url = cfg.apiUrl || '';
  return url.endsWith('/') ? url : `${url}/`;
}

export const useSessionHistory = (authReady: boolean = false) => {
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (!authReady) return;
    const load = async () => {
      try {
        const apiBase = getApiBase();
        if (!apiBase || apiBase === '/') { setLoaded(true); return; }
        const headers = await getAuthHeaders();
        const res = await fetch(`${apiBase}api/sessions`, { headers });
        if (res.ok) {
          const data = await res.json();
          setSessions(data.sessions || []);
        } else {
          console.error('Failed to load sessions:', res.status);
        }
      } catch (e) {
        console.error('Failed to load sessions:', e);
      } finally {
        setLoaded(true);
      }
    };
    load();
  }, [authReady]);

  const addSession = useCallback(async (session: {
    sceneName: string;
    thumbnailUrl: string;
    seedImageS3Key: string;
  }): Promise<string> => {
    const apiBase = getApiBase();
    const headers = await getAuthHeaders();
    const res = await fetch(`${apiBase}api/sessions`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        sceneName: session.sceneName,
        seedImageS3Key: session.seedImageS3Key,
        thumbnailUrl: session.thumbnailUrl,
        startedAt: new Date().toISOString(),
      }),
    });
    if (!res.ok) throw new Error(`Failed to create session: ${res.status}`);
    const data = await res.json();
    const record: SessionRecord = {
      id: data.id,
      sceneName: data.sceneName,
      thumbnailUrl: data.thumbnailUrl || session.thumbnailUrl,
      seedImageS3Key: data.seedImageS3Key || session.seedImageS3Key,
      startedAt: data.startedAt,
    };
    setSessions(prev => [record, ...prev]);
    return record.id;
  }, []);

  const clearHistory = useCallback(() => {
    setSessions([]);
  }, []);

  return { sessions, addSession, clearHistory, loaded };
};
