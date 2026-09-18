// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, useEffect, useCallback, useRef } from 'react';
import { GameMessage } from '../types';
import { getConfig } from '../services/config';

interface UseWebSocketProps {
  url: string;
  /** Called for JSON text messages (control, pong, connected, started, etc.) */
  onMessage?: (data: GameMessage) => void;
  /** Called for binary messages (raw JPEG frame bytes) */
  onBinaryMessage?: (data: Blob) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

export const useWebSocket = ({ url, onMessage, onBinaryMessage, onConnect, onDisconnect }: UseWebSocketProps) => {
  const [isConnected, setIsConnected] = useState(false);
  const [latency, setLatency] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const pingIntervalRef = useRef<number | null>(null);

  const connect = useCallback(async () => {
    try {
      // R1: the backend validates the shared WORLD_MODEL_API_TOKEN, so the
      // frontend presents that same token (written into config.js by
      // ./deploy.sh ui) rather than a Cognito idToken the backend never checks.
      const token = getConfig().apiToken;

      // M2: the browser WebSocket API cannot set an Authorization header, so the
      // token is passed as a query param. Always use a wss:// (TLS) endpoint so
      // the URL is encrypted in transit, and do not log full WebSocket URLs
      // server-side. When no token is configured (local/demo), connect without
      // one — the backend only enforces auth when a token is set.
      const wsUrl = token ? `${url}?token=${encodeURIComponent(token)}` : url;
      const ws = new WebSocket(wsUrl);
      // Receive binary data as Blob (default, but explicit for clarity)
      ws.binaryType = 'blob';
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        onConnect?.();

        pingIntervalRef.current = window.setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
          }
        }, 1000);
      };

      ws.onmessage = (event) => {
        // Binary frame = raw JPEG bytes (sent by server via send_bytes)
        if (event.data instanceof Blob) {
          onBinaryMessage?.(event.data);
          return;
        }

        // Text frame = JSON control message
        try {
          const message: GameMessage = JSON.parse(event.data);
          if (message.type === 'pong') {
            const timestamp = message.data?.timestamp || message.timestamp;
            if (timestamp) setLatency(Date.now() - timestamp);
          } else {
            onMessage?.(message);
          }
        } catch (error) {
          console.error('Error parsing message:', error);
        }
      };

      ws.onerror = (error) => console.error('WebSocket error:', error);

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        onDisconnect?.();
        if (pingIntervalRef.current) { clearInterval(pingIntervalRef.current); pingIntervalRef.current = null; }
      };
    } catch (error) {
      console.error('Error connecting to WebSocket:', error);
      throw error;
    }
  }, [url, onMessage, onBinaryMessage, onConnect, onDisconnect]);

  const disconnect = useCallback(() => {
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
    if (pingIntervalRef.current) { clearInterval(pingIntervalRef.current); pingIntervalRef.current = null; }
  }, []);

  const sendMessage = useCallback((message: GameMessage) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  useEffect(() => () => disconnect(), [disconnect]);

  return { isConnected, latency, connect, disconnect, sendMessage };
};
