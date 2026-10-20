// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'
import type { IncomingMessage } from 'http'

// When `./deploy.sh ui` finds a live endpoint it launches vite with:
//   WM_PROXY_TARGET=https://<endpoint>   WM_PROXY_TOKEN=<machine-token>
// The dev server then proxies the inference API paths to the ALB and injects
// the bearer token SERVER-SIDE. Two wins: (1) the browser only ever talks to
// localhost, so there is no cross-origin/CORS problem; (2) the token lives in
// this node process, never in the browser bundle. Without these env vars the
// proxy is disabled (plain local/demo UI).
const proxyTarget = process.env.WM_PROXY_TARGET
const proxyToken = process.env.WM_PROXY_TOKEN

// Inference API paths the UI calls. Everything else (the SPA, assets) is served
// by vite itself. Keep in sync with inference/lib/app.py routes.
const API_PATHS = ['/health', '/ping', '/generate', '/status', '/result', '/jobs', '/examples', '/invocations']
const WS_PATHS = ['/ws', '/invocations-bidirectional-stream']

// The proxy holds a machine token, so it must only lend it to the SPA this dev
// server serves. `changeOrigin: true` rewrites only the OUTGOING Host, so
// req.headers here still carry what the browser sent.
const UI_HOSTS = new Set(['localhost:3000', '127.0.0.1:3000'])
const UI_ORIGINS = new Set(['http://localhost:3000', 'http://127.0.0.1:3000'])

// True only for requests the SPA served from this dev server makes itself.
// Host stops DNS rebinding; Origin and Sec-Fetch-Site stop cross-site pages
// (a form POST or `new WebSocket('ws://localhost:3000/ws')` from another site).
function fromLocalUi(req: IncomingMessage): boolean {
  if (!UI_HOSTS.has(req.headers.host ?? '')) return false
  const origin = req.headers.origin
  if (origin !== undefined && !UI_ORIGINS.has(origin)) return false
  const site = req.headers['sec-fetch-site']
  return site === undefined || site === 'same-origin' || site === 'none'
}

function buildProxy() {
  if (!proxyTarget) return undefined
  const addAuth = (proxy: any) => {
    if (!proxyToken) return
    // REST: set the Authorization header on each proxied request server-side.
    // Only for requests the local UI made itself (see fromLocalUi): anything else
    // is still forwarded, just without the token, so the endpoint answers 401 or
    // closes the WebSocket with 1008. The proxy needs no rejection path of its own.
    proxy.on('proxyReq', (proxyReq: any, req: IncomingMessage) => {
      if (fromLocalUi(req)) proxyReq.setHeader('Authorization', `Bearer ${proxyToken}`)
    })
    // WebSocket upgrades fire proxyReqWs, NOT proxyReq — without this hook the
    // upgrade reaches the backend with no credentials (the browser never has the
    // token to append) and every session is refused with close code 1008.
    // Setting the header rather than a query param also keeps the token out of
    // ALB access logs, which is where `?token=` ends up.
    proxy.on('proxyReqWs', (proxyReq: any, req: IncomingMessage) => {
      if (fromLocalUi(req)) proxyReq.setHeader('Authorization', `Bearer ${proxyToken}`)
    })
  }
  const entry = (ws: boolean) => ({
    target: proxyTarget,
    changeOrigin: true,
    // Verify the endpoint's certificate. The target is HTTPS by default (CloudFront
    // or an ACM-backed domain); deploy.sh refuses an http:// one without ALLOW_PLAINTEXT.
    secure: true,
    ws,
    configure: addAuth,
  })
  const proxy: Record<string, any> = {}
  for (const p of API_PATHS) proxy[p] = entry(false)
  for (const p of WS_PATHS) proxy[p] = entry(true)
  return proxy
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  // Loopback only: a proxy that injects a token must not be reachable from the
  // LAN. strictPort because fromLocalUi trusts port 3000 alone — vite would
  // otherwise move to 3001 when 3000 is busy and every request would lose auth.
  // No LAN mode on purpose: colleagues each run their own `./deploy.sh ui`.
  server: {
    port: 3000,
    strictPort: true,
    host: '127.0.0.1',
    proxy: buildProxy(),
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'aws-vendor': ['aws-amplify'],
        },
      },
    },
  },
})
