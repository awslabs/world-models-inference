// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// When `./deploy.sh ui` finds a live endpoint it launches vite with:
//   WM_PROXY_TARGET=http://<alb-dns>   WM_PROXY_TOKEN=<shared-token>
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

function buildProxy() {
  if (!proxyTarget) return undefined
  const addAuth = (proxy: any) => {
    if (!proxyToken) return
    // REST: set the Authorization header on each proxied request server-side.
    proxy.on('proxyReq', (proxyReq: any) => {
      proxyReq.setHeader('Authorization', `Bearer ${proxyToken}`)
    })
// WebSocket upgrades fire proxyReqWs, NOT proxyReq — without this hook the
    // upgrade reaches the backend with no credentials (the browser never has the
    // token to append) and every session is refused with close code 1008.
    // Setting the header rather than a query param also keeps the token out of
    // ALB access logs, which is where `?token=` ends up.
    proxy.on('proxyReqWs', (proxyReq: any) => {
      proxyReq.setHeader('Authorization', `Bearer ${proxyToken}`)
    })
  }
  const entry = (ws: boolean) => ({
    target: proxyTarget,
    changeOrigin: true,
    secure: false, // ALB may be plain HTTP (no ACM cert); ingress is IP-locked
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
  server: {
    port: 3000,
    host: true,
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
