// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// The GPU capacity wizard page. Built separately from the catalogue UI so it never
// ships in the hosted UI bucket: it needs scripts/capacity.py (your local AWS
// credentials) behind it. `python scripts/capacity.py` builds and serves it.
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  build: {
    outDir: 'dist-capacity',
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: { input: path.resolve(__dirname, 'capacity.html') },
  },
})
