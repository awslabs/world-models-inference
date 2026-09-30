// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { useRef, useCallback, useEffect } from 'react';

interface UseFrameBufferOptions {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  onFpsUpdate?: (fps: number) => void;
}

/**
 * Renders raw JPEG Blobs from binary WebSocket frames directly to canvas.
 *
 * Decodes ONE frame at a time and keeps only the newest one waiting. If frames
 * arrive faster than the browser can decode them — which they do, since the
 * server streams a whole chunk in a burst — the alternative is worse in two
 * ways: every frame starts a concurrent decode, and they land in completion
 * rather than arrival order, so the canvas jitters between old and new frames.
 * Dropping stale frames instead keeps what you see pinned to the newest frame
 * available, which is what matters when you are steering.
 */
export const useFrameBuffer = ({ canvasRef, onFpsUpdate }: UseFrameBufferOptions) => {
  const runningRef = useRef(false);
  const displayCountRef = useRef(0);
  const fpsTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pendingRef = useRef<Blob | null>(null);
  const decodingRef = useRef(false);
  const cbRef = useRef(onFpsUpdate);
  useEffect(() => { cbRef.current = onFpsUpdate; }, [onFpsUpdate]);

  const startPlayback = useCallback(() => {
    runningRef.current = true;
    displayCountRef.current = 0;
    fpsTimerRef.current = setInterval(() => {
      cbRef.current?.(displayCountRef.current);
      displayCountRef.current = 0;
    }, 1000);
  }, []);

  const stopPlayback = useCallback(() => {
    runningRef.current = false;
    pendingRef.current = null;
    if (fpsTimerRef.current) { clearInterval(fpsTimerRef.current); fpsTimerRef.current = null; }
  }, []);

  const drainRef = useRef<() => void>(() => {});
  drainRef.current = () => {
    if (decodingRef.current || !runningRef.current) return;
    const blob = pendingRef.current;
    if (!blob) return;
    pendingRef.current = null;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    decodingRef.current = true;
    // createImageBitmap decodes off the main thread, unlike Image + object URL.
    createImageBitmap(blob)
      .then((bmp) => {
        if (runningRef.current) {
          ctx.imageSmoothingEnabled = true;
          ctx.imageSmoothingQuality = 'high';
          ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
          displayCountRef.current++;
        }
        bmp.close();
      })
      .catch(() => { /* a corrupt frame is not worth killing the session over */ })
      .finally(() => {
        decodingRef.current = false;
        if (pendingRef.current) drainRef.current();
      });
  };

  /** Queue a binary JPEG Blob frame for display, replacing any frame still waiting. */
  const pushFrame = useCallback((blob: Blob) => {
    if (!runningRef.current) return;
    pendingRef.current = blob;
    drainRef.current();
  }, []);

  useEffect(() => () => {
    runningRef.current = false;
    if (fpsTimerRef.current) clearInterval(fpsTimerRef.current);
  }, []);

  return { pushFrame, startPlayback, stopPlayback };
};
