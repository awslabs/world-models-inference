// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { useRef, useCallback, useEffect } from 'react';

interface UseFrameBufferOptions {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  onFpsUpdate?: (fps: number) => void;
}

/**
 * Renders raw JPEG Blobs from binary WebSocket frames directly to canvas.
 * Uses URL.createObjectURL for zero-copy display — no base64 overhead.
 */
export const useFrameBuffer = ({ canvasRef, onFpsUpdate }: UseFrameBufferOptions) => {
  const runningRef = useRef(false);
  const displayCountRef = useRef(0);
  const fpsTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
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
    if (fpsTimerRef.current) { clearInterval(fpsTimerRef.current); fpsTimerRef.current = null; }
  }, []);

  /** Display a binary JPEG Blob frame directly on the canvas. */
  const pushFrame = useCallback((blob: Blob) => {
    if (!runningRef.current) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
    };
    img.onerror = () => URL.revokeObjectURL(url);
    img.src = url;
    displayCountRef.current++;
  }, [canvasRef]);

  useEffect(() => () => {
    runningRef.current = false;
    if (fpsTimerRef.current) clearInterval(fpsTimerRef.current);
  }, []);

  return { pushFrame, startPlayback, stopPlayback };
};
