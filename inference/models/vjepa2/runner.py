# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""V-JEPA 2 — realtime video understanding with sliding-window encoding.

Pattern: dashcam-style stateful per-session.
  - Client streams frames (one at a time) via WebSocket
  - Server maintains a sliding deque of last 16 frames per session
  - Each new frame: append → encode window → emit embedding + alert

Concurrency: one client per container. SM auto-scales N instances for N users.
Set sagemaker.max_concurrent: 1 in endpoint.yaml.

Output per frame:
  [4B frame_num LE][4B meta_len LE][JSON meta][float16 embedding bytes]
  meta = {"frame": N, "alert": bool, "distance": 0.42, "embed_dim": ...}
"""

from __future__ import annotations

import collections
import io
import json
import os
import struct
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F

from lib.runner import ActionBuffer, Runner

WINDOW_SIZE = 16
ALERT_THRESHOLD = 0.5  # cosine distance from reference triggers alert


class VJepa2Runner(Runner):
    def setup(self, model_dir: str, device: torch.device, rank: int, world_size: int):
        self.device = device

        from transformers import AutoModel, AutoVideoProcessor

        # Pin the upstream revision so the Hub-download fallback fetches a fixed
        # commit rather than whatever HEAD is (supply-chain hardening). Ignored
        # when weights are already staged locally (model_dir has config.json).
        if os.path.exists(os.path.join(model_dir, "config.json")):
            model_path = model_dir
            revision = None
        else:
            model_path = "facebook/vjepa2-vitg-fpc64-384"
            revision = "12ca91694b230e0d4b5b0078af6f4ae1d51e933d"

        # revision is a pinned commit when pulling from the Hub, and None only
        # when model_path is a local directory (loading from disk, no download).
        self.model = AutoModel.from_pretrained(model_path, revision=revision).to(device).eval()  # nosec B615 - revision pinned for Hub; local path loads from disk
        self.processor = AutoVideoProcessor.from_pretrained(model_path, revision=revision)  # nosec B615 - revision pinned for Hub; local path loads from disk

        self.window: collections.deque = collections.deque(maxlen=WINDOW_SIZE)
        self.reference: Optional[torch.Tensor] = None
        self.frame_count = 0

    def generate(self, **params) -> str:
        raise NotImplementedError("vjepa2 is realtime-only — use WebSocket /ws")

    @property
    def supports_streaming(self) -> bool:
        return True

    def reset(self):
        self.window.clear()
        self.reference = None
        self.frame_count = 0

    @torch.no_grad()
    def stream(self, actions: ActionBuffer) -> Iterator[bytes]:
        from PIL import Image

        last = None
        while actions.active:
            frame_bytes = actions.get()
            if frame_bytes is None:
                break

            # ActionBuffer is a latest-wins slot, not a queue: get() re-returns
            # the payload already seen every <=0.1s whether or not the client
            # sent anything. A cartridge that reads *held input* can re-apply it
            # harmlessly, but here every read is a new video frame, so a camera
            # that pauses would have its last frame appended ~10x a second —
            # filling the 16-frame window with copies of one frame, racing
            # frame_count ahead of the frames actually sent, and taking a
            # duplicate as the session baseline if the pause lands during
            # warm-up.
            #
            # Identity, not equality: the buffer hands back the very object it
            # was given, so `is` distinguishes a re-delivery from a new message.
            # Comparing bytes instead would also throw away real frames whenever
            # the source is synthetic — a test pattern, a screen capture of a
            # still, a looping clip — which encode byte-identically.
            if frame_bytes is last:
                continue
            last = frame_bytes

            try:
                img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            except Exception:
                continue

            self.window.append(np.array(img))
            self.frame_count += 1

            # Need full window before encoding (model expects 16 frames)
            if len(self.window) < WINDOW_SIZE:
                meta = json.dumps({
                    "frame": self.frame_count,
                    "warming_up": True,
                    "frames_buffered": len(self.window),
                }).encode()
                yield struct.pack("<II", self.frame_count, len(meta)) + meta
                continue

            video = np.stack(list(self.window))  # [16, H, W, 3]
            inputs = self.processor(video, return_tensors="pt").to(self.device)
            embedding = self.model.get_vision_features(**inputs)  # [1, T*P, D]
            pooled = embedding.mean(dim=1)  # [1, D]

            # First full window becomes the "normal" baseline
            if self.reference is None:
                self.reference = pooled.clone()

            sim = F.cosine_similarity(pooled, self.reference).item()
            distance = 1.0 - sim
            alert = distance > ALERT_THRESHOLD

            emb_f16 = pooled.squeeze(0).half().cpu().numpy()
            meta = json.dumps({
                "frame": self.frame_count,
                "alert": alert,
                "distance": round(distance, 4),
                "embed_dim": int(emb_f16.shape[0]),
                "dtype": "float16",
            }).encode()
            yield struct.pack("<II", self.frame_count, len(meta)) + meta + emb_f16.tobytes()


def create_runner():
    return VJepa2Runner()
