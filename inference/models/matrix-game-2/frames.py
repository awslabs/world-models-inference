# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame encoding and pacing for the Matrix Game 2 cartridge.

Each binary WebSocket message is a bare JPEG — the catalogue UI draws it
straight onto a canvas. Kept separate from runner.py so it is testable without
the model or a GPU.
"""

from __future__ import annotations

import io
from typing import Iterable, List

import numpy as np

try:  # faster than PIL; optional so the tests run anywhere
    import simplejpeg
except ImportError:  # pragma: no cover
    simplejpeg = None

DEFAULT_QUALITY = 80


def resolve_quality(value: str | None) -> int:
    """A configured JPEG quality, clamped to 1-100; the default if unusable.

    Resolved once at setup so a typo in the container environment costs one
    warning at boot, not an exception per frame.
    """
    try:
        quality = int(str(value).strip())
    except (TypeError, ValueError):
        return DEFAULT_QUALITY
    return max(1, min(100, quality))


def encode_jpeg(frame: np.ndarray, quality: int = DEFAULT_QUALITY) -> bytes:
    """Encode one HWC uint8 RGB frame as JPEG bytes."""
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    if simplejpeg is not None:
        return simplejpeg.encode_jpeg(frame, quality=quality, colorspace="RGB",
                                      colorsubsampling="420", fastdct=True)
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame, mode="RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_batch(frames: Iterable[np.ndarray], quality: int = DEFAULT_QUALITY) -> List[bytes]:
    return [encode_jpeg(f, quality) for f in frames]


class Pacer:
    """Spreads each chunk's frames across the time the next chunk takes.

    The model emits 12 frames per ~0.7 s chunk. Sent as one burst, the
    catalogue player — which keeps only the newest undecoded frame — would
    drop most of them and the world would stutter at chunk rate. Instead the
    interval tracks generation time per frame, so the buffer drains about as
    fast as it fills. Smoothed because chunk times jitter; the first chunk is
    skipped because it carries seed encoding (and, on a cold start, compile).
    """

    ALPHA = 0.7
    FLOOR = 0.010     # never faster than 100 fps
    OVERHEAD = 0.002  # JPEG encode + send per frame, already spent

    def __init__(self, initial: float = 0.05):
        self.interval = initial
        self._chunks = 0

    def observe(self, chunk_seconds: float, n_frames: int) -> float:
        self._chunks += 1
        if self._chunks > 1 and n_frames > 0 and chunk_seconds > 0:
            ideal = max(chunk_seconds / n_frames - self.OVERHEAD, self.FLOOR)
            self.interval = self.ALPHA * ideal + (1 - self.ALPHA) * self.interval
        return self.interval
