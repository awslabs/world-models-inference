# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame encoding for the Matrix Game 3.0 cartridge.

The catalogue UI renders each binary WebSocket message as an image directly
(frontend/src/hooks/useFrameBuffer.ts turns the Blob into an <img> src), so the
wire format for a real-time frame is a bare JPEG with no JSON envelope.

Kept separate from runner.py so the tensor-to-JPEG conversion is testable
without the model or a GPU.
"""

from __future__ import annotations

import io
import os
from typing import Iterator

import numpy as np

# Upstream decodes to float in [-1, 1]. Anything outside that is clipped rather
# than wrapped, so an out-of-range pixel shows as saturated instead of noise.
VALUE_RANGE = (-1.0, 1.0)

# JPEG quality for streamed frames. 80 keeps bandwidth low for a game loop; a
# demo on a big screen benefits from raising it (MG3_JPEG_QUALITY=92). The model
# is far more of the quality story than this, but it is a free knob.
DEFAULT_QUALITY = int(os.environ.get("MG3_JPEG_QUALITY", "80"))


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert one CHW or HWC float frame in [-1, 1] to an HWC uint8 array."""
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D frame, got shape {arr.shape}")

    # Channels-first is what torch hands us; the encoder wants channels-last.
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype == np.uint8:
        return arr

    lo, hi = VALUE_RANGE
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def encode_jpeg(frame: np.ndarray, quality: int = DEFAULT_QUALITY) -> bytes:
    """Encode one frame as JPEG bytes, ready to send over the WebSocket."""
    from PIL import Image

    arr = to_uint8(frame)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def split_chunk(chunk) -> Iterator[np.ndarray]:
    """Yield individual frames from one decoded video chunk.

    Upstream's VAE hands back a batched tensor with time on a middle axis —
    shaped (B, C, T, H, W) — because it decodes a whole chunk at once. We walk
    the time axis so the client sees frames arriving rather than one late batch.
    Bare (C, H, W) input is passed through as a single frame.
    """
    if hasattr(chunk, "detach"):
        # .float() before .numpy(): the VAE decodes in bfloat16 and numpy has no
        # bfloat16, so converting straight from the tensor raises
        # "TypeError: Got unsupported ScalarType BFloat16". Calling .float() here
        # keeps this module free of a torch import, which is what lets the tests
        # run on a laptop.
        arr = chunk.detach().float().cpu().numpy()
    else:
        arr = np.asarray(chunk)

    if arr.ndim == 5:  # (B, C, T, H, W) — take the first item in the batch
        arr = arr[0]
    if arr.ndim == 4:  # (C, T, H, W)
        for t in range(arr.shape[1]):
            yield arr[:, t]
        return
    if arr.ndim == 3:
        yield arr
        return
    raise ValueError(f"unexpected chunk shape {arr.shape}")
