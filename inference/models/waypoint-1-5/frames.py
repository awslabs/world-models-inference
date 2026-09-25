# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame encoding for the Waypoint-1.5 cartridge.

The wire format for a real-time frame is a bare JPEG with no envelope — the
catalogue UI renders each binary WebSocket message as an image directly.

world_engine's gen_frame returns a uint8 RGB tensor shaped [T, H, W, 3]
(T = temporal_compression, 4 for Waypoint-1.5). This module turns one such
batch into a list of JPEG payloads. Kept separate from runner.py so it is
testable without the model or a GPU.
"""

from __future__ import annotations

import io
from typing import List

import numpy as np

try:  # ~4x faster than PIL for 720p; optional so tests run anywhere
    import simplejpeg
except ImportError:  # pragma: no cover
    simplejpeg = None

DEFAULT_QUALITY = 60

# Chroma subsampling. simplejpeg defaults to '444' — full-resolution colour,
# which costs 17.9% more bytes on this model's frames (73.1 → 86.2 KiB measured
# at 720p q60, so 4:2:0 saves 15.2%) and buys nothing a player can see in a
# moving world. Delivered fps is bytes-bound, so those bytes are frames. PIL's
# fallback path uses the equivalent integer code.
DEFAULT_SUBSAMPLING = "420"
_PIL_SUBSAMPLING = {"444": 0, "422": 1, "420": 2}
# simplejpeg accepts more names than PIL does; offer only the three both
# backends encode identically, so a deployment behaves the same whether or not
# the fast encoder is present in the image.
SUBSAMPLINGS = frozenset(_PIL_SUBSAMPLING)


def resolve_subsampling(name: str | None) -> str:
    """Normalise a configured subsampling name, falling back to the default.

    Called once at setup rather than per frame: an unusable value should show up
    as one warning in the container log, not as an exception in the encode loop
    that takes the player's session down with it.
    """
    cleaned = (name or "").strip()
    return cleaned if cleaned in SUBSAMPLINGS else DEFAULT_SUBSAMPLING


def resolve_quality(value: str | None) -> int:
    """Normalise a configured JPEG quality, falling back to the default.

    Same reasoning as resolve_subsampling: a typo in the container environment
    should not stop the endpoint booting. Clamped because libjpeg's usable range
    is 1-100 and the values outside it are silently reinterpreted.
    """
    try:
        quality = int(str(value).strip())
    except (TypeError, ValueError):
        return DEFAULT_QUALITY
    return max(1, min(100, quality))


def to_uint8_batch(frames) -> np.ndarray:
    """Coerce a [T,H,W,C] or [H,W,C] tensor/array to a uint8 [T,H,W,3] array."""
    if hasattr(frames, "detach"):  # torch tensor, possibly on GPU
        frames = frames.detach().to("cpu").numpy()
    arr = np.asarray(frames)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4:
        raise ValueError(f"expected [T,H,W,C] frames, got shape {arr.shape}")
    if arr.shape[-1] not in (1, 3) and arr.shape[1] in (1, 3):
        arr = np.transpose(arr, (0, 2, 3, 1))  # CHW → HWC
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.ascontiguousarray(arr)


def encode_jpeg(frame: np.ndarray, quality: int = DEFAULT_QUALITY,
                subsampling: str = DEFAULT_SUBSAMPLING) -> bytes:
    """Encode one HWC uint8 frame as JPEG bytes."""
    if simplejpeg is not None:
        return simplejpeg.encode_jpeg(
            np.ascontiguousarray(frame), quality=quality, colorspace="RGB",
            colorsubsampling=subsampling,
            # The fast DCT is 4-5% quicker for a loss invisible next to what
            # quality 60 already discards.
            fastdct=True,
        )
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame, mode="RGB").save(
        buf, format="JPEG", quality=quality,
        subsampling=_PIL_SUBSAMPLING.get(subsampling, 2),
    )
    return buf.getvalue()


def encode_batch(frames, quality: int = DEFAULT_QUALITY,
                 subsampling: str = DEFAULT_SUBSAMPLING) -> List[bytes]:
    """Encode a gen_frame result into one JPEG per output frame."""
    return [encode_jpeg(f, quality, subsampling) for f in to_uint8_batch(frames)]
