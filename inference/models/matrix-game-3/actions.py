# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action decoding for the Matrix Game 3.0 cartridge.

The browser sends keyboard and mouse state; upstream wants two tensors per
generated chunk. Keeping the translation here means it is testable without a
GPU, the model, or the upstream package installed.

Upstream's encoding (pipeline/inference_interactive_pipeline.py):

  keyboard: 6-dim one-hot over [forward, back, left, right, _, _]
            w -> [1,0,0,0,0,0]   s -> [0,1,0,0,0,0]
            a -> [0,0,1,0,0,0]   d -> [0,0,0,1,0,0]
            q (no move) -> all zeros
  mouse:    2-dim [pitch, yaw] in units of their CAM_VALUE (0.1)
            i -> [+0.1, 0]  (up)     k -> [-0.1, 0]  (down)
            j -> [0, -0.1]  (left)   l -> [0, +0.1]  (right)
            u (no move) -> [0, 0]

Note the axis order: upstream indexes mouse as [pitch, yaw], so vertical look
comes first. Getting this backwards swaps looking up with looking sideways.
"""

from __future__ import annotations

import json
import struct
from typing import Tuple

# Upstream's per-tick camera step. Matching it keeps our motion scale identical
# to their reference run rather than subtly faster or slower.
CAM_VALUE = 0.1

KEYBOARD_DIM = 6
MOUSE_DIM = 2

_KEY_TO_INDEX = {
    "w": 0,
    "s": 1,
    "a": 2,
    "d": 3,
}

Action = Tuple[list, list]


def neutral() -> Action:
    """The 'no input' action: standing still, camera unchanged."""
    return [0.0] * KEYBOARD_DIM, [0.0] * MOUSE_DIM


def decode(payload: bytes) -> Action:
    """Turn one client message into (keyboard, mouse) lists.

    Accepts three shapes, in this order:

      1. 12-byte binary: 4 keys as bitflags in a uint32, then two float32
         mouse deltas. This is the low-overhead path for a game loop.
      2. JSON: {"keys": "wa", "mouse": [dx, dy]} where mouse is already in
         CAM_VALUE units, or {"keyboard": [...6], "mouse": [...2]} to pass
         upstream's vectors straight through.
      3. Raw key bytes, e.g. b"wa".

    Anything unparseable yields the neutral action rather than raising: a
    malformed frame from a browser should drop input for one tick, not kill
    the session.
    """
    if not payload:
        return neutral()

    # A compact JSON message can be exactly 12 bytes too — b'{"keys":"w"}' —
    # and any 12 bytes unpack "successfully" as the binary struct, turning JSON
    # into garbage key flags. JSON always starts with '{' here, and a binary
    # frame's first byte is the low flag bits (values 0-15), never 0x7b.
    if len(payload) == 12 and payload[:1] != b"{":
        try:
            flags, dx, dy = struct.unpack("<Iff", payload)
            keyboard = [0.0] * KEYBOARD_DIM
            for bit, key in enumerate("wsad"):
                if flags & (1 << bit):
                    keyboard[_KEY_TO_INDEX[key]] = 1.0
            return keyboard, _clamp_mouse([dx, dy])
        except struct.error:
            return neutral()

    try:
        obj = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _from_raw_keys(payload.decode("ascii", errors="ignore"))

    if not isinstance(obj, dict):
        return neutral()

    # Explicit upstream vectors win, so a caller can drive the model directly.
    if "keyboard" in obj:
        keyboard = _fit(obj.get("keyboard"), KEYBOARD_DIM)
        mouse = _clamp_mouse(_fit(obj.get("mouse"), MOUSE_DIM))
        return keyboard, mouse

    keyboard, mouse = _from_keys(str(obj.get("keys", "")))
    if "mouse" in obj:
        mouse = _clamp_mouse(_fit(obj.get("mouse"), MOUSE_DIM))
    return keyboard, mouse


_ALL_KEYS = set("wsadqijklu")


def _from_raw_keys(text: str) -> Action:
    """Decode a bare key string, but only if every character is a known key.

    Without this guard any non-JSON payload gets read as keys, so a stray
    string like "not json" moves the player backwards and pans the camera
    because it happens to contain 's' and 'j'. Requiring the whole payload to
    be keys keeps the b"wa" convenience without turning noise into input.
    """
    if not text or not set(text.lower()) <= _ALL_KEYS:
        return neutral()
    return _from_keys(text)


def _from_keys(keys: str) -> Action:
    """Map a set of held keys to upstream's vectors.

    WASD drives movement; IJKL drives the camera, mirroring upstream's own
    keybinding so their docs still describe our endpoint.
    """
    keyboard = [0.0] * KEYBOARD_DIM
    pitch = yaw = 0.0
    for ch in keys.lower():
        if ch in _KEY_TO_INDEX:
            keyboard[_KEY_TO_INDEX[ch]] = 1.0
        elif ch == "i":
            pitch += CAM_VALUE
        elif ch == "k":
            pitch -= CAM_VALUE
        elif ch == "j":
            yaw -= CAM_VALUE
        elif ch == "l":
            yaw += CAM_VALUE
    return keyboard, _clamp_mouse([pitch, yaw])


def _fit(value, size: int) -> list:
    """Coerce a client-supplied list to exactly `size` floats."""
    if not isinstance(value, (list, tuple)):
        return [0.0] * size
    out = [0.0] * size
    for i, item in enumerate(value[:size]):
        try:
            out[i] = float(item)
        except (TypeError, ValueError):
            out[i] = 0.0
    return out


def _clamp_mouse(mouse: list) -> list:
    """Bound camera deltas to one CAM_VALUE step per axis per tick.

    A client can otherwise send an arbitrarily large delta and teleport the
    camera, which breaks the model's pose continuity and its scene memory.
    """
    return [max(-CAM_VALUE, min(CAM_VALUE, float(v))) for v in mouse[:MOUSE_DIM]]
