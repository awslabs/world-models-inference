# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action decoding for the Matrix Game 2 cartridge.

The model takes four movement keys (W, S, A, D) and a ternary yaw; pitch is
disabled (see engine.action_tensors). Clients speak one of:

  12-byte binary   <uint32 LE wsad bitflags><float32 pitch><float32 yaw>
                   — what the catalogue's RealtimePlayer sends (shared with
                   matrix-game-3). Any non-zero yaw turns that way.
  {"type": "control", "buttons": ["W", "J"], "mouse_dx": -100..100, ...}
                   — the booth kiosk / gamepad dialect. J/L buttons turn;
                   mouse_dx is a stick position scaled by 100 and only turns
                   past YAW_THRESHOLD, so stick drift does not spin the camera.
  {"type": "start", "image_data": "<base64 or data: URL>"}
                   — seed the world with an image.
  {"keys": "wl"} or raw key bytes b"wl"

Protocol chatter (ping/stop/unknown types) decodes to None, meaning "no
change": returning a neutral action instead would cancel keys the player is
still holding. Unparseable input is also None — a malformed browser frame
should drop one tick of input, not end the session.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

# Gamepad right stick at full deflection sends ±100; smaller deflections are
# drift or accidents, not an intent to turn.
YAW_THRESHOLD = 30.0

_KEYS = "wsad"
_RAW_KEYS = set("wsadjl")


@dataclass(frozen=True)
class Control:
    keyboard: Tuple[int, int, int, int] = (0, 0, 0, 0)  # W, S, A, D
    yaw: int = 0  # -1 left, 0, +1 right
    seed_b64: Optional[str] = None

    @property
    def is_seed(self) -> bool:
        return self.seed_b64 is not None


NEUTRAL = Control()


def _keyboard(pressed) -> Tuple[int, int, int, int]:
    return tuple(1 if k in pressed else 0 for k in _KEYS)  # type: ignore[return-value]


def _sign(x: float, threshold: float = 0.0) -> int:
    return 1 if x > threshold else (-1 if x < -threshold else 0)


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return f if f == f else 0.0  # NaN → 0


def strip_data_url(image) -> Optional[str]:
    if not isinstance(image, str) or not image:
        return None
    if image.startswith("data:"):
        image = image.partition(",")[2]
    return image or None


def decode(payload: bytes) -> Optional[Control]:
    """Turn one client message into a Control, or None for "no change".

    Never raises: the action pump runs for the whole session, and one bad
    message must cost a tick of input, not the rest of the session.
    """
    try:
        return _decode(payload)
    except Exception:  # noqa: BLE001 — any malformed shape is "no change"
        return None


def _decode(payload: bytes) -> Optional[Control]:
    if not payload:
        return None

    # JSON before the binary check: `{"keys":"w"}` is also exactly 12 bytes.
    if len(payload) == 12 and payload[:1] not in (b"{", b"["):
        try:
            flags, _pitch, yaw = struct.unpack("<Iff", payload)
        except struct.error:
            return None
        pressed = {k for bit, k in enumerate(_KEYS) if flags & (1 << bit)}
        return Control(keyboard=_keyboard(pressed), yaw=_sign(_num(yaw)))

    try:
        obj = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _from_keys(payload.decode("ascii", errors="ignore"))
    if not isinstance(obj, dict):
        return None

    kind = obj.get("type")
    if kind == "start":
        seed = strip_data_url(obj.get("image_data"))
        return Control(seed_b64=seed) if seed else None
    if kind == "control":
        raw = obj.get("buttons") or []
        if not isinstance(raw, list):
            raw = []
        buttons = {str(b).lower() for b in raw if isinstance(b, (str, int))}
        yaw = _sign(_num(obj.get("mouse_dx")), YAW_THRESHOLD)
        if not yaw:
            yaw = ("l" in buttons) - ("j" in buttons)
        return Control(keyboard=_keyboard(buttons), yaw=yaw)
    if kind is not None:
        return None
    if "keys" in obj:
        keys = str(obj.get("keys") or "")
        return _from_keys(keys) if keys else NEUTRAL  # {"keys": ""} = all released
    return None


def _from_keys(text: str) -> Optional[Control]:
    """Bare key strings, but only when every character is a known key —
    otherwise arbitrary noise containing an 's' would walk the player backwards."""
    text = text.lower()
    if not text or not set(text) <= _RAW_KEYS:
        return None
    return Control(keyboard=_keyboard(set(text)), yaw=("l" in text) - ("j" in text))
