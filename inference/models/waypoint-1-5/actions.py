# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action decoding for the Waypoint-1.5 cartridge.

world_engine's CtrlInput wants:
  button:  a set of int key codes (Windows virtual-key codes — W=87, A=65,
           S=83, D=68, SPACE=32, SHIFT=16 — the Owl-Control convention that
           Overworld's own Biome client uses)
  mouse:   (dx, dy) raw deltas per generated step; positive dx looks right,
           positive dy looks down

The catalogue frontend sends JSON text messages at an input-poll interval:
  {"type": "control", "buttons": ["W","A"], "mouse_dx": -100..100, "mouse_dy": ...}
where mouse_dx/dy are a stick position scaled by 100, NOT a raw delta — so we
rescale by MOUSE_SCALE to get a per-step delta the model was trained on.

For parity with the matrix-game-3 cartridge we also accept its formats:
  {"keys": "wa", "mouse": [pitch, yaw]}   pitch/yaw in ±0.1 CAM_VALUE units
  12-byte binary: <uint32 wsad bitflags><float32 pitch><float32 yaw>
  raw key bytes, e.g. b"wa"

Anything unparseable yields the neutral action rather than raising: a
malformed frame from a browser should drop input for one tick, not kill the
session. Kept separate from runner.py so it is testable without a GPU.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Optional, Set, Tuple

# Full-deflection stick → this many mouse counts per generated step. Matches
# a moderate hand-mouse turn speed; override with WAYPOINT_MOUSE_SCALE.
DEFAULT_MOUSE_SCALE = 15.0

# matrix-game-3's per-tick camera unit, used to normalise its ±0.1 payloads.
MG3_CAM_VALUE = 0.1

# Windows virtual-key codes for the movement keys the frontends send.
VK = {
    **{chr(c): c for c in range(ord("A"), ord("Z") + 1)},
    **{str(d): ord(str(d)) for d in range(10)},
    "SPACE": 0x20,
    "SHIFT": 0x10,
    "CTRL": 0x11,
    "TAB": 0x09,
    "ENTER": 0x0D,
    "UP": 0x26,
    "DOWN": 0x28,
    "LEFT": 0x25,
    "RIGHT": 0x27,
    "MOUSE_LEFT": 0x01,
    "MOUSE_RIGHT": 0x02,
    "MOUSE_MIDDLE": 0x04,
}


def mouse_scale() -> float:
    try:
        return float(os.environ.get("WAYPOINT_MOUSE_SCALE", DEFAULT_MOUSE_SCALE))
    except ValueError:
        return DEFAULT_MOUSE_SCALE


@dataclass
class Control:
    """Decoded client intent for one step."""

    buttons: Set[int] = field(default_factory=set)
    mouse: Tuple[float, float] = (0.0, 0.0)  # (dx, dy)
    # Base64 seed image from a {"type":"start","image_data":...} message; the
    # runner reseeds the world with it instead of treating it as input.
    seed_b64: Optional[str] = None
    is_seed: bool = False


def neutral() -> Control:
    return Control()


def decode(payload: bytes) -> Control:
    """Turn one client message into a Control."""
    if not payload:
        return neutral()

    if len(payload) == 12:
        try:
            flags, pitch, yaw = struct.unpack("<Iff", payload)
        except struct.error:
            return neutral()
        buttons = {VK[k.upper()] for bit, k in enumerate("wsad") if flags & (1 << bit)}
        return Control(buttons=buttons, mouse=_mg3_mouse(pitch, yaw))

    try:
        obj = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _from_raw_keys(payload.decode("ascii", errors="ignore"))

    if not isinstance(obj, dict):
        return neutral()

    msg_type = obj.get("type")

    if msg_type == "start":
        image = obj.get("image_data")
        return Control(seed_b64=_strip_data_url(image), is_seed=True)

    if msg_type == "control":
        buttons = {
            VK[str(b).upper()]
            for b in obj.get("buttons", [])
            if isinstance(b, (str, int)) and str(b).upper() in VK
        }
        scale = mouse_scale() / 100.0
        dx = _f(obj.get("mouse_dx")) * scale
        dy = _f(obj.get("mouse_dy")) * scale
        ctrl = Control(buttons=buttons, mouse=(dx, dy))
        _apply_ijkl_from_buttons(obj.get("buttons", []), ctrl)
        return ctrl

    # Non-input protocol chatter (ping/stop/...) — ignore.
    if msg_type is not None:
        return neutral()

    # matrix-game-3 JSON shapes.
    if "keyboard" in obj:
        kb = obj.get("keyboard") or []
        buttons = set()
        for idx, key in enumerate("WSAD"):
            if idx < len(kb) and _f(kb[idx]) > 0.5:
                buttons.add(VK[key])
        mouse = obj.get("mouse") or [0.0, 0.0]
        return Control(buttons=buttons, mouse=_mg3_mouse(_f(_at(mouse, 0)), _f(_at(mouse, 1))))

    if "keys" in obj:
        ctrl = _from_keys(str(obj.get("keys", "")))
        if "mouse" in obj:
            mouse = obj.get("mouse") or [0.0, 0.0]
            ctrl.mouse = _mg3_mouse(_f(_at(mouse, 0)), _f(_at(mouse, 1)))
        return ctrl

    return neutral()


def _strip_data_url(image) -> Optional[str]:
    if not isinstance(image, str) or not image:
        return None
    if image.startswith("data:"):
        _, _, b64 = image.partition(",")
        return b64 or None
    return image


_MG3_KEYS = set("wsadqijklu")


def _from_raw_keys(text: str) -> Control:
    """Bare key strings, but only when the whole payload is known keys —
    otherwise arbitrary noise containing 's' would walk the player backwards."""
    if not text or not set(text.lower()) <= _MG3_KEYS:
        return neutral()
    return _from_keys(text)


def _from_keys(keys: str) -> Control:
    """WASD moves; IJKL is the matrix-game-3 camera convention."""
    buttons: Set[int] = set()
    pitch = yaw = 0.0
    for ch in keys.lower():
        if ch in "wsad":
            buttons.add(VK[ch.upper()])
        elif ch == "i":
            pitch += MG3_CAM_VALUE
        elif ch == "k":
            pitch -= MG3_CAM_VALUE
        elif ch == "j":
            yaw -= MG3_CAM_VALUE
        elif ch == "l":
            yaw += MG3_CAM_VALUE
    return Control(buttons=buttons, mouse=_mg3_mouse(pitch, yaw))


def _apply_ijkl_from_buttons(buttons_raw, ctrl: Control) -> None:
    """Frontend camera keys arrive as buttons I/J/K/L when a client speaks the
    button list dialect; fold them into the mouse delta and drop them from the
    button set so the model doesn't see them as gameplay keys."""
    scale = mouse_scale()
    for b in buttons_raw:
        name = str(b).upper()
        if name == "I":
            ctrl.mouse = (ctrl.mouse[0], ctrl.mouse[1] - scale)
        elif name == "K":
            ctrl.mouse = (ctrl.mouse[0], ctrl.mouse[1] + scale)
        elif name == "J":
            ctrl.mouse = (ctrl.mouse[0] - scale, ctrl.mouse[1])
        elif name == "L":
            ctrl.mouse = (ctrl.mouse[0] + scale, ctrl.mouse[1])
        else:
            continue
        ctrl.buttons.discard(VK[name])


def _mg3_mouse(pitch: float, yaw: float) -> Tuple[float, float]:
    """matrix-game-3 [pitch, yaw] in ±CAM_VALUE → Waypoint (dx, dy) counts.
    Positive pitch looks up, which is negative dy in screen convention."""
    scale = mouse_scale()
    lim = MG3_CAM_VALUE
    pitch = max(-lim, min(lim, pitch))
    yaw = max(-lim, min(lim, yaw))
    return (yaw / lim * scale, -pitch / lim * scale)


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _at(seq, i):
    try:
        return seq[i]
    except (IndexError, TypeError, KeyError):
        return 0.0
