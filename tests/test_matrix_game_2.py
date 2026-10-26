# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matrix Game 2 cartridge tests — structure, action decode, frames, pacing.

Everything here runs without the model, a GPU or the upstream repo. The engine's
pure helpers are tested only when real torch is importable; the session
behaviour of runner.py is in test_matrix_game_2_session.py.
"""

import base64
import importlib.util
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

CARTRIDGE_DIR = Path(__file__).resolve().parent.parent / "inference" / "models" / "matrix-game-2"


def _load(name: str, unique: str):
    # Other cartridges ship modules called `actions`/`frames`; load ours under
    # unique names so test ordering cannot poison sys.modules.
    spec = importlib.util.spec_from_file_location(unique, CARTRIDGE_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique] = mod
    spec.loader.exec_module(mod)
    return mod


actions = _load("actions", "mg2_actions_test_mod")
frames = _load("frames", "mg2_frames_test_mod")


def binary(keys: str = "", pitch: float = 0.0, yaw: float = 0.0) -> bytes:
    """What frontend/src/components/RealtimePlayer.tsx sends each tick."""
    flags = sum(1 << i for i, k in enumerate("wsad") if k in keys)
    return struct.pack("<Iff", flags, pitch, yaw)


class TestCartridgeStructure:
    def test_required_files_exist(self):
        for name in ("endpoint.yaml", "runner.py", "engine.py", "Dockerfile",
                     "requirements.txt", "README.md"):
            assert (CARTRIDGE_DIR / name).is_file(), name

    def test_realtime_mode_inferred(self):
        content = (CARTRIDGE_DIR / "endpoint.yaml").read_text()
        assert "output:" not in content
        assert "max_concurrent: 1" in content

    def test_single_gpu_pinned(self):
        assert "NPROC_PER_NODE=1" in (CARTRIDGE_DIR / "Dockerfile").read_text()

    def test_toolchain_pins_that_worked_on_the_booth(self):
        # torch 2.9 has no cp310 flash-attn wheel and 2.6 lacks recompile_limit;
        # a drive-by bump of any of these breaks the image or the compile.
        dockerfile = (CARTRIDGE_DIR / "Dockerfile").read_text()
        assert "torch==2.8.0" in dockerfile
        assert "cu12torch2.8cxx11abiFALSE-cp310" in dockerfile
        assert "ubuntu22.04" in dockerfile  # system Python 3.10

    def test_upstream_is_pinned_to_a_commit(self):
        dockerfile = (CARTRIDGE_DIR / "Dockerfile").read_text()
        assert "MATRIX_GAME_REF=71c3cd7f741311f8100f6cf9cde942b6c1378d11" in dockerfile

    def test_requirements_do_not_pull_torch(self):
        lines = [l.split("#")[0].strip().lower()
                 for l in (CARTRIDGE_DIR / "requirements.txt").read_text().splitlines()]
        assert not [l for l in lines if l.startswith(("torch", "flash-attn", "flash_attn"))]


class TestBinaryActions:
    def test_neutral(self):
        assert actions.decode(binary()) == actions.NEUTRAL

    @pytest.mark.parametrize("keys,expected", [
        ("w", (1, 0, 0, 0)), ("s", (0, 1, 0, 0)), ("a", (0, 0, 1, 0)),
        ("d", (0, 0, 0, 1)), ("wd", (1, 0, 0, 1)),
    ])
    def test_wasd_bits(self, keys, expected):
        assert actions.decode(binary(keys)).keyboard == expected

    def test_yaw_sign_turns_and_pitch_is_ignored(self):
        assert actions.decode(binary(yaw=0.1)).yaw == 1
        assert actions.decode(binary(yaw=-0.1)).yaw == -1
        assert actions.decode(binary(pitch=0.1)).yaw == 0

    def test_nan_yaw_is_no_turn(self):
        assert actions.decode(binary(yaw=float("nan"))).yaw == 0


class TestJsonActions:
    def test_kiosk_control_buttons(self):
        ctrl = actions.decode(json.dumps({"type": "control", "buttons": ["W", "A"]}).encode())
        assert ctrl.keyboard == (1, 0, 1, 0) and ctrl.yaw == 0

    def test_jl_buttons_turn(self):
        assert actions.decode(json.dumps({"type": "control", "buttons": ["L"]}).encode()).yaw == 1
        assert actions.decode(json.dumps({"type": "control", "buttons": ["J"]}).encode()).yaw == -1

    @pytest.mark.parametrize("dx,yaw", [(100, 1), (-100, -1), (29, 0), (-29, 0), (31, 1)])
    def test_stick_turns_only_past_threshold(self, dx, yaw):
        msg = {"type": "control", "buttons": [], "mouse_dx": dx, "mouse_dy": 80}
        assert actions.decode(json.dumps(msg).encode()).yaw == yaw

    def test_empty_control_releases_everything(self):
        # The kiosk sends this when the last key goes up; the server would
        # otherwise keep walking on the last held action.
        assert actions.decode(b'{"type":"control","buttons":[]}') == actions.NEUTRAL

    def test_start_carries_a_seed(self):
        assert actions.decode(b'{"type":"start","image_data":"QUJD"}').seed_b64 == "QUJD"

    def test_start_accepts_a_data_url(self):
        msg = {"type": "start", "image_data": "data:image/png;base64,QUJD"}
        assert actions.decode(json.dumps(msg).encode()).seed_b64 == "QUJD"

    def test_start_without_image_is_ignored(self):
        assert actions.decode(b'{"type":"start"}') is None

    @pytest.mark.parametrize("payload", [b'{"type":"ping"}', b'{"type":"stop"}', b"[1,2]", b"{}"])
    def test_chatter_is_no_change_not_neutral(self, payload):
        # Returning NEUTRAL here would cancel keys the player is still holding.
        assert actions.decode(payload) is None

    def test_keys_dialect(self):
        assert actions.decode(b'{"keys":"wl"}') == actions.Control(keyboard=(1, 0, 0, 0), yaw=1)
        assert actions.decode(b'{"keys":""}') == actions.NEUTRAL

    def test_raw_keys_only_when_every_char_is_a_key(self):
        assert actions.decode(b"wa").keyboard == (1, 0, 1, 0)
        assert actions.decode(b"hello, sad world") is None

    def test_twelve_byte_json_is_not_read_as_binary(self):
        # `{"keys":"w"}` is exactly 12 bytes; as <Iff it would press W+S+D and
        # turn right at a huge yaw.
        payload = b'{"keys":"w"}'
        assert len(payload) == 12
        assert actions.decode(payload) == actions.Control(keyboard=(1, 0, 0, 0), yaw=0)

    @pytest.mark.parametrize("payload", [
        b'{"type":"control","buttons":5}',
        b'{"type":"control","buttons":{"w":1}}',
        b'{"type":"control","buttons":[],"mouse_dx":' + b"9" * 400 + b"}",
        b"[" * 5000 + b"]" * 5000,
        b'{"type":"start","image_data":7}',
    ])
    def test_malformed_messages_never_raise(self, payload):
        # The pump runs for the whole session: an exception here used to kill
        # it and freeze the player's input until they reconnected.
        result = actions.decode(payload)
        assert result is None or isinstance(result, actions.Control)

    def test_garbage_is_ignored(self):
        assert actions.decode(b"\xff\xfe\x00") is None
        assert actions.decode(b"") is None


class TestFrames:
    def test_encode_round_trips_size(self):
        from PIL import Image

        frame = np.full((352, 640, 3), 128, dtype=np.uint8)
        jpeg = frames.encode_jpeg(frame)
        assert jpeg[:2] == b"\xff\xd8"
        assert Image.open(io.BytesIO(jpeg)).size == (640, 352)

    def test_batch_is_one_jpeg_per_frame(self):
        batch = np.zeros((12, 32, 32, 3), dtype=np.uint8)
        assert len(frames.encode_batch(batch)) == 12

    @pytest.mark.parametrize("value,expected", [
        (None, 80), ("", 80), ("abc", 80), ("92", 92), (" 70 ", 70), ("0", 1), ("200", 100),
    ])
    def test_resolve_quality(self, value, expected):
        assert frames.resolve_quality(value) == expected


class TestPacer:
    def test_first_chunk_is_ignored(self):
        # It carries seed encode (and on a cold start, compile), so it would
        # set an interval far too slow for the steady state.
        p = frames.Pacer(initial=0.05)
        assert p.observe(10.0, 9) == 0.05

    def test_converges_to_generation_time_per_frame(self):
        p = frames.Pacer()
        p.observe(1.0, 9)
        for _ in range(10):
            interval = p.observe(0.72, 12)
        assert interval == pytest.approx(0.72 / 12 - frames.Pacer.OVERHEAD, rel=0.01)

    def test_never_faster_than_the_floor(self):
        p = frames.Pacer()
        p.observe(1.0, 9)
        for _ in range(10):
            interval = p.observe(0.001, 12)
        assert interval == pytest.approx(frames.Pacer.FLOOR, rel=0.01)


class TestManifestLicence:
    def test_licence_fields(self):
        import yaml

        manifest = yaml.safe_load((CARTRIDGE_DIR / "endpoint.yaml").read_text())
        assert manifest["license"] == "MIT"
        assert manifest["license_restricted"] is False
        assert manifest["license_url"] == "https://huggingface.co/Skywork/Matrix-Game-2.0"
        assert manifest["ec2"]["instance"] == "g6e.4xlarge"


def _real_torch():
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch, "_wm_stub", False):
        return None
    try:
        import torch  # noqa: F811
    except ImportError:
        return None
    return torch


@pytest.fixture(scope="module")
def engine():
    return _load("engine", "mg2_engine_test_mod")


@pytest.mark.skipif(_real_torch() is None, reason="needs real torch")
class TestEngineHelpers:
    def test_first_chunk_covers_nine_pixel_frames(self, engine):
        assert engine.chunk_pixel_span(0, 10_000) == (0, 9)

    def test_later_chunks_cover_twelve_contiguous_frames(self, engine):
        spans = [engine.chunk_pixel_span(i, 10_000) for i in range(4)]
        assert spans[1] == (9, 21)
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))

    def test_span_is_clipped_at_the_end(self, engine):
        assert engine.chunk_pixel_span(2, 30) == (21, 30)
        start, end = engine.chunk_pixel_span(5, 30)
        assert end <= start  # past the end: no frames, the action is skipped

    def test_action_tensors_zero_pitch_and_ternary_yaw(self, engine):
        kbd, mouse = engine.action_tensors((1, 0, 0, 1), 1, "cpu")
        assert kbd.tolist() == [1, 0, 0, 1]
        assert mouse.float().tolist() == pytest.approx([0.0, 0.1], abs=1e-3)
        assert engine.action_tensors((0, 0, 0, 0), -1, "cpu")[1].float().tolist() == \
            pytest.approx([0.0, -0.1], abs=1e-3)

    def test_resizecrop_matches_target_aspect(self, engine):
        from PIL import Image

        out = engine.resizecrop(Image.new("RGB", (1000, 1000)), 352, 640)
        assert out.size[0] / out.size[1] == pytest.approx(640 / 352, rel=0.01)
