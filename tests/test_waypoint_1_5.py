# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waypoint-1.5 cartridge tests — offline structure, action decode, frames.

Covers everything that does not need the model, a GPU, or world_engine: the
manifest, the action wire formats (catalogue frontend JSON, matrix-game-3
compatibility shapes, binary), and batch JPEG encoding. The engine adapter in
runner.py is exercised by the live smoke test during deployment, not here.
"""

import importlib.util
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

CARTRIDGE_DIR = Path(__file__).resolve().parent.parent / "inference" / "models" / "waypoint-1-5"


def _load(name: str, unique: str):
    # The matrix-game-3 tests also import modules named `actions`/`frames`
    # from their cartridge dir; load ours under unique names so whichever
    # test file runs first doesn't poison sys.modules for the other.
    spec = importlib.util.spec_from_file_location(unique, CARTRIDGE_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique] = mod
    spec.loader.exec_module(mod)
    return mod


actions = _load("actions", "waypoint_actions_test_mod")
frames = _load("frames", "waypoint_frames_test_mod")

W, A, S, D = 87, 65, 83, 68


class TestCartridgeStructure:
    def test_required_files_exist(self):
        for name in ("endpoint.yaml", "runner.py", "Dockerfile", "requirements.txt"):
            assert (CARTRIDGE_DIR / name).is_file(), name

    def test_uses_shared_serve_entrypoint(self):
        assert not (CARTRIDGE_DIR / "server.py").exists()
        assert "lib/serve" in (CARTRIDGE_DIR / "Dockerfile").read_text()

    def test_realtime_mode_inferred(self):
        # No sagemaker.output key → the CDK manifest infers real-time mode.
        content = (CARTRIDGE_DIR / "endpoint.yaml").read_text()
        assert "output:" not in content
        assert "max_concurrent: 1" in content

    def test_single_gpu_pinned(self):
        assert "NPROC_PER_NODE=1" in (CARTRIDGE_DIR / "Dockerfile").read_text()


class TestFrontendControlDecode:
    def test_control_buttons_map_to_vk_codes(self):
        msg = json.dumps({"type": "control", "buttons": ["W", "A"], "mouse_dx": 0, "mouse_dy": 0})
        ctrl = actions.decode(msg.encode())
        assert ctrl.buttons == {W, A}
        assert ctrl.mouse == (0.0, 0.0)
        assert not ctrl.is_seed

    def test_mouse_stick_scaled(self, monkeypatch):
        monkeypatch.setenv("WAYPOINT_MOUSE_SCALE", "10")
        msg = json.dumps({"type": "control", "buttons": [], "mouse_dx": 100, "mouse_dy": -50})
        ctrl = actions.decode(msg.encode())
        assert ctrl.mouse == pytest.approx((10.0, -5.0))

    def test_ijkl_buttons_become_camera(self, monkeypatch):
        monkeypatch.setenv("WAYPOINT_MOUSE_SCALE", "10")
        msg = json.dumps({"type": "control", "buttons": ["W", "L", "I"], "mouse_dx": 0, "mouse_dy": 0})
        ctrl = actions.decode(msg.encode())
        assert ctrl.buttons == {W}  # camera keys removed from the button set
        assert ctrl.mouse == pytest.approx((10.0, -10.0))  # L → right, I → up

    def test_start_message_extracts_seed(self):
        msg = json.dumps({"type": "start", "image_data": "data:image/jpeg;base64,QUJD"})
        ctrl = actions.decode(msg.encode())
        assert ctrl.is_seed
        assert ctrl.seed_b64 == "QUJD"

    def test_start_without_image_is_seed_event_with_no_payload(self):
        ctrl = actions.decode(json.dumps({"type": "start"}).encode())
        assert ctrl.is_seed and ctrl.seed_b64 is None

    def test_ping_is_neutral(self):
        ctrl = actions.decode(json.dumps({"type": "ping", "timestamp": 1}).encode())
        assert ctrl.buttons == set() and ctrl.mouse == (0.0, 0.0) and not ctrl.is_seed


class TestMatrixGameCompatDecode:
    def test_keys_string(self):
        ctrl = actions.decode(json.dumps({"keys": "wa"}).encode())
        assert ctrl.buttons == {W, A}

    def test_keys_with_camera(self, monkeypatch):
        monkeypatch.setenv("WAYPOINT_MOUSE_SCALE", "10")
        ctrl = actions.decode(json.dumps({"keys": "wl"}).encode())
        assert ctrl.buttons == {W}
        assert ctrl.mouse == pytest.approx((10.0, 0.0))  # yaw right

    def test_mg3_mouse_pitch_up_is_negative_dy(self, monkeypatch):
        monkeypatch.setenv("WAYPOINT_MOUSE_SCALE", "10")
        ctrl = actions.decode(json.dumps({"keys": "", "mouse": [0.1, 0.0]}).encode())
        assert ctrl.mouse == pytest.approx((0.0, -10.0))

    def test_keyboard_vector(self):
        ctrl = actions.decode(json.dumps({"keyboard": [1, 0, 0, 1, 0, 0], "mouse": [0, 0]}).encode())
        assert ctrl.buttons == {W, D}

    def test_binary_action(self, monkeypatch):
        monkeypatch.setenv("WAYPOINT_MOUSE_SCALE", "10")
        payload = struct.pack("<Iff", 0b0001, 0.0, 0.1)  # w held, yaw right
        ctrl = actions.decode(payload)
        assert ctrl.buttons == {W}
        assert ctrl.mouse == pytest.approx((10.0, 0.0))

    def test_raw_keys(self):
        assert actions.decode(b"wa").buttons == {W, A}

    def test_noise_is_neutral(self):
        for payload in (b"", b"not json at all", b"\x00" * 5, json.dumps([1, 2]).encode()):
            ctrl = actions.decode(payload)
            assert ctrl.buttons == set() and ctrl.mouse == (0.0, 0.0)


class TestFrames:
    def test_batch_encodes_one_jpeg_per_frame(self):
        batch = np.random.randint(0, 255, size=(4, 32, 64, 3), dtype=np.uint8)
        payloads = frames.encode_batch(batch)
        assert len(payloads) == 4
        for p in payloads:
            assert p[:2] == b"\xff\xd8"  # JPEG SOI marker

    def test_single_frame_promoted_to_batch(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        assert len(frames.encode_batch(frame)) == 1

    def test_float_input_clipped(self):
        batch = np.full((1, 8, 8, 3), 300.0, dtype=np.float32)
        arr = frames.to_uint8_batch(batch)
        assert arr.dtype == np.uint8 and arr.max() == 255

    def test_torch_tensor_accepted(self):
        torch = pytest.importorskip("torch")
        if getattr(torch, "_wm_stub", False):
            # test_app_integration installs a torch stub in sys.modules so the
            # app imports without a GPU; it cannot make tensors.
            pytest.skip("sys.modules['torch'] is the app-import stub")
        batch = torch.zeros((2, 8, 8, 3), dtype=torch.uint8)
        assert len(frames.encode_batch(batch)) == 2

    def test_default_is_420_chroma(self):
        # Delivered fps is bytes-bound, so the default subsampling is a
        # performance decision worth pinning: 4:4:4 costs 17.9% more bytes.
        assert frames.DEFAULT_SUBSAMPLING == "420"

    def test_subsampling_is_honoured(self):
        # A colourful gradient: 4:2:0 halves the chroma planes, so it must be
        # smaller than 4:4:4 on an image that actually has colour detail.
        h = w = 128
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[..., 0] = np.arange(w, dtype=np.uint8)[None, :] * 2
        frame[..., 1] = np.arange(h, dtype=np.uint8)[:, None] * 2
        frame[..., 2] = 255 - frame[..., 0]
        small = frames.encode_jpeg(frame, subsampling="420")
        large = frames.encode_jpeg(frame, subsampling="444")
        assert len(small) < len(large)

    def test_pil_fallback_matches_subsampling_codes(self, monkeypatch):
        # Containers have simplejpeg; laptops and CI may not. Both paths must
        # accept the same string names and produce a decodable JPEG.
        Image = pytest.importorskip("PIL.Image")
        monkeypatch.setattr(frames, "simplejpeg", None)
        frame = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
        sizes = {}
        for sub in ("444", "422", "420"):
            payload = frames.encode_jpeg(frame, subsampling=sub)
            assert Image.open(io.BytesIO(payload)).size == (32, 32)
            sizes[sub] = len(payload)
        assert sizes["420"] <= sizes["422"] <= sizes["444"]

    def test_pil_fallback_defaults_to_420_on_unknown_name(self, monkeypatch):
        monkeypatch.setattr(frames, "simplejpeg", None)
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        assert frames.encode_jpeg(frame, subsampling="nonsense")[:2] == b"\xff\xd8"

    def test_advertised_subsamplings_are_the_ones_both_backends_take(self):
        # runner.py validates WAYPOINT_JPEG_SUBSAMPLING against this set, so it
        # must not advertise a name the PIL fallback would silently ignore.
        assert frames.SUBSAMPLINGS == {"420", "422", "444"}
        assert frames.DEFAULT_SUBSAMPLING in frames.SUBSAMPLINGS

    @pytest.mark.parametrize("configured,expected", [
        ("444", "444"),
        (" 422 ", "422"),          # env vars pick up stray whitespace
        ("411", "420"),            # simplejpeg knows it, PIL does not
        ("nonsense", "420"),
        ("", "420"),
        (None, "420"),             # unset
    ])
    def test_configured_subsampling_resolves_or_falls_back(self, configured, expected):
        # A bad env var must cost one log line at setup, not an exception per
        # frame inside a live session.
        assert frames.resolve_subsampling(configured) == expected
