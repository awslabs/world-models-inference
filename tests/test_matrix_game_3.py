# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matrix Game 3.0 cartridge tests — offline structure, action decode, frames.

What is covered here is everything that does not need the model, a GPU, or the
upstream package: the manifest, the action wire format, and the tensor-to-JPEG
conversion. The pipeline adapter in runner.py is exercised by the live smoke
test in the deployment docs, not here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

CARTRIDGE_DIR = Path(__file__).resolve().parent.parent / "inference" / "models" / "matrix-game-3"
sys.path.insert(0, str(CARTRIDGE_DIR))

import actions  # noqa: E402
import frames  # noqa: E402


class TestCartridgeStructure:
    def test_required_files_exist(self):
        for name in ("endpoint.yaml", "runner.py", "Dockerfile", "requirements.txt"):
            assert (CARTRIDGE_DIR / name).is_file(), name

    def test_uses_shared_serve_entrypoint(self):
        # A cartridge must not ship its own server: the shared lib.serve
        # entrypoint is what applies auth, CORS and rate limiting.
        assert not (CARTRIDGE_DIR / "server.py").exists()
        assert "lib.serve" in (CARTRIDGE_DIR / "Dockerfile").read_text()

    def test_targets_p5_on_both_paths(self):
        content = (CARTRIDGE_DIR / "endpoint.yaml").read_text()
        assert "p5.48xlarge" in content
        assert "ml.p5.48xlarge" in content

    def test_manifest_has_no_sagemaker_output_so_mode_is_realtime(self):
        # cdk inferMode() reads real-time from the absence of sagemaker.output.
        # An output bucket here would silently make this an async endpoint.
        content = (CARTRIDGE_DIR / "endpoint.yaml").read_text()
        assert "output:" not in content

    def test_manifest_pins_weights_repo(self):
        assert "Skywork/Matrix-Game-3.0" in (CARTRIDGE_DIR / "endpoint.yaml").read_text()

    def test_manifest_declares_its_own_base_image(self):
        # Upstream needs torch 2.10; the shared default image is torch 2.4.
        assert "base_image:" in (CARTRIDGE_DIR / "endpoint.yaml").read_text()

    def test_runner_declares_streaming_and_refuses_batch(self):
        content = (CARTRIDGE_DIR / "runner.py").read_text()
        assert "def stream(" in content
        assert "def supports_streaming" in content
        assert "NotImplementedError" in content

    def test_dockerfile_pins_upstream_ref(self):
        content = (CARTRIDGE_DIR / "Dockerfile").read_text()
        assert "MATRIX_GAME_REF" in content
        assert "git checkout" in content


class TestActionDecoding:
    def test_neutral_shapes_match_upstream(self):
        keyboard, mouse = actions.neutral()
        assert len(keyboard) == actions.KEYBOARD_DIM == 6
        assert len(mouse) == actions.MOUSE_DIM == 2

    @pytest.mark.parametrize("key,index", [("w", 0), ("s", 1), ("a", 2), ("d", 3)])
    def test_movement_keys_map_to_upstream_indices(self, key, index):
        keyboard, _ = actions.decode(f'{{"keys": "{key}"}}'.encode())
        assert keyboard[index] == 1.0
        assert sum(keyboard) == 1.0

    def test_mouse_axis_order_is_pitch_then_yaw(self):
        # Upstream indexes [pitch, yaw]; swapping these swaps look-up with
        # look-sideways, which is invisible in code review but obvious in play.
        _, up = actions.decode(b'{"keys": "i"}')
        assert up[0] == pytest.approx(actions.CAM_VALUE)
        assert up[1] == 0.0

        _, right = actions.decode(b'{"keys": "l"}')
        assert right[0] == 0.0
        assert right[1] == pytest.approx(actions.CAM_VALUE)

    def test_combined_keys_and_camera(self):
        keyboard, mouse = actions.decode(b'{"keys": "wl"}')
        assert keyboard[0] == 1.0
        assert mouse[1] == pytest.approx(actions.CAM_VALUE)

    def test_binary_frame_decodes(self):
        import struct

        payload = struct.pack("<Iff", 0b0001, 0.05, -0.05)  # 'w' held
        keyboard, mouse = actions.decode(payload)
        assert keyboard[0] == 1.0
        assert mouse == pytest.approx([0.05, -0.05])

    def test_explicit_vectors_pass_through(self):
        keyboard, mouse = actions.decode(
            b'{"keyboard": [0,0,0,1,0,0], "mouse": [0.1, 0.0]}'
        )
        assert keyboard[3] == 1.0
        assert mouse[0] == pytest.approx(0.1)

    def test_camera_delta_is_clamped(self):
        # An unclamped delta teleports the camera and breaks pose continuity,
        # which destroys the model's scene memory.
        _, mouse = actions.decode(b'{"keys": "", "mouse": [99.0, -99.0]}')
        assert mouse == pytest.approx([actions.CAM_VALUE, -actions.CAM_VALUE])

    @pytest.mark.parametrize("payload", [b"", b"not json", b"\x00\x01", b"[]", b"{}"])
    def test_malformed_input_yields_neutral_not_an_exception(self, payload):
        # One bad frame from a browser should cost one tick of input, not the
        # whole session.
        keyboard, mouse = actions.decode(payload)
        assert sum(abs(v) for v in keyboard) == 0.0
        assert sum(abs(v) for v in mouse) == 0.0

    def test_raw_key_bytes_are_accepted(self):
        keyboard, _ = actions.decode(b"a")
        assert keyboard[2] == 1.0

    def test_raw_keys_are_case_insensitive(self):
        keyboard, _ = actions.decode(b"WA")
        assert keyboard[0] == 1.0
        assert keyboard[2] == 1.0

    def test_twelve_byte_json_is_parsed_as_json_not_binary(self):
        # b'{"keys":"w"}' is exactly 12 bytes, and any 12 bytes "unpack" as the
        # binary struct — without the '{' guard this held w+s and panned the
        # camera from float garbage.
        payload = b'{"keys":"w"}'
        assert len(payload) == 12
        keyboard, mouse = actions.decode(payload)
        assert keyboard[0] == 1.0
        assert sum(keyboard) == 1.0
        assert mouse == [0.0, 0.0]

    def test_binary_frame_holds_multiple_keys(self):
        import struct

        payload = struct.pack("<Iff", 0b1111, 0.0, 0.0)
        keyboard, _ = actions.decode(payload)
        assert keyboard[:4] == [1.0, 1.0, 1.0, 1.0]

    def test_binary_mouse_is_clamped_like_json_mouse(self):
        import struct

        payload = struct.pack("<Iff", 0, 5.0, -5.0)
        _, mouse = actions.decode(payload)
        assert mouse == pytest.approx([actions.CAM_VALUE, -actions.CAM_VALUE])

    def test_opposing_camera_keys_cancel(self):
        _, mouse = actions.decode(b'{"keys": "ik"}')
        assert mouse == [0.0, 0.0]
        _, mouse = actions.decode(b'{"keys": "jl"}')
        assert mouse == [0.0, 0.0]

    def test_short_keyboard_vector_is_padded_and_bad_mouse_ignored(self):
        keyboard, mouse = actions.decode(b'{"keyboard": [1], "mouse": "bad"}')
        assert keyboard == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert mouse == [0.0, 0.0]


class TestFrameEncoding:
    def test_chw_float_becomes_hwc_uint8(self):
        frame = np.zeros((3, 8, 16), dtype=np.float32)
        out = frames.to_uint8(frame)
        assert out.shape == (8, 16, 3)
        assert out.dtype == np.uint8

    def test_value_range_maps_to_full_byte_range(self):
        assert frames.to_uint8(np.full((3, 2, 2), -1.0, dtype=np.float32)).min() == 0
        assert frames.to_uint8(np.full((3, 2, 2), 1.0, dtype=np.float32)).max() == 255

    def test_out_of_range_is_clipped_not_wrapped(self):
        out = frames.to_uint8(np.full((3, 2, 2), 4.0, dtype=np.float32))
        assert out.max() == 255

    def test_uint8_input_passes_through(self):
        frame = np.full((4, 4, 3), 120, dtype=np.uint8)
        assert frames.to_uint8(frame).tolist() == frame.tolist()

    def test_encode_jpeg_produces_a_jpeg(self):
        data = frames.encode_jpeg(np.zeros((3, 16, 16), dtype=np.float32))
        assert data[:2] == b"\xff\xd8"  # JPEG SOI
        assert data[-2:] == b"\xff\xd9"  # JPEG EOI

    def test_split_chunk_walks_the_time_axis(self):
        chunk = np.zeros((1, 3, 5, 8, 8), dtype=np.float32)  # (B, C, T, H, W)
        out = list(frames.split_chunk(chunk))
        assert len(out) == 5
        assert out[0].shape == (3, 8, 8)

    def test_split_chunk_accepts_a_single_frame(self):
        assert len(list(frames.split_chunk(np.zeros((3, 8, 8))))) == 1

    def test_split_chunk_rejects_unexpected_shape(self):
        with pytest.raises(ValueError):
            list(frames.split_chunk(np.zeros((4,))))

    def test_to_uint8_rejects_non_image(self):
        with pytest.raises(ValueError):
            frames.to_uint8(np.zeros((8, 8)))

    def test_hwc_float_keeps_its_orientation(self):
        out = frames.to_uint8(np.zeros((8, 16, 3), dtype=np.float32))
        assert out.shape == (8, 16, 3)

    def test_grayscale_frame_encodes_as_rgb_jpeg(self):
        data = frames.encode_jpeg(np.zeros((1, 16, 16), dtype=np.float32))
        assert data[:2] == b"\xff\xd8"

    def test_split_chunk_takes_first_batch_item(self):
        chunk = np.zeros((2, 3, 4, 8, 8), dtype=np.float32)
        assert len(list(frames.split_chunk(chunk))) == 4

    def test_split_chunk_handles_torch_like_tensors_without_torch(self):
        # The VAE hands back bfloat16 torch tensors; split_chunk must go through
        # .detach().float().cpu().numpy() without this module importing torch.
        class FakeTensor:
            def __init__(self, arr):
                self._arr = arr

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._arr

        out = list(frames.split_chunk(FakeTensor(np.zeros((1, 3, 2, 4, 4)))))
        assert len(out) == 2
        assert out[0].shape == (3, 4, 4)


class TestLockstepChunkMath:
    """The frame-window arithmetic the multi-GPU loop shares with upstream.

    Upstream: first chunk denoises pixel frames [0, 57); every later chunk
    re-denoises a 56-frame window overlapping the previous one by 16, so 40 of
    its frames are new. These invariants are what keeps the frames we emit and
    the latent indices for memory selection consistent with their pipeline.
    """

    def test_first_chunk_window(self):
        import lockstep
        assert lockstep.chunk_frame_window(0) == (0, 57)

    def test_later_windows_overlap_by_sixteen(self):
        import lockstep
        for idx in range(1, 6):
            start, end = lockstep.chunk_frame_window(idx)
            assert end - start == 56
            prev_end = lockstep.chunk_frame_window(idx - 1)[1]
            assert prev_end - start == 16

    def test_new_frames_match_upstream_counts(self):
        import lockstep
        assert lockstep.new_frames_in_chunk(0) == 57
        assert all(lockstep.new_frames_in_chunk(i) == 40 for i in range(1, 5))

    def test_total_frames_accumulate_like_upstream(self):
        # Upstream: num_frames = 57 + (n - 1) * 40.
        import lockstep
        n = 7
        total = sum(lockstep.new_frames_in_chunk(i) for i in range(n))
        assert total == 57 + (n - 1) * 40
        assert lockstep.chunk_frame_window(n - 1)[1] == total

    def test_latent_index_matches_upstream_formula(self):
        import lockstep
        assert lockstep.get_latent_idx(0) == 0  # (0-1)//4+1 floor-divides
        assert lockstep.get_latent_idx(57) == 15
        assert lockstep.get_latent_idx(97) == 25

    def test_align_frame_to_block(self):
        import lockstep
        assert lockstep.align_frame_to_block(0) == 1
        assert lockstep.align_frame_to_block(1) == 1
        assert lockstep.align_frame_to_block(5) == 5
        assert lockstep.align_frame_to_block(7) == 5

    def test_module_imports_without_torch_or_upstream(self):
        # The helpers must stay usable on a laptop; heavy imports live inside
        # LockstepSession's methods (same rule as frames.py).
        import lockstep
        src = (CARTRIDGE_DIR / "lockstep.py").read_text()
        head = src.split("class LockstepSession")[0]
        assert "\nimport torch" not in head
        assert lockstep is not None
