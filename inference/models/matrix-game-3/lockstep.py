# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lockstep chunk loop for multi-GPU Matrix Game 3.0 sessions.

This is a port of the per-chunk body of upstream's interactive generate()
(pipeline/inference_interactive_pipeline.py at the pinned commit 71c3cd7),
restructured so that every rank executes an identical sequence of work. The
approach is the one validated at ~20 fps on 8 GPUs in the world-models-infra
`rahul-mg3` engine.

Why upstream's own loop cannot be driven multi-GPU from a server:

  * It reads the action only on rank 0 (get_current_action is called inside an
    `if self.rank == 0:` block), then rank 0 broadcasts the accumulated
    condition tensors to the other ranks. Ending a session by raising inside
    the action hook therefore unwinds rank 0 alone, past the dist.barrier()
    where ranks 1..N-1 are still waiting — the session deadlocks and every
    session after it inherits a wedged process group.

  * Those condition broadcasts use dist.broadcast_object_list on CUDA tensors,
    which pickles each tensor together with its device: every follower
    receives tensors tagged cuda:0 and dies in the DiT's first torch.cat.

This loop removes both hazards instead of patching around them:

  * The client action travels as a dozen CPU floats on the framework's gloo
    command channel, broadcast at the top of every chunk. The stop decision
    rides the same message, so all ranks always leave the loop together.

  * Every rank then computes the condition tensors locally from those floats —
    the arithmetic is deterministic, so no tensor ever needs to cross ranks.
    The only collectives left are the seed-image broadcast at session start,
    the memory-index broadcast upstream also does, and the DiT's own Ulysses
    collectives — all of them symmetric by construction.

The VAE decode runs on rank 0 only and contains no collectives, exactly as in
upstream's loop.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

# torch is imported inside LockstepSession, not here: the frame-window helpers
# below are pure arithmetic, and keeping the module importable without torch is
# what lets the tests exercise them on a laptop (same rule as frames.py).

logger = logging.getLogger(__name__)

# Chunk layout, from upstream's generate(): the first chunk covers 57 pixel
# frames; each later chunk re-denoises a 56-frame window that overlaps the
# previous one by 16, so 40 frames of it are new.
FIRST_CLIP_FRAME = 57
CLIP_FRAME = 56
PAST_FRAME = 16
STEP_FRAME = CLIP_FRAME - PAST_FRAME

# Upstream selects 5 past latent frames as episodic memory.
MEMORY_FRAMES = 5

# Upstream's latent channel count for matrix_game3 (hardcoded 48 in their loop).
LATENT_CHANNELS = 48


def chunk_frame_window(chunk_idx: int) -> tuple[int, int]:
    """Pixel-frame [start, end) that chunk `chunk_idx` denoises."""
    if chunk_idx == 0:
        return 0, FIRST_CLIP_FRAME
    end = FIRST_CLIP_FRAME + chunk_idx * STEP_FRAME
    return end - CLIP_FRAME, end


def new_frames_in_chunk(chunk_idx: int) -> int:
    """How many frames of chunk `chunk_idx` the client has not seen before."""
    return FIRST_CLIP_FRAME if chunk_idx == 0 else STEP_FRAME


def align_frame_to_block(frame_idx: int) -> int:
    return (frame_idx - 1) // 4 * 4 + 1 if frame_idx > 0 else 1


def get_latent_idx(frame_idx: int) -> int:
    return (frame_idx - 1) // 4 + 1


class LockstepSession:
    """One interactive session where all ranks advance chunk by chunk.

    Construct and drive identically on every rank; only rank 0 gets frames
    back from run_chunk(). The caller owns action distribution — this class
    performs no collectives on the action itself.
    """

    def __init__(self, pipe, cfg, args, device, rank: int, anchor_image,
                 text_cond, neg_cond=None) -> None:
        import torch
        import torch.distributed as dist

        self.pipe = pipe
        self.cfg = cfg
        self.args = args
        self.device = device
        self.rank = rank
        self.chunk_idx = 0

        self._dtype = torch.bfloat16
        self._steps = int(args.num_inference_steps)
        self._cond = text_cond
        self._neg_cond = neg_cond
        self._use_base_model = bool(getattr(args, "use_base_model", False))

        # Upstream parses size as HEIGHT*WIDTH.
        height, _, width = str(args.size).partition("*")
        self._height, self._width = int(height), int(width)

        # Latent grid by exact integer division (the rahul-mg3 refinement):
        # upstream's sqrt(max_area * aspect) round-trip can land one latent off
        # for sizes outside MAX_AREA_CONFIGS, which breaks the img_cond concat.
        vs, ps = cfg.vae_stride, cfg.patch_size
        self._lat_h = (self._height // vs[1] // ps[1]) * ps[1]
        self._lat_w = (self._width // vs[2] // ps[2]) * ps[2]
        self._target_h = self._lat_h * vs[1]
        self._target_w = self._lat_w * vs[2]

        from utils.cam_utils import get_intrinsics

        self._base_K = get_intrinsics(self._target_h, self._target_w)

        # Sequence length padded to the SP degree, per upstream.
        max_lat_f = (FIRST_CLIP_FRAME - 1) // vs[0] + 1
        max_seq = (max_lat_f + MEMORY_FRAMES) * self._lat_h * self._lat_w // (ps[1] * ps[2])
        sp = int(getattr(args, "ulysses_size", 1))
        if sp > 1:
            max_seq = -(-max_seq // sp) * sp
        self._max_seq_len = max_seq

        self._generator = torch.Generator(device=self.device).manual_seed(
            int(getattr(args, "seed", 42))
        )

        # img_cond: rank 0 encodes the anchor, everyone else receives it. The
        # zeros tensor is created on the LOCAL device, so the in-place NCCL
        # broadcast cannot re-tag devices the way broadcast_object_list does.
        image = self._image_tensor(anchor_image)
        if self.rank == 0:
            img_cond = (
                self.pipe.vae.encode([image[0]])[0]
                .unsqueeze(0)
                .to(device=self.device, dtype=self._dtype)
                .contiguous()
            )
        else:
            img_cond = torch.zeros(
                (1, LATENT_CHANNELS, 1, self._lat_h, self._lat_w),
                device=self.device, dtype=self._dtype,
            ).contiguous()
        if dist.is_initialized():
            dist.broadcast(img_cond, src=0)
        self._img_cond = img_cond

        # Session accumulators. Per-chunk lists with one cat per chunk keep the
        # accumulation O(n) instead of the O(n²) of a running torch.cat.
        self._keyboard_chunks: list[torch.Tensor] = []
        self._mouse_chunks: list[torch.Tensor] = []
        self._extrinsics_chunks: list[torch.Tensor] = []
        self._keyboard_all: Optional[torch.Tensor] = None
        self._mouse_all: Optional[torch.Tensor] = None
        self._extrinsics_all: Optional[torch.Tensor] = None
        self._last_pose = np.zeros(5)
        self._all_latents: list[torch.Tensor] = []

        # One scheduler per session; set_timesteps() per chunk resets the
        # stateful model_outputs buffer without reallocating the object.
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        self._scheduler = FlowUniPCMultistepScheduler()

        # 34-slot streaming cache, matching upstream's vae_cache = [None] * 34.
        self._vae_cache: list = [None] * 34

        if self.rank == 0:
            logger.info(
                "lockstep session: %dx%d, latent %dx%d, seq_len %d, steps %d",
                self._width, self._height, self._lat_w, self._lat_h,
                self._max_seq_len, self._steps,
            )

    def _image_tensor(self, pil_image):
        import torch
        import torch.nn.functional as torch_F

        arr = torch.from_numpy(np.array(pil_image)).unsqueeze(0)
        arr = arr.float().permute(0, 3, 1, 2) / 127.5 - 1.0
        arr = torch_F.interpolate(
            arr, size=(self._height, self._width), mode="bicubic", align_corners=False
        )
        return arr.transpose(0, 1).unsqueeze(0).to(self.device, self._dtype)

    # ------------------------------------------------------------------ chunk

    def run_chunk(self, keyboard: list, mouse: list) -> Optional[np.ndarray]:
        """Denoise one chunk under the given action; rank 0 returns the new
        frames as a uint8 array of shape [T, H, W, 3], other ranks None.

        Every rank must call this with the same action, in the same order.
        """
        import torch

        with torch.no_grad():
            return self._run_chunk(keyboard, mouse)

    def _run_chunk(self, keyboard: list, mouse: list) -> Optional[np.ndarray]:
        import torch
        import torch.distributed as dist

        from utils.cam_utils import (
            _interpolate_camera_poses_handedness,
            compute_relative_poses,
            select_memory_idx_fov,
        )
        from utils.utils import (
            build_plucker_from_c2ws,
            build_plucker_from_pose,
            compute_all_poses_from_actions,
            get_extrinsics,
        )

        chunk_idx = self.chunk_idx
        first_clip = chunk_idx == 0
        dtype = self._dtype
        lat_h, lat_w = self._lat_h, self._lat_w

        # ---- Conditions from the action, computed identically on every rank.
        action_frames = new_frames_in_chunk(chunk_idx)
        keyboard_curr = torch.tensor(keyboard, dtype=torch.float32).repeat(action_frames, 1)
        mouse_curr = torch.tensor(mouse, dtype=torch.float32).repeat(action_frames, 1)

        all_poses, self._last_pose = compute_all_poses_from_actions(
            keyboard_curr, mouse_curr, first_pose=self._last_pose, return_last_pose=True
        )
        positions = all_poses[:, :3].tolist()
        rotations = np.concatenate(
            [np.zeros((all_poses.shape[0], 1)), all_poses[:, 3:5]], axis=1
        ).tolist()
        extrinsics_curr = get_extrinsics(rotations, positions)

        self._keyboard_chunks.append(
            keyboard_curr.unsqueeze(0).to(device=self.device, dtype=dtype)
        )
        self._mouse_chunks.append(
            mouse_curr.unsqueeze(0).to(device=self.device, dtype=dtype)
        )
        self._extrinsics_chunks.append(extrinsics_curr)
        self._keyboard_all = torch.cat(self._keyboard_chunks, dim=1)
        self._mouse_all = torch.cat(self._mouse_chunks, dim=1)
        self._extrinsics_all = torch.cat(self._extrinsics_chunks, dim=0)

        start_frame, end_frame = chunk_frame_window(chunk_idx)
        latent_start_idx = get_latent_idx(start_frame)
        latent_end_idx = get_latent_idx(end_frame)

        # ---- Plücker embeddings for the predicted frames.
        c2ws_chunk = self._extrinsics_all[start_frame:end_frame].to(device=self.device)
        n_src = FIRST_CLIP_FRAME if first_clip else CLIP_FRAME
        tgt_len = (FIRST_CLIP_FRAME - 1) // 4 + 1 if first_clip else CLIP_FRAME // 4
        src_indices = np.linspace(start_frame, end_frame - 1, n_src)
        tgt_indices = np.linspace(
            0 if first_clip else start_frame + 3, end_frame - 1, tgt_len
        )
        plucker = build_plucker_from_c2ws(
            c2ws_chunk, src_indices, tgt_indices, framewise=True,
            base_K=self._base_K, target_h=self._target_h, target_w=self._target_w,
            lat_h=lat_h, lat_w=lat_w,
        )
        plucker_no_mem = plucker

        # ---- Episodic memory (upstream's shape, including its collective).
        x_memory = None
        memory_mouse = None
        memory_keyboard = None
        timestep_memory = None
        latent_idx = None
        if not first_clip:
            selected_index_base = [end_frame - o for o in range(1, 34, 8)]
            if self.rank == 0:
                selected_index = select_memory_idx_fov(
                    self._extrinsics_all, start_frame, selected_index_base, use_gpu=True
                )
                selected_index[-1] = 4
            else:
                selected_index = [0] * MEMORY_FRAMES
            if dist.is_initialized():
                # A list of Python ints — no CUDA tensors, so no device re-tagging.
                dist.broadcast_object_list(selected_index, src=0)

            memory_pluckers = []
            latent_idx = []
            for mem_idx, ref_idx in zip(selected_index, selected_index_base):
                latent_idx.append(get_latent_idx(mem_idx))
                mem_aligned = align_frame_to_block(mem_idx)
                mem_block = self._extrinsics_all[mem_aligned:mem_aligned + 4]
                mem_pose = _interpolate_camera_poses_handedness(
                    src_indices=np.linspace(mem_aligned, mem_aligned + 3, mem_block.shape[0]),
                    src_rot_mat=mem_block[:, :3, :3].cpu().numpy(),
                    src_trans_vec=mem_block[:, :3, 3].cpu().numpy(),
                    tgt_indices=np.array([mem_aligned + 3], dtype=np.float32),
                )
                ref_pose = self._extrinsics_all[ref_idx:ref_idx + 1]
                rel_pair = torch.cat([ref_pose, mem_pose.to(ref_pose.device)], dim=0)
                rel_pose = compute_relative_poses(rel_pair, framewise=False)[1:2]
                memory_pluckers.append(
                    build_plucker_from_pose(
                        rel_pose.to(device=self.device),
                        base_K=self._base_K, target_h=self._target_h,
                        target_w=self._target_w, lat_h=lat_h, lat_w=lat_w,
                    )
                )
            plucker = torch.cat(memory_pluckers + [plucker], dim=2)

            stacked = torch.cat(self._all_latents, dim=2)
            x_memory = stacked[:, :, latent_idx]
            memory_mouse = torch.ones((1, MEMORY_FRAMES, 2), device=self.device, dtype=dtype)
            memory_keyboard = -torch.ones((1, MEMORY_FRAMES, 6), device=self.device, dtype=dtype)
            timestep_memory = x_memory.new_zeros(
                (1, x_memory.shape[2] * x_memory.shape[3] * x_memory.shape[4] // 4)
            )

        keyboard_cond = self._keyboard_all[:, start_frame:end_frame]
        mouse_cond = self._mouse_all[:, start_frame:end_frame]
        plucker = plucker.to(device=self.device, dtype=dtype)
        plucker_no_mem = plucker_no_mem.to(device=self.device, dtype=dtype)

        # ---- Noise, anchored to the running image condition.
        latents = torch.randn(
            (1, LATENT_CHANNELS, latent_end_idx - latent_start_idx, lat_h, lat_w),
            generator=self._generator, device=self.device, dtype=dtype,
        )
        latents = torch.cat(
            [self._img_cond, latents[:, :, self._img_cond.shape[2]:]], dim=2
        )

        # fa_version is deliberately NOT passed: the pipeline already fixed its
        # attention backend at init, and forwarding it per call is what trips
        # upstream issue #80 in the Ulysses dispatch.
        conditions = {
            "mouse_cond": mouse_cond,
            "keyboard_cond": keyboard_cond,
            "context": self._cond,
            "plucker_emb": plucker,
            "x_memory": x_memory,
            "timestep_memory": timestep_memory,
            "keyboard_cond_memory": memory_keyboard,
            "mouse_cond_memory": memory_mouse,
            "memory_latent_idx": latent_idx,
            "predict_latent_idx": (latent_start_idx, latent_end_idx),
        }

        self._scheduler.set_timesteps(
            self._steps, device=self.device, shift=float(self.args.sample_shift)
        )
        for t in self._scheduler.timesteps:
            timestep = latents.new_full(
                (latents.shape[2], latents.shape[3] * latents.shape[4] // 4), t
            )
            timestep[: self._img_cond.shape[2]].zero_()
            timestep = timestep.flatten().unsqueeze(0)
            kwargs = {"x": latents, "t": timestep, "seq_len": self._max_seq_len, **conditions}

            if self._use_base_model and self._neg_cond is not None:
                null = dict(kwargs)
                null.update(
                    mouse_cond=torch.ones_like(mouse_cond),
                    keyboard_cond=-torch.ones_like(keyboard_cond),
                    context=self._neg_cond,
                    plucker_emb=plucker_no_mem,
                    x_memory=None, timestep_memory=None,
                    keyboard_cond_memory=None, mouse_cond_memory=None,
                    memory_latent_idx=None,
                )
                noise_full = self.pipe.model(**kwargs)
                noise_null = self.pipe.model(**null)
                guide = float(self.args.sample_guide_scale)
                noise_pred = noise_null + guide * (noise_full - noise_null)
            else:
                noise_pred = self.pipe.model(**kwargs)

            latents = self._scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents = torch.cat(
                [self._img_cond, latents[:, :, self._img_cond.shape[2]:]], dim=2
            )

        self._img_cond = latents[:, :, -4:]
        denoised = latents if first_clip else latents[:, :, -10:]

        # Memory selection indexes latents from session start, so the list must
        # keep growing (truncating shifts every index and eventually walks off
        # the end). The caller bounds session length instead.
        self._all_latents.append(denoised)

        self.chunk_idx += 1

        # ---- Decode on rank 0 only; no collectives from here on.
        if self.rank != 0:
            return None
        do_compile = bool(getattr(self.args, "compile_vae", False)) and chunk_idx >= 1
        segment = int(os.environ.get("WAN_VAE_SEGMENT_SIZE", "4"))
        video, self._vae_cache = self.pipe.vae.stream_decode(
            denoised.to(dtype=self.pipe.vae.dtype),
            self._vae_cache,
            first_chunk=first_clip,
            segment_size=segment,
            profiler={},
            compile_decoder=do_compile,
        )
        frames = video[0].permute(1, 2, 3, 0)  # C T H W -> T H W C
        frames = ((frames.float() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        return np.ascontiguousarray(frames.cpu().numpy())
