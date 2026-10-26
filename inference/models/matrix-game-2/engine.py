# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matrix-Game-2.0 inference engine (universal mode), one chunk per call.

Ported from the engine that ran Matrix Game 2 at the AWS Summit Paris and Dubai
booths (world-models-infra, `src/engines/matrixgame2.py`), minus the parts the
shared server already owns: there is no background thread or frame queue here —
the runner drives `step()` from its device thread and paces the frames itself —
and session ownership is the framework's session gate, not this class's job.

The model is a causal Wan 2.1 DiT (30 blocks, dim 1536) conditioned on a seed
image plus per-frame keyboard (W, S, A, D) and mouse (pitch, yaw). Each chunk
denoises 3 latent frames in 3 distilled steps and the streaming VAE decodes them
to 12 RGB frames at 640x352 (9 for the first chunk: the causal VAE emits one
frame for the first latent).

Upstream code is imported from a pinned clone of SkyworkAI/Matrix-Game (MIT),
which the Dockerfile places at /opt/ml/code/Matrix-Game.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

DTYPE = torch.bfloat16
VAE_DTYPE = torch.float16

TARGET_H, TARGET_W = 352, 640
LATENT_H, LATENT_W = 44, 80
FRAME_SEQ_LENGTH = 880  # (44/2) * (80/2) tokens per latent frame
NUM_TRANSFORMER_BLOCKS = 30
VAE_TIME_RATIO = 4
FRAMES_PER_CHUNK = 3  # latent frames denoised per step()

# The one camera magnitude the model was trained on. It sees ternary yaw
# (+CAM_VALUE, 0, -CAM_VALUE); there is no proportional control.
CAM_VALUE = 0.1

REPO_CONFIG = "configs/distilled_model/universal"
INFERENCE_YAML = "configs/inference_yaml/inference_universal.yaml"
CKPT_PATH = "base_distilled_model/base_distill.safetensors"
# A path running away from the camera: walking forward reads as travel. The
# Minecraft scene (0016) puts a tree dead ahead, so W fills the frame with leaves.
DEFAULT_SEED_IMAGE = "demo_images/universal/0009.png"

# Latent frames of conditioning per session. When a session uses them all up
# the engine re-seeds from the last frame it generated and carries on, so this
# sets how often that (sub-second) pause happens: 360 latents = 120 chunks ≈ 80 s.
DEFAULT_MAX_LATENTS = 360

# Upstream encodes the seed followed by 4*(N-1) all-zero frames through the
# causal VAE, which costs ~13.5 s at N=360. The latents for that zero padding
# go constant a few latents past the seed (measured on an H100: exact over the
# prefix, then flat to within one bf16 ulp), so encoding a short prefix and
# repeating its last latent gives the same conditioning in ~1.2 s.
ENC_PREFIX_LATENTS = 32

# Seed conditioning keyed by image content: revisiting a world skips the
# VAE + CLIP encode entirely (~0.7 s to first frame instead of ~1.2 s).
COND_CACHE_SIZE = 16


def resizecrop(image: Image.Image, target_h: int, target_w: int) -> Image.Image:
    """Centre-crop to the target aspect ratio (the resize happens afterwards)."""
    w, h = image.size
    if h / w > target_h / target_w:
        new_w, new_h = w, int(w * target_h / target_w)
    else:
        new_h, new_w = h, int(h * target_w / target_h)
    left, top = (w - new_w) / 2, (h - new_h) / 2
    return image.crop((left, top, left + new_w, top + new_h))


def action_tensors(keyboard: Sequence[int], yaw: float, device, dtype=DTYPE):
    """(keyboard[4], mouse[pitch, yaw]) tensors for one chunk.

    Pitch is always zero: the model handles vertical look badly, so it is
    disabled here rather than shown misbehaving (the Paris booth's finding).
    """
    w, s, a, d = (list(keyboard) + [0, 0, 0, 0])[:4]
    kbd = torch.tensor([w, s, a, d], dtype=dtype, device=device)
    yaw_val = CAM_VALUE if yaw > 0 else (-CAM_VALUE if yaw < 0 else 0.0)
    mouse = torch.tensor([0.0, yaw_val], dtype=dtype, device=device)
    return kbd, mouse


def chunk_pixel_span(chunk_index: int, max_pixel: int) -> tuple[int, int]:
    """Pixel-frame range [start, end) that chunk `chunk_index`'s action covers."""
    start_latent = chunk_index * FRAMES_PER_CHUNK
    if start_latent == 0:
        start, n = 0, 1 + VAE_TIME_RATIO * (FRAMES_PER_CHUNK - 1)
    else:
        start, n = 1 + VAE_TIME_RATIO * (start_latent - 1), VAE_TIME_RATIO * FRAMES_PER_CHUNK
    return start, min(start + n, max_pixel)


class MatrixGame2Engine:
    """Load once, then `start(image)` per world and `step(keys, yaw)` per chunk.

    Not thread-safe: every call must come from the same thread (the runner's
    device thread), because torch.compile'd graphs belong to the thread that
    compiled them.
    """

    def __init__(self, model_path: str, mg_dir: str, device: torch.device,
                 max_latents: int = DEFAULT_MAX_LATENTS):
        self.device = device
        self.mg_dir = os.path.abspath(mg_dir)
        self.max_latents = max_latents
        self.max_chunks = max_latents // FRAMES_PER_CHUNK

        if self.mg_dir not in sys.path:
            sys.path.insert(0, self.mg_dir)

        # The VAE decoder is compiled; it sees ~8 distinct shapes, more than
        # dynamo's default recompile budget, and uses .item() inside the graph.
        os.environ.pop("TORCHDYNAMO_DISABLE", None)
        torch._dynamo.config.recompile_limit = 32
        torch._dynamo.config.capture_scalar_outputs = True

        prev_cwd = os.getcwd()
        try:
            os.chdir(self.mg_dir)  # upstream resolves some configs relative to cwd
            self._load(os.path.abspath(model_path))
        finally:
            os.chdir(prev_cwd)

        self._cond_cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._cond: Optional[dict] = None
        self._chunk_idx = 0
        self._last_frame: Optional[np.ndarray] = None
        self.reseeds = 0

    # ------------------------------------------------------------------ load

    def _load(self, model_path: str) -> None:
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from torchvision.transforms import v2

        from demo_utils.vae_block3 import VAEDecoderWrapper
        from utils.wan_wrapper import WanDiffusionWrapper
        from wan.vae.wanx_vae import get_wanx_vae_wrapper

        config = OmegaConf.load(os.path.join(self.mg_dir, INFERENCE_YAML))
        config.model_kwargs.model_config = os.path.join(self.mg_dir, REPO_CONFIG)

        generator = WanDiffusionWrapper(**config.model_kwargs, is_causal=True)
        ckpt = os.path.join(model_path, CKPT_PATH)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Matrix-Game-2 checkpoint not found: {ckpt}")
        generator.load_state_dict(load_file(ckpt))
        generator.to(device=self.device, dtype=DTYPE).eval().requires_grad_(False)
        self._generator = generator

        self._scheduler = generator.get_scheduler()
        steps = torch.tensor(config.denoising_step_list, dtype=torch.long)
        if config.warp_denoising_step:
            timesteps = torch.cat((self._scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            steps = timesteps[1000 - steps]
        self._denoising_steps = steps
        self._context_noise = getattr(config, "context_noise", 0)

        self._local_attn_size = generator.model.local_attn_size
        if self._local_attn_size == -1:
            raise RuntimeError("Matrix-Game-2 config must set a finite local_attn_size")
        generator.model.num_frame_per_block = FRAMES_PER_CHUNK

        vae_decoder = VAEDecoderWrapper()
        vae_state = torch.load(os.path.join(model_path, "Wan2.1_VAE.pth"), map_location="cpu")
        vae_decoder.load_state_dict({k: v for k, v in vae_state.items() if "decoder." in k or "conv2" in k})
        vae_decoder.to(self.device, VAE_DTYPE).eval().requires_grad_(False)
        self._vae_decoder = torch.compile(vae_decoder, mode="max-autotune-no-cudagraphs")

        vae = get_wanx_vae_wrapper(model_path, VAE_DTYPE)
        vae.requires_grad_(False)
        vae.eval()
        self._vae = vae.to(self.device, DTYPE)

        self._frame_process = v2.Compose([
            v2.Resize(size=(TARGET_H, TARGET_W), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        logger.info("Matrix-Game-2 loaded: steps=%s local_attn_size=%d",
                    self._denoising_steps.tolist(), self._local_attn_size)

    def default_seed(self) -> Image.Image:
        path = os.path.join(self.mg_dir, DEFAULT_SEED_IMAGE)
        if os.path.isfile(path):
            return Image.open(path).convert("RGB")
        return Image.new("RGB", (TARGET_W, TARGET_H), (90, 110, 140))

    # --------------------------------------------------------------- session

    @torch.no_grad()
    def start(self, image: Image.Image, cache: bool = True) -> None:
        """Begin a fresh world from `image`.

        cache=False for re-seeds from a generated frame: each is unique, and
        caching them would evict the preset worlds (~50 MB an entry).
        """
        image = resizecrop(image.convert("RGB"), TARGET_H, TARGET_W).resize((TARGET_W, TARGET_H))
        pixels = self._frame_process(image)[None, :, None, :, :].to(dtype=DTYPE, device=self.device)
        n_total = self.max_latents

        key = hashlib.sha1(np.asarray(image).tobytes()).hexdigest()
        cached = self._cond_cache.get(key)
        if cached is not None:
            self._cond_cache.move_to_end(key)
            cond_concat, visual_context = cached
        else:
            t0 = time.perf_counter()
            n_enc = min(ENC_PREFIX_LATENTS, n_total)
            padding = torch.zeros_like(pixels).repeat(1, 1, VAE_TIME_RATIO * (n_enc - 1), 1, 1)
            # The encoder's causal caches survive between calls; a stale one
            # from the previous world raises a shape mismatch.
            if hasattr(self._vae, "vae") and hasattr(self._vae.vae, "model"):
                self._vae.vae.model.clear_cache()
            img_cond = self._vae.encode(
                torch.cat([pixels, padding], dim=2), device=self.device,
                tiled=True, tile_size=[LATENT_H, LATENT_W], tile_stride=[23, 38],
            ).to(self.device)
            if n_total > n_enc:
                tail = img_cond[:, :, -1:].expand(-1, -1, n_total - n_enc, -1, -1)
                img_cond = torch.cat([img_cond, tail], dim=2)
            mask = torch.ones_like(img_cond)
            mask[:, :, 1:] = 0
            cond_concat = torch.cat([mask[:, :4], img_cond], dim=1).to(device=self.device, dtype=DTYPE)
            visual_context = self._vae.clip.encode_video(pixels).to(device=self.device, dtype=DTYPE)
            if cache:
                self._cond_cache[key] = (cond_concat, visual_context)
                while len(self._cond_cache) > COND_CACHE_SIZE:
                    self._cond_cache.popitem(last=False)
            logger.info("seed conditioning encoded in %.0f ms", (time.perf_counter() - t0) * 1000)

        n_pixel = (n_total - 1) * VAE_TIME_RATIO + 1
        self._cond = {
            "cond_concat": cond_concat,
            "visual_context": visual_context,
            "keyboard_cond": torch.zeros(1, n_pixel, 4, device=self.device, dtype=DTYPE),
            "mouse_cond": torch.zeros(1, n_pixel, 2, device=self.device, dtype=DTYPE),
        }
        self._chunk_idx = 0
        self._vae_cache = [None] * 32
        self._init_kv_caches()

    def end(self) -> None:
        """Drop per-session GPU state (the conditioning cache is kept)."""
        self._cond = None
        self._kv_cache = self._kv_keyboard = self._kv_mouse = self._crossattn = None
        self._vae_cache = [None] * 32
        self._chunk_idx = 0
        self._last_frame = None

    @property
    def started(self) -> bool:
        return self._cond is not None

    def _init_kv_caches(self) -> None:
        dev, dt = self.device, DTYPE

        def cache(rows: int, length: int, heads: int, dim: int) -> list:
            return [{"k": torch.zeros([rows, length, heads, dim], dtype=dt, device=dev),
                     "v": torch.zeros([rows, length, heads, dim], dtype=dt, device=dev),
                     "global_end_index": torch.tensor([0], dtype=torch.long, device=dev),
                     "local_end_index": torch.tensor([0], dtype=torch.long, device=dev)}
                    for _ in range(NUM_TRANSFORMER_BLOCKS)]

        self._kv_cache = cache(1, self._local_attn_size * FRAME_SEQ_LENGTH, 12, 128)
        self._kv_keyboard = cache(1, self._local_attn_size, 16, 64)
        self._kv_mouse = cache(FRAME_SEQ_LENGTH, self._local_attn_size, 16, 64)
        self._crossattn = [{"k": torch.zeros([1, 257, 12, 128], dtype=dt, device=dev),
                            "v": torch.zeros([1, 257, 12, 128], dtype=dt, device=dev),
                            "is_init": False}
                           for _ in range(NUM_TRANSFORMER_BLOCKS)]

    # ------------------------------------------------------------------ step

    @torch.no_grad()
    def step(self, keyboard: Sequence[int], yaw: float) -> np.ndarray:
        """Generate the next chunk under the given action; returns uint8 [T, H, W, 3]."""
        if self._cond is None:
            self.start(self.default_seed())
        if self._chunk_idx >= self.max_chunks:
            # Out of conditioning slots: keep the world going from the frame on
            # screen rather than freezing (the booth's ~80 s freeze).
            self.reseeds += 1
            logger.info("conditioning exhausted after %d chunks; re-seeding from last frame",
                        self._chunk_idx)
            if self._last_frame is not None:
                self.start(Image.fromarray(self._last_frame), cache=False)
            else:
                self.start(self.default_seed())

        self._apply_action(keyboard, yaw)
        latent = self._denoise()
        video, self._vae_cache = self._vae_decoder(latent.transpose(1, 2).half(), *self._vae_cache)
        frames = video[0].clamp(-1, 1).add(1).mul(127.5).to(torch.uint8)
        frames = frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        self._chunk_idx += 1
        if len(frames):
            self._last_frame = frames[-1]
        return frames

    def _apply_action(self, keyboard: Sequence[int], yaw: float) -> None:
        cond = self._cond
        start, end = chunk_pixel_span(self._chunk_idx, cond["keyboard_cond"].shape[1])
        if end <= start:
            return
        kbd, mouse = action_tensors(keyboard, yaw, self.device)
        cond["keyboard_cond"][:, start:end] = kbd[None, None, :].expand(1, end - start, -1)
        cond["mouse_cond"][:, start:end] = mouse[None, None, :].expand(1, end - start, -1)

    def _chunk_cond(self, start_frame: int) -> dict:
        cond = self._cond
        end_pixel = 1 + VAE_TIME_RATIO * (start_frame + FRAMES_PER_CHUNK - 1)
        return {
            "cond_concat": cond["cond_concat"][:, :, start_frame:start_frame + FRAMES_PER_CHUNK],
            "visual_context": cond["visual_context"],
            "keyboard_cond": cond["keyboard_cond"][:, :end_pixel],
            "mouse_cond": cond["mouse_cond"][:, :end_pixel],
        }

    def _denoise(self) -> torch.Tensor:
        from einops import rearrange

        nf = FRAMES_PER_CHUNK
        start_frame = self._chunk_idx * nf
        chunk_cond = self._chunk_cond(start_frame)
        caches = dict(kv_cache=self._kv_cache, kv_cache_mouse=self._kv_mouse,
                      kv_cache_keyboard=self._kv_keyboard, crossattn_cache=self._crossattn,
                      current_start=start_frame * FRAME_SEQ_LENGTH)

        noisy = torch.randn([1, 16, nf, LATENT_H, LATENT_W], device=self.device, dtype=DTYPE)
        denoised = None
        for i, t in enumerate(self._denoising_steps):
            timestep = torch.ones([1, nf], device=self.device, dtype=torch.int64) * t
            _, denoised = self._generator(noisy_image_or_video=noisy, conditional_dict=chunk_cond,
                                          timestep=timestep, **caches)
            if i < len(self._denoising_steps) - 1:
                flat = rearrange(denoised, "b c f h w -> (b f) c h w")
                next_t = self._denoising_steps[i + 1]
                noisy = self._scheduler.add_noise(
                    flat, torch.randn_like(flat),
                    next_t * torch.ones([nf], device=self.device, dtype=torch.long))
                noisy = rearrange(noisy, "(b f) c h w -> b c f h w", b=1)

        # One more pass at the context-noise level writes this chunk into the
        # KV cache for the next one.
        ctx_t = torch.ones([1, nf], device=self.device, dtype=torch.int64) * self._context_noise
        self._generator(noisy_image_or_video=denoised, conditional_dict=chunk_cond,
                        timestep=ctx_t, **caches)
        return denoised
