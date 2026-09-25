# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waypoint-1.5-1B cartridge — Overworld's real-time interactive world model.

Wraps the upstream `world_engine` library (GPL-3.0, installed by this
cartridge's Dockerfile) behind the shared Runner contract. The engine is an
autoregressive diffusion transformer: seed it with one image, then each
`gen_frame(ctrl)` call denoises the next latent frame conditioned on the
pressed buttons and mouse deltas, and the TAEHV autoencoder decodes it to
4 RGB frames at 720p (temporal_compression=4, inference_fps=60 → 15 engine
steps per second of video).

Threading: every engine call — construction, warmup, reset, gen_frame — runs
on one dedicated device thread. world_engine leans on torch.compile with CUDA
graphs, and compiled graphs must execute on the thread that compiled them
(the same discipline Overworld's own Biome server applies). The shared server
calls reset() on the asyncio thread and stream() on an executor thread, so
both trampoline onto the device thread instead of touching the engine.

Runtime knobs (container env):
  WAYPOINT_QUANT        none | intw8a8 | fp8w8a8 | nvfp4   (default fp8w8a8)
  WAYPOINT_MOUSE_SCALE  mouse counts per step at full stick (default 15)
  WAYPOINT_SEED         path to a seed image; default is the first frame of
                        the demo video shipped inside the weights repo
  WAYPOINT_WARMUP       1 to compile graphs at startup (default 1)
  WAYPOINT_JPEG_QUALITY JPEG quality for the WebSocket frames (default 60)
  WAYPOINT_JPEG_SUBSAMPLING  chroma subsampling: 420 | 422 | 444 (default 420)

Frame rate over the wire is bandwidth-bound, not GPU-bound: the engine
generates ~82 fps but a 720p frame is ~73 KiB of JPEG at quality 60 and 4:2:0,
so 60 fps needs ~36 Mbit/s sustained to the player. Measured on real frames
from this model, per delivered frame: quality 80 costs 127 KiB, 4:4:4 chroma
costs 86 KiB, and H.264 crf23 would cost 15.7 KiB — which is why the remaining
big win here is a video codec, not a faster GPU. The per-frame byte costs were
measured on real output; see docs/EVIDENCE.md for where those captures live.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread

import numpy as np
import torch

from lib.runner import ActionBuffer, Runner

import actions as waypoint_actions
import frames as waypoint_frames

logger = logging.getLogger(__name__)

SEED_SIZE = (720, 1280)  # (H, W) — Waypoint-1.5's native aspect/resolution


class WaypointRunner(Runner):
    def setup(self, ckpt_dir: str, device: torch.device, rank: int, world_size: int) -> None:
        self.device = device
        self.rank = rank
        self.ckpt_dir = ckpt_dir
        # Both encoding knobs are resolved rather than parsed: a typo in the
        # container environment should cost one warning at boot, not a dead
        # endpoint (setup) or an exception per frame (the encode loop).
        for var, resolve, attr in (
            ("WAYPOINT_JPEG_QUALITY", waypoint_frames.resolve_quality, "jpeg_quality"),
            ("WAYPOINT_JPEG_SUBSAMPLING", waypoint_frames.resolve_subsampling, "jpeg_subsampling"),
        ):
            configured = os.environ.get(var)
            resolved = resolve(configured)
            setattr(self, attr, resolved)
            if configured and configured.strip() != str(resolved):
                logger.warning("ignoring %s=%r; using %s", var, configured, resolved)

        if world_size > 1 and rank != 0:
            # Single-GPU model. The image pins NPROC_PER_NODE=1 so this should
            # not happen; if it does, followers idle in the command loop.
            logger.warning("waypoint-1-5 is single-GPU; rank %d loads nothing", rank)
            self.engine = None
            return

        # One thread owns the GPU for the life of the process (see module doc).
        self._dev = ThreadPoolExecutor(max_workers=1, thread_name_prefix="waypoint-device")

        quant = os.environ.get("WAYPOINT_QUANT", "fp8w8a8").strip().lower()
        quant = None if quant in ("", "none") else quant
        if quant == "fp8w8a8" and torch.cuda.is_available():
            # torch._scaled_mm needs Ada (8.9) or newer; on Ampere parts like
            # the A10G fall back to int8 rather than dying in warmup.
            cc = torch.cuda.get_device_capability(self.device)
            if cc < (8, 9):
                logger.warning("compute capability %s has no fp8 — using intw8a8", cc)
                quant = "intw8a8"

        # The TAEHV autoencoder is staged next to the model weights; point the
        # engine at the local copy so nothing is fetched from the Hub at boot.
        overrides = {}
        ae_dir = Path(ckpt_dir) / "taehv1_5"
        if ae_dir.is_dir():
            overrides["ae_uri"] = str(ae_dir)

        def _load():
            from world_engine import CtrlInput, WorldEngine

            t0 = time.perf_counter()
            engine = WorldEngine(
                ckpt_dir,
                device=self.device,
                quant=quant,
                dtype=torch.bfloat16,
                model_config_overrides=overrides or None,
            )
            logger.info(
                "WorldEngine loaded in %.1fs (quant=%s, ae=%s)",
                time.perf_counter() - t0, quant, overrides.get("ae_uri", "hub"),
            )
            return engine, CtrlInput

        self.engine, self._ctrl_cls = self._dev.submit(_load).result()
        self.temporal_compression = int(self.engine.model_cfg.temporal_compression)

        self._seed = self._dev.submit(self._load_default_seed).result()

        if os.environ.get("WAYPOINT_WARMUP", "1") == "1":
            def _warmup():
                t0 = time.perf_counter()
                self.engine.reset()
                self.engine.append_frame(self._seed)
                self.engine.gen_frame(ctrl=self._ctrl_cls())
                logger.info("Warmup (torch.compile) done in %.1fs", time.perf_counter() - t0)

            self._dev.submit(_warmup).result()

    # ------------------------------------------------------------------ seed

    def _load_default_seed(self) -> torch.Tensor:
        """Resolve the boot seed image: env override, then the first frame of
        the demo video that ships in the weights repo, then a plain gradient
        (never fail setup over a missing picture)."""
        path = os.environ.get("WAYPOINT_SEED", "")
        if path and Path(path).is_file():
            from PIL import Image

            return self._seed_from_array(np.array(Image.open(path).convert("RGB")))

        demo = Path(self.ckpt_dir) / "assets" / "wp_1.5.mp4"
        if demo.is_file():
            try:
                import imageio

                frame = imageio.get_reader(str(demo)).get_data(0)
                logger.info("Seed: first frame of %s", demo)
                return self._seed_from_array(np.asarray(frame))
            except Exception:
                logger.exception("Could not read demo video; using gradient seed")

        h, w = SEED_SIZE
        ramp = np.linspace(60, 180, h, dtype=np.uint8)[:, None, None]
        return self._seed_from_array(np.broadcast_to(ramp, (h, w, 3)).copy())

    def _seed_from_array(self, rgb: np.ndarray) -> torch.Tensor:
        """HWC uint8 → the [T,H,W,3] uint8 device tensor append_frame expects."""
        import torch.nn.functional as F

        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().unsqueeze(0)
        t = F.interpolate(t, size=SEED_SIZE, mode="bilinear", align_corners=False)[0]
        t = t.clamp(0, 255).to(dtype=torch.uint8, device=self.device).permute(1, 2, 0).contiguous()
        return t.unsqueeze(0).expand(self.temporal_compression, -1, -1, -1).contiguous()

    def _seed_from_b64(self, b64: str) -> torch.Tensor | None:
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            return self._seed_from_array(np.array(img))
        except Exception:
            logger.exception("Bad client seed image; keeping current world")
            return None

    # ------------------------------------------------------------- lifecycle

    def generate(self, **params) -> str:
        raise NotImplementedError("waypoint-1-5 is a real-time cartridge; connect to /ws")

    @property
    def supports_streaming(self) -> bool:
        return True

    def reset(self) -> None:
        """Fresh world per session: reset caches and re-append the seed."""
        if self.engine is None:
            return

        def _reset():
            self.engine.reset()
            self.engine.append_frame(self._seed)

        self._dev.submit(_reset).result()

    # --------------------------------------------------------------- stream

    def stream(self, actions: ActionBuffer):
        if self.engine is None:
            return

        state = {"ctrl": waypoint_actions.neutral(), "seed": None}
        closed = Event()

        def pump():
            # ActionBuffer.get() re-returns the latest payload every ≤0.1s
            # whether or not the client sent anything new, and None once the
            # session closes. Re-applying a held control is harmless, but
            # re-applying a `start` is not: it would reset the engine and throw
            # away the frame context ~10x a second for as long as the client
            # stayed quiet. So act only when the payload actually changes.
            last = None
            while True:
                payload = actions.get()
                if payload is None:
                    if not actions.active:
                        break
                    continue
                if payload == last:
                    continue
                last = payload
                ctrl = waypoint_actions.decode(payload)
                if ctrl.is_seed:
                    if ctrl.seed_b64:
                        state["seed"] = ctrl.seed_b64
                else:
                    state["ctrl"] = ctrl
            closed.set()

        Thread(target=pump, name="waypoint-actions", daemon=True).start()

        def step():
            b64 = state["seed"]
            if b64 is not None:
                state["seed"] = None
                seed = self._seed_from_b64(b64)
                if seed is not None:
                    self.engine.reset()
                    self.engine.append_frame(seed)
            ctrl = state["ctrl"]
            return self.engine.gen_frame(
                ctrl=self._ctrl_cls(button=set(ctrl.buttons), mouse=tuple(ctrl.mouse))
            )

        # Pipeline: while the GPU denoises step N+1, the CPU JPEG-encodes and
        # ships step N. At 720p this overlap is what keeps 60 fps reachable.
        pending = self._dev.submit(step)
        n_steps = 0
        t0 = time.perf_counter()
        try:
            while not closed.is_set():
                frames = pending.result()
                pending = self._dev.submit(step)
                for jpeg in waypoint_frames.encode_batch(
                        frames, self.jpeg_quality, self.jpeg_subsampling):
                    yield jpeg
                n_steps += 1
                if n_steps % 150 == 0:
                    fps = n_steps * self.temporal_compression / (time.perf_counter() - t0)
                    logger.info("session throughput: %.1f fps over %d steps", fps, n_steps)
        finally:
            try:
                pending.result(timeout=30)
            except Exception:
                pass
            if n_steps:
                fps = n_steps * self.temporal_compression / (time.perf_counter() - t0)
                logger.info("session ended: %d steps, %.1f fps avg", n_steps, fps)


def create_runner() -> Runner:
    """Required factory — the shared server imports this by name."""
    return WaypointRunner()
