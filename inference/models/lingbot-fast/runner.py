# Copyright 2025 LingBot-World Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
# Modifications Copyright Amazon.com, Inc. or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Persistent wrapper around upstream `wan.WanI2VFast` used by every rank.

The pipeline is loaded once per process at startup. On every generate call,
all ranks call `self.pipeline.generate(...)` with identical kwargs — Ulysses
sequence parallelism and FSDP inside the pipeline handle per-rank work.
"""

from __future__ import annotations

import logging
import os
import tempfile
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from PIL import Image

from lib.runner import Runner

# The `wan` package lives next to this file (vendored from upstream).
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.distributed.util import init_distributed_group  # noqa: E402
from wan.image2video_fast import WanI2VFast  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402

logger = logging.getLogger(__name__)


# --- Defaults — match upstream run_fast.sh --------------------------------
DEFAULT_TASK = "i2v-A14B"
DEFAULT_SIZE = "480*832"
DEFAULT_FRAME_NUM = 81
DEFAULT_SEED = 42
DEFAULT_CHUNK_SIZE = 3


class LingBotFastPipeline(Runner):
    """WanI2VFast pipeline wrapped as a generic Runner."""

    def setup(self, ckpt_dir: str, device: torch.device, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device_id = self.local_rank
        self.task = DEFAULT_TASK
        self.ckpt_dir = ckpt_dir

        use_sp = world_size > 1
        t5_fsdp = use_sp
        dit_fsdp = use_sp

        if world_size > 1:
            torch.cuda.set_device(self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    rank=rank,
                    world_size=world_size,
                )
            if use_sp:
                init_distributed_group()

        cfg = WAN_CONFIGS[self.task]
        if use_sp and world_size > 1:
            assert cfg.num_heads % world_size == 0, (
                f"num_heads={cfg.num_heads} must be divisible by world_size={world_size}"
            )

        if rank == 0:
            logger.info("Building WanI2VFast pipeline from %s (world_size=%d)",
                        ckpt_dir, world_size)

        # WanI2VFast sniffs 'cam' in checkpoint_dir to enable camera control.
        # Symlink if needed so the path contains 'cam'.
        if "cam" not in ckpt_dir:
            cam_path = ckpt_dir.rstrip("/") + "-cam"
            os.symlink(ckpt_dir, cam_path) if not os.path.exists(cam_path) else None
            ckpt_dir = cam_path

        self.pipeline = WanI2VFast(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=self.device_id,
            rank=rank,
            t5_fsdp=t5_fsdp,
            dit_fsdp=dit_fsdp,
            use_sp=use_sp,
            t5_cpu=False,
            convert_model_dtype=False,
        )
        self.cfg = cfg
        if rank == 0:
            logger.info("Pipeline ready on rank 0.")

    # -------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        image_path: str,
        *,
        action_path: Optional[str] = None,
        size: str = DEFAULT_SIZE,
        frame_num: int = DEFAULT_FRAME_NUM,
        seed: int = DEFAULT_SEED,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        sample_shift: Optional[float] = None,
        max_attention_size: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Run one synchronous generate across all ranks.

        Returns the output MP4 path on rank 0, empty string on other ranks.
        Must be called on every rank with identical kwargs.
        """
        if size not in MAX_AREA_CONFIGS:
            raise ValueError(
                f"Unsupported size '{size}'. Supported: {list(MAX_AREA_CONFIGS)}"
            )

        img = Image.open(image_path).convert("RGB")

        if self.rank == 0:
            logger.info(
                "generate: prompt=%r size=%s frame_num=%d seed=%d action_path=%s",
                prompt[:80], size, frame_num, seed, action_path,
            )

        video = self.pipeline.generate(
            prompt,
            img,
            action_path=action_path,
            chunk_size=chunk_size,
            max_area=MAX_AREA_CONFIGS[size],
            frame_num=frame_num,
            shift=sample_shift if sample_shift is not None else self.cfg.sample_shift,
            seed=seed,
            offload_model=False,                  # world_size > 1 → keep on GPU
            max_attention_size=max_attention_size,
        )

        if self.rank != 0:
            return ""

        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(Path(tempfile.gettempdir()) / "lingbot-fast" / f"{self.task}_{size.replace('*','x')}_{ts}.mp4")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Upstream save_video expects a [-1, 1] tensor with shape (C, T, H, W).
        save_video(
            tensor=video[None],
            save_file=output_path,
            fps=self.cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        logger.info("Video saved: %s", output_path)
        return output_path


def create_runner():
    return LingBotFastPipeline()
