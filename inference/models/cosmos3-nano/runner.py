# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-Nano Runner — wraps Cosmos3OmniPipeline as a cartridge.

Supports three generation modes:
  - text-to-video: prompt → 720p video (up to 300 frames)
  - image-to-video: image + prompt → 720p video
  - forward-dynamics: image + action trajectory → predicted future video
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from lib.runner import Runner

logger = logging.getLogger(__name__)

DEFAULT_NUM_FRAMES = 189
DEFAULT_HEIGHT = 720
DEFAULT_WIDTH = 1280
DEFAULT_FPS = 24.0
DEFAULT_GUIDANCE_SCALE = 7.0
DEFAULT_NUM_STEPS = 60


def create_runner():
    return Cosmos3NanoPipeline()


class Cosmos3NanoPipeline(Runner):
    """Cosmos3-Nano wrapped as a generic Runner."""

    def setup(self, ckpt_dir: str, device: torch.device, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = device

        if rank == 0:
            logger.info("Loading Cosmos3-Nano from %s", ckpt_dir)

        from diffusers import Cosmos3OmniPipeline
        from diffusers.utils import export_to_video

        self.export_to_video = export_to_video

        self.pipe = Cosmos3OmniPipeline.from_pretrained(
            ckpt_dir,
            torch_dtype=torch.bfloat16,
            device_map="balanced" if world_size > 1 else "cuda",
        )

        if rank == 0:
            logger.info("Cosmos3-Nano loaded successfully")

    def generate(self, **params) -> str:
        prompt = params.get("prompt", "A cinematic aerial shot over a mountain range at sunrise.")
        image_path = params.get("image")
        num_frames = int(params.get("num_frames", DEFAULT_NUM_FRAMES))
        height = int(params.get("height", DEFAULT_HEIGHT))
        width = int(params.get("width", DEFAULT_WIDTH))
        fps = float(params.get("fps", DEFAULT_FPS))
        guidance_scale = float(params.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
        num_inference_steps = int(params.get("num_inference_steps", DEFAULT_NUM_STEPS))
        seed = int(params.get("seed", 42))

        generator = torch.Generator(device="cuda").manual_seed(seed)

        pipe_kwargs = dict(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        if image_path:
            pipe_kwargs["image"] = Image.open(image_path).convert("RGB")

        result = self.pipe(**pipe_kwargs)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(tempfile.gettempdir()) / "cosmos3-outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"cosmos3_{timestamp}.mp4")

        self.export_to_video(result.video, output_path, fps=int(fps), macro_block_size=1)

        if self.rank == 0:
            logger.info("Generated video: %s (%d frames, %dx%d)", output_path, num_frames, width, height)

        return output_path
