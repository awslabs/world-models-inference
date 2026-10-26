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

# Forward-dynamics sampling, as in NVIDIA's Cosmos3 action cookbook
# (cookbooks/cosmos3/generator/action/run_fd_with_diffusers.ipynb): 30 UniPC
# steps, no classifier-free guidance, flow shift 10.
ACTION_NUM_STEPS = 30
ACTION_GUIDANCE_SCALE = 1.0
ACTION_FLOW_SHIFT = 10.0
# Per-domain defaults from the same cookbook: (chunk_size, resolution_tier, fps).
ACTION_DOMAIN_DEFAULTS = {
    "av": (60, 480, 10),
    "camera_pose": (60, 480, 30),
    "umi": (16, 256, 20),
}


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

        # Action generation samples with its own scheduler; keep both and swap per job.
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        self.default_scheduler = self.pipe.scheduler
        self.action_scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=ACTION_FLOW_SHIFT, use_karras_sigmas=False
        )

        if rank == 0:
            logger.info("Cosmos3-Nano loaded successfully")

    def generate(self, **params) -> str:
        if params.get("action_mode"):
            return self._generate_action(**params)
        self.pipe.scheduler = self.default_scheduler
        prompt = params.get("prompt", "A cinematic aerial shot over a mountain range at sunrise.")
        # /generate and /invocations stage inputs as image_path; "image" is the older key.
        image_path = params.get("image_path") or params.get("image")
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

    def _generate_action(self, **params) -> str:
        """Forward dynamics: a start image plus an action trajectory → the video that follows.

        ``raw_actions`` is one row per step in the domain's own units (AV: 9D ego-pose
        deltas; UMI: 9D end-effector pose + 1D gripper), exactly as the cookbook's
        ``assets/actions/*.json`` files hold them.
        """
        from diffusers import CosmosActionCondition

        image_path = params.get("image_path") or params.get("image")
        if not image_path:
            raise ValueError("forward dynamics needs a start image")
        if not params.get("raw_actions"):
            raise ValueError("forward dynamics needs raw_actions")
        domain = params.get("domain_name", "av")
        chunk_default, tier_default, fps_default = ACTION_DOMAIN_DEFAULTS.get(domain, (60, 480, 10))
        chunk_size = int(params.get("chunk_size") or chunk_default)
        fps = int(float(params.get("fps") or fps_default))
        seed = int(params.get("seed", 0))
        raw_actions = torch.as_tensor(params["raw_actions"], dtype=torch.float32)

        self.pipe.scheduler = self.action_scheduler
        result = self.pipe(
            prompt=params.get("prompt") or "You are an autonomous vehicle planning system.",
            action=CosmosActionCondition(
                mode=params["action_mode"],
                chunk_size=chunk_size,
                domain_name=domain,
                resolution_tier=int(params.get("resolution_tier") or tier_default),
                raw_actions=raw_actions,
                image=Image.open(image_path).convert("RGB"),
                view_point=params.get("view_point", "ego_view"),
            ),
            fps=fps,
            num_inference_steps=int(params.get("num_inference_steps") or ACTION_NUM_STEPS),
            guidance_scale=float(params.get("guidance_scale") or ACTION_GUIDANCE_SCALE),
            use_system_prompt=False,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )

        output_path = params.get("output_path") or str(
            Path(tempfile.gettempdir()) / "cosmos3-outputs"
            / f"cosmos3_fd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.export_to_video(result.video, output_path, fps=fps, macro_block_size=1)
        if self.rank == 0:
            logger.info("Forward dynamics (%s, %d steps of actions): %s", domain, len(raw_actions), output_path)
        return output_path
