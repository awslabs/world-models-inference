# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Echo runner — multi-GPU distributed inference test.

Exercises the full distributed path:
  - torchrun spawns N processes (one per GPU)
  - Each rank runs DummyUNet on its shard of frames
  - Rank 0 gathers results from all ranks via all_gather
  - Rank 0 encodes the final video

This validates:
  - NCCL init + process group
  - broadcast_command from rank 0 → all ranks
  - All ranks call generate() in lockstep
  - torch.distributed.all_gather works across GPUs
  - Only rank 0 writes output
"""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn

from lib.runner import Runner


class DummyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(16, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class EchoRunner(Runner):
    def setup(self, model_dir, device, rank, world_size):
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.model = DummyUNet().to(device)
        self.model.eval()

        if world_size > 1 and not dist.is_initialized():
            torch.cuda.set_device(device)
            dist.init_process_group(backend="nccl", init_method="env://",
                                    rank=rank, world_size=world_size)

    @torch.no_grad()
    def generate(self, **params):
        fps, total_frames = 24, 72
        h, w = 240, 320

        # Each rank generates its shard of frames
        frames_per_rank = total_frames // self.world_size
        start_frame = self.rank * frames_per_rank
        my_frames = frames_per_rank + (total_frames % self.world_size if self.rank == self.world_size - 1 else 0)

        noise = torch.randn(1, 3, my_frames, h, w, device=self.device)
        local_video = self.model(noise).squeeze(0)

        if self.world_size > 1:
            # All-gather: each rank contributes its shard
            # Pad to same size for gather
            padded = torch.zeros(3, frames_per_rank, h, w, device=self.device)
            padded[:, :local_video.shape[1]] = local_video[:, :frames_per_rank]
            gathered = [torch.zeros_like(padded) for _ in range(self.world_size)]
            dist.all_gather(gathered, padded)

            if self.rank == 0:
                video = torch.cat(gathered, dim=1)[:, :total_frames]
            else:
                return ""
        else:
            video = local_video

        # Only rank 0 encodes
        video = ((video.clamp(-1, 1) + 1) / 2 * 255).byte()
        frames_np = video.permute(1, 2, 3, 0).cpu().numpy()

        output_path = params.get("output_path") or os.path.join(tempfile.gettempdir(), "echo.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        import imageio
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
        for i in range(frames_np.shape[0]):
            writer.append_data(frames_np[i])
        writer.close()

        return output_path


def create_runner():
    return EchoRunner()
