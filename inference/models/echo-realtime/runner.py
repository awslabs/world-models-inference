# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Echo realtime runner — multi-GPU action-conditioned frame streaming.

Distributed streaming protocol:
  - Rank 0: receives actions from client, broadcasts position tensor to all ranks
  - All ranks: run model on their GPU, produce a tile
  - All ranks: all_gather tiles → rank 0 assembles and yields to client
  - End signal: rank 0 broadcasts position[0] = inf → all ranks exit stream()

Single-GPU mode works identically (no dist ops, full frame per rank).
"""

import struct
import json
import tempfile
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn

from lib.runner import ActionBuffer, Runner

KEYMAP = {
    ord('w'): [0.0, 1.0, 0.0, 0.0],
    ord('a'): [-1.0, 0.0, 0.0, 0.0],
    ord('s'): [0.0, -1.0, 0.0, 0.0],
    ord('d'): [1.0, 0.0, 0.0, 0.0],
    ord(' '): [0.0, 0.0, 1.0, 0.0],
}


def parse_action(data: bytes) -> list:
    if len(data) == 16:
        return list(struct.unpack('<4f', data))
    try:
        obj = json.loads(data)
        if "action" in obj:
            return obj["action"][:4]
        keys = obj.get("keys", "")
        vec = [0.0, 0.0, 0.0, 0.0]
        for ch in keys.encode():
            if ch in KEYMAP:
                for i, v in enumerate(KEYMAP[ch]):
                    vec[i] += v
        return vec
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    vec = [0.0, 0.0, 0.0, 0.0]
    for ch in data:
        if ch in KEYMAP:
            for i, v in enumerate(KEYMAP[ch]):
                vec[i] += v
    return vec


class TileGen(nn.Module):
    """Each rank generates a 32x32 tile conditioned on position + rank_id."""

    def __init__(self, rank_id: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.SiLU(),
            nn.Linear(64, 3 * 32 * 32),
        )
        self.rank_id = rank_id

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        rid = torch.full((position.shape[0], 1), self.rank_id, device=position.device)
        x = torch.cat([position, rid], dim=1)
        return self.net(x).view(-1, 3, 32, 32)


class EchoRealtimeRunner(Runner):
    def setup(self, model_dir, device, rank, world_size):
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.model = TileGen(rank_id=float(rank)).to(device)
        self.model.eval()
        self.frame_count = 0
        self.position = torch.zeros(1, 4, device=device)

        if world_size > 1 and not dist.is_initialized():
            torch.cuda.set_device(device)
            dist.init_process_group(backend="nccl", init_method="env://",
                                    rank=rank, world_size=world_size)

    def generate(self, **params):
        return params.get("output_path", str(Path(tempfile.gettempdir()) / "echo-rt.bin"))

    @property
    def supports_streaming(self) -> bool:
        return True

    def reset(self):
        self.frame_count = 0
        self.position = torch.zeros(1, 4, device=self.device)

    @torch.no_grad()
    def stream(self, actions: ActionBuffer) -> Iterator[bytes]:
        """Generate frames continuously at GPU speed, reading latest action each tick."""
        is_leader = self.rank == 0

        while actions.active:
            if is_leader:
                action_bytes = actions.get()
                if action_bytes is None:
                    break
                action_vec = parse_action(action_bytes)
                for i in range(4):
                    self.position[0, i] += action_vec[i] * 0.1

            # Broadcast position from rank 0 to all
            if self.world_size > 1:
                dist.broadcast(self.position, src=0)

            # Check sentinel (position[0] == inf means end)
            if self.position[0, 0].item() == float('inf'):
                return

            # Generate tile on this rank's GPU
            tile = self.model(self.position).squeeze(0)
            tile_pixels = ((tile.clamp(-1, 1) + 1) / 2 * 255).byte()

            if self.world_size > 1:
                gathered = [torch.zeros_like(tile_pixels) for _ in range(self.world_size)]
                dist.all_gather(gathered, tile_pixels)
            else:
                gathered = [tile_pixels]

            # Only rank 0 yields output
            if not is_leader:
                continue

            # Assemble tiles into grid
            cols = 2 if self.world_size >= 2 else 1
            rows = (self.world_size + cols - 1) // cols
            frame_h, frame_w = rows * 32, cols * 32
            frame = torch.zeros(3, frame_h, frame_w, dtype=torch.uint8, device='cpu')
            for idx, t in enumerate(gathered):
                r, c = divmod(idx, cols)
                frame[:, r*32:(r+1)*32, c*32:(c+1)*32] = t.cpu()

            self.frame_count += 1
            pixels = frame.permute(1, 2, 0).numpy().tobytes()

            response_meta = json.dumps({
                "frame": self.frame_count,
                "action_received": action_vec,
                "position": self.position[0].cpu().tolist(),
                "world_size": self.world_size,
                "frame_size": [frame_h, frame_w],
            }).encode()

            header = struct.pack("<II", self.frame_count, len(response_meta))
            yield header + response_meta + pixels

        # Signal followers to exit
        if is_leader and self.world_size > 1:
            self.position[0, 0] = float('inf')
            dist.broadcast(self.position, src=0)


def create_runner():
    return EchoRealtimeRunner()
