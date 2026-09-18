# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Template Runner — copy this file and fill in setup() + generate().

The shared server (`inference/lib/serve`) does everything else: it starts
torchrun with one process per GPU, calls setup() once on every rank, exposes the
secured FastAPI app on rank 0, and broadcasts each job so all ranks run
generate() in lockstep. You never write a server.

Contract:
  setup(ckpt_dir, device, rank, world_size) -> None    load the model once
  generate(**params) -> str                            return a path to the output
  stream(actions) -> Iterator[bytes]                   real-time models only
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from lib.runner import Runner

logger = logging.getLogger(__name__)


class TemplateRunner(Runner):
    def setup(self, ckpt_dir: str, device, rank: int, world_size: int) -> None:
        """Load the model. Called once per rank before any request is served.

        `ckpt_dir` is where the weights declared in endpoint.yaml's `hf:` block
        were synced from S3. Keep everything expensive here, not in generate().
        """
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.ckpt_dir = ckpt_dir

        # e.g.
        # self.pipe = YourPipeline.from_pretrained(ckpt_dir).to(device).eval()
        raise NotImplementedError("Implement setup() for your model")

    def generate(self, **params) -> str:
        """Run one job and return the path to the file it produced.

        `params` are the form fields the UI/API sent (prompt, seed, ...). Write
        output under the system temp dir; the server streams it back and, on the
        SageMaker async path, uploads it to S3.

        In a multi-GPU deployment every rank runs this together — only rank 0's
        return value is used, so non-zero ranks may return "" after their share
        of the collective work.
        """
        prompt = params.get("prompt", "")
        seed = int(params.get("seed", 0))
        logger.info("generate: prompt=%r seed=%s", prompt[:60], seed)

        out_dir = Path(tempfile.gettempdir()) / "template-output"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = params.get("output_path") or str(out_dir / "output.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # e.g. frames = self.pipe(prompt=prompt, generator=...).frames
        #      write_video(frames, output_path)
        raise NotImplementedError("Implement generate() for your model")

    # Real-time models only: set supports_streaming and implement stream().
    #
    # @property
    # def supports_streaming(self) -> bool:
    #     return True
    #
    # def stream(self, actions):
    #     for action in actions:
    #         yield self.step(action)  # bytes, e.g. an encoded JPEG frame


def create_runner() -> Runner:
    """Required factory — the shared server imports this by name."""
    return TemplateRunner()
