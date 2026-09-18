# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the distributed inference server.

Launch: torchrun --nproc_per_node=N -m lib.serve --model-dir /path --code-dir /path

Orchestrates:
  1. Discovers and loads the Runner from code-dir
  2. Calls runner.setup() on all ranks
  3. Rank 0: starts HTTP server + async job worker
  4. Ranks 1..N: enters follower loop
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import threading
from pathlib import Path

import torch

from lib.app import create_app
from lib.distributed import CMD_SHUTDOWN, broadcast_command, follower_loop, rank_zero, shutdown_cluster
from lib.jobs import JobStore, run_worker
from lib.runner import Runner

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [rank %(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(str(os.environ.get("RANK", 0)))


def discover_runner(code_dir: str) -> Runner:
    """Import runner.py from code_dir and call create_runner()."""
    sys.path.insert(0, code_dir)
    mod = importlib.import_module("runner")
    if not hasattr(mod, "create_runner"):
        raise RuntimeError(f"{code_dir}/runner.py must define create_runner() -> Runner")
    return mod.create_runner()


def main():
    parser = argparse.ArgumentParser(description="World model inference server")
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--code-dir", default=os.environ.get("SM_MODULE_DIR", "/opt/ml/code"))
    parser.add_argument("--port", type=int, default=8080)
    # Bind all interfaces: the server runs inside a container whose only exposed
    # port is reached through the ALB; the GPU instance itself sits in a private
    # subnet with no public IP. Binding to 0.0.0.0 is required for the container
    # port mapping to work.
    parser.add_argument("--host", default="0.0.0.0")  # nosec B104 - container in private subnet, ALB-only ingress
    parser.add_argument("cmd", nargs="*")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.code_dir).parent.parent))

    runner = discover_runner(args.code_dir)
    rank = int(os.environ.get("RANK", 0))
    ws = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    runner.setup(args.model_dir, device, rank, ws)

    if rank_zero():
        jobs = JobStore()
        stop = threading.Event()
        threading.Thread(target=run_worker, args=(runner, jobs, stop), daemon=True).start()

        app = create_app(runner, jobs)
        import uvicorn
        try:
            uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        finally:
            try:
                broadcast_command({"cmd": CMD_SHUTDOWN})
            except Exception:
                pass
            stop.set()
            shutdown_cluster()
    else:
        try:
            follower_loop(runner)
        finally:
            shutdown_cluster()


if __name__ == "__main__":
    main()
