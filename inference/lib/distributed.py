# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed coordination for multi-GPU inference.

Provides two things:
  1. NCCL primitives: broadcast_command, rank_zero, world_size, barrier, shutdown
  2. Follower loop: ranks 1..N wait for commands and execute in lockstep with rank 0

Commands:
  CMD_GENERATE — all ranks call runner.generate() with identical args
  CMD_STREAM   — all ranks enter runner.stream(); per-frame sync is
                 dist.broadcast inside the runner, not here
  CMD_SHUTDOWN — followers exit, process group destroyed
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from lib.runner import ActionBuffer, Runner

logger = logging.getLogger(__name__)

CMD_GENERATE = "generate"
CMD_STREAM = "stream"
CMD_PING = "ping"
CMD_SHUTDOWN = "shutdown"


# =============================================================================
# NCCL primitives
# =============================================================================


def broadcast_command(command: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Broadcast a command dict from rank 0 to all ranks.

    Rank 0 passes the command; other ranks pass None.
    Returns the command on every rank.
    """
    if not dist.is_initialized():
        return command or {"cmd": CMD_PING}

    payload: List[Optional[Dict[str, Any]]] = [command if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0] or {"cmd": CMD_PING}


def rank_zero() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    if dist.is_initialized():
        dist.barrier()


def shutdown_cluster():
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as exc:
            logger.warning("destroy_process_group failed: %s", exc)


def cuda_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return torch.device(f"cuda:{local_rank}")


# =============================================================================
# Follower loop (ranks 1..N)
# =============================================================================


def follower_loop(runner: Runner):
    """Block forever, executing commands from rank 0 until shutdown.

    CMD_GENERATE: call runner.generate() — the model handles internal parallelism
    CMD_STREAM:   enter runner.stream() with an infinite iterator; the runner
                  receives real per-frame data via dist.broadcast internally
    CMD_SHUTDOWN: return
    """
    logger.info("Follower ready (rank %s).", os.environ.get("RANK"))

    while True:
        cmd = broadcast_command(None)
        kind = cmd.get("cmd")

        if kind == CMD_SHUTDOWN:
            return

        elif kind == CMD_GENERATE:
            try:
                runner.generate(**cmd["args"])
            except Exception as exc:
                logger.exception("Follower generate error: %s", exc)

        elif kind == CMD_STREAM:
            try:
                runner.reset()
                # Followers get a dummy ActionBuffer — real action data arrives
                # via dist.broadcast inside the runner, not from the client.
                dummy = ActionBuffer()
                dummy.put(b"")
                for _ in runner.stream(dummy):
                    pass
            except Exception as exc:
                logger.exception("Follower stream ended: %s", exc)
