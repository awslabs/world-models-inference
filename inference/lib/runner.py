# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for model runners.

Model authors subclass Runner and implement setup() + generate().
For realtime streaming models, override stream() instead of (or in addition to) generate().
The framework handles distributed coordination, job queue, HTTP routes, and WebSocket.
"""

from __future__ import annotations

import abc
import threading
from typing import Iterator, Optional

import torch


class ActionBuffer:
    """Thread-safe buffer holding the latest action from the client.

    The WebSocket reader writes actions here; the runner reads them at its own rate.
    This decouples client input rate from model generation rate.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._action: bytes = b""
        self._active = True
        self._new = threading.Event()

    def put(self, action: bytes) -> None:
        """Store the latest action (overwrites previous)."""
        with self._lock:
            self._action = action
        self._new.set()

    def get(self) -> Optional[bytes]:
        """Block until action available, return it. Returns None when session is over."""
        if not self._active and not self._new.is_set():
            return None
        self._new.wait(timeout=0.1)
        if not self._new.is_set() and not self._active:
            return None
        with self._lock:
            self._new.clear()
            return self._action

    def close(self) -> None:
        """Signal end of session."""
        self._active = False
        self._new.set()

    @property
    def active(self) -> bool:
        return self._active


class Runner(abc.ABC):
    @abc.abstractmethod
    def setup(self, ckpt_dir: str, device: torch.device, rank: int, world_size: int) -> None:
        """Load model weights. Called once at startup on each rank."""

    @abc.abstractmethod
    def generate(self, **params) -> str:
        """Run inference. Returns path to output file. Called on all ranks in lockstep."""

    @property
    def supports_streaming(self) -> bool:
        """Override to return True if this runner implements stream()."""
        return False

    def reset(self) -> None:
        """Called at the start of each streaming session. Override to reset per-session state."""
        pass

    def stream(self, actions: ActionBuffer) -> Iterator[bytes]:
        """Generate frames continuously, reading latest actions from the buffer.

        The model runs at its own rate (GPU-bound). Each iteration:
          1. Read latest action from actions.get()
          2. Run model forward pass
          3. Yield output frame bytes

        The session ends when actions.get() returns None (client disconnected).
        """
        raise NotImplementedError("Override stream() for realtime models")
