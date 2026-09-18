# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async job queue — stores submitted jobs and processes them via a worker thread."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Dict, Optional

from lib.distributed import CMD_GENERATE, broadcast_command
from lib.runner import Runner

logger = logging.getLogger(__name__)


class JobStore:
    """Thread-safe in-memory job store with a processing queue."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self.pending: "queue.Queue[str]" = queue.Queue()

    def submit(self, job_id: str, params: Dict[str, Any]) -> None:
        with self._lock:
            self._jobs[job_id] = {
                "status": "queued", "params": params,
                "submitted_at": time.time(), "started_at": None,
                "completed_at": None, "result_path": None, "error": None,
            }
        self.pending.put(job_id)

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._jobs.get(job_id)
            return dict(entry) if entry else None


def run_worker(runner: Runner, jobs: JobStore, stop: threading.Event):
    """Worker thread: drains the job queue, broadcasting each job to all ranks."""
    logger.info("Job worker started.")
    while not stop.is_set():
        try:
            job_id = jobs.pending.get(timeout=0.5)
        except queue.Empty:
            continue
        entry = jobs.get(job_id)
        if not entry:
            continue
        jobs.update(job_id, status="running", started_at=time.time())
        try:
            broadcast_command({"cmd": CMD_GENERATE, "args": entry["params"]})
            output = runner.generate(**entry["params"])
            jobs.update(job_id, status="complete", completed_at=time.time(), result_path=output)
            logger.info("Job %s done → %s", job_id, output)
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            jobs.update(job_id, status="failed", completed_at=time.time(), error=str(exc))
