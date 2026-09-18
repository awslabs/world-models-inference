# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP application for the inference server.

Routes:
  /ping                              health check (SM contract)
  /health                            detailed status
  /invocations                       sync inference (SM contract)
  /invocations-bidirectional-stream  WebSocket streaming (SM bidi)
  /ws                                WebSocket streaming (EC2)
  /generate                          async job submission
  /jobs/{id}                         job status
  /jobs/{id}/output                  download result
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.requests import Request
from starlette.responses import JSONResponse as StarletteJSON, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from lib.distributed import CMD_GENERATE, CMD_STREAM, broadcast_command, world_size
from lib.jobs import JobStore
from lib.runner import ActionBuffer, Runner
from lib.security import (
    AuthMiddleware,
    RateLimitMiddleware,
    SecurityConfig,
    SecurityHeadersMiddleware,
    extract_token,
    safe_subdir,
    token_valid,
)

logger = logging.getLogger(__name__)

JOB_OUTPUT_DIR = Path(tempfile.gettempdir()) / "world-model" / "jobs"
EXAMPLES_DIR = Path("/opt/ml/code/examples")


# =============================================================================
# App factory
# =============================================================================


def create_app(runner: Runner, jobs: JobStore) -> FastAPI:
    app = FastAPI(title="World Model Inference")
    config = SecurityConfig()
    config.log_summary()
    app.state.security = config

    # Middleware runs in reverse registration order (last added = outermost).
    # Register so the effective order is: headers -> CORS -> rate limit -> auth.
    app.add_middleware(AuthMiddleware, config=config)
    app.add_middleware(RateLimitMiddleware, config=config)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,  # deny-by-default when empty
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    JOB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _register_health(app)
    _register_invocations(app, runner)
    if runner.supports_streaming:
        _register_streaming(app, runner, config)
    _register_jobs(app, jobs)

    return app


# =============================================================================
# Health
# =============================================================================


def _register_health(app: FastAPI):
    @app.get("/ping")
    def ping():
        return {"status": "healthy"}

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "model_loaded": True,
            "world_size": world_size(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }


# =============================================================================
# /invocations — SageMaker sync endpoint
# =============================================================================


def _register_invocations(app: FastAPI, runner: Runner):
    async def invocations(request: Request):
        body = await request.body()

        if runner.supports_streaming:
            actions = ActionBuffer()
            actions.put(body)
            actions.close()
            output = next(runner.stream(actions), b"")
            return StreamingResponse(iter([output]), media_type="application/octet-stream")

        params = json.loads(body)
        params.setdefault("output_path", str(JOB_OUTPUT_DIR / f"{uuid.uuid4().hex}.mp4"))
        broadcast_command({"cmd": CMD_GENERATE, "args": params})
        output = runner.generate(**params)
        return StarletteJSON({"status": "complete", "output_path": output})

    app.routes.insert(0, Route("/invocations", invocations, methods=["POST"]))


# =============================================================================
# WebSocket streaming — /ws (EC2) and /invocations-bidirectional-stream (SM)
# =============================================================================


def _register_streaming(app: FastAPI, runner: Runner, config: SecurityConfig):
    async def ws_handler(ws: WebSocket):
        # Enforce auth before accepting the socket. The browser client passes the
        # token as a ?token= query param (WebSocket has no Authorization header).
        if config.auth_enabled:
            presented = extract_token(
                ws.headers.get("authorization"),
                ws.query_params.get("token"),
            )
            if not token_valid(config, presented):
                await ws.close(code=1008)  # policy violation
                logger.info("Rejected unauthenticated WebSocket connection")
                return

        await ws.accept()
        logger.info("Stream session started")
        runner.reset()
        broadcast_command({"cmd": CMD_STREAM})

        actions = ActionBuffer()

        async def read_from_client():
            try:
                while True:
                    actions.put(await ws.receive_bytes())
            except WebSocketDisconnect:
                pass
            finally:
                actions.close()

        async def write_to_client():
            loop = asyncio.get_event_loop()
            output = runner.stream(actions)
            while actions.active:
                chunk = await loop.run_in_executor(None, lambda: next(output, None))
                if chunk is None:
                    break
                await ws.send_bytes(chunk)

        reader = asyncio.create_task(read_from_client())
        try:
            await write_to_client()
        finally:
            actions.close()
            reader.cancel()
            try:
                await ws.close()
            except Exception:
                pass
        logger.info("Stream session ended")

    app.routes.insert(0, WebSocketRoute("/ws", ws_handler))
    app.routes.insert(0, WebSocketRoute("/invocations-bidirectional-stream", ws_handler))


# =============================================================================
# Async job queue — /generate, /jobs/{id}, /jobs/{id}/output
# =============================================================================


def _register_jobs(app: FastAPI, jobs: JobStore):
    @app.post("/generate")
    async def submit_job(
        prompt: str = Form(""),
        image: Optional[UploadFile] = File(None),
        example_id: Optional[str] = Form(None),
        frame_num: Optional[str] = Form(None),
        size: Optional[str] = Form(None),
        sampling_steps: Optional[str] = Form(None),
        guide_scale: Optional[str] = Form(None),
        seed: Optional[str] = Form(None),
        use_dmd: Optional[str] = Form(None),
        export_ply: Optional[str] = Form(None),
    ):
        job_dir = JOB_OUTPUT_DIR / uuid.uuid4().hex
        job_dir.mkdir(parents=True, exist_ok=True)

        image_path = None
        if image and image.filename:
            image_path = job_dir / "input.png"
            with image_path.open("wb") as f:
                shutil.copyfileobj(image.file, f)
        elif example_id:
            # Validate + bounds-check the id so it cannot escape EXAMPLES_DIR.
            candidate = safe_subdir(EXAMPLES_DIR, example_id) / "image.jpg"
            if candidate.exists():
                image_path = candidate

        params = {"prompt": prompt, "output_path": str(job_dir / "output.mp4")}
        if image_path:
            params["image_path"] = str(image_path)
        # Forward the optional generation knobs when supplied. The route accepts
        # them as strings (multipart form); runners coerce/interpret as needed.
        # Booleans use a truthy-string check so `use_dmd=true|1|yes` all work.
        def _truthy(v):
            return str(v).strip().lower() in ("1", "true", "yes", "on")
        if frame_num is not None:
            params["frame_num"] = frame_num
        if size is not None:
            params["size"] = size
        if sampling_steps is not None:
            params["sampling_steps"] = sampling_steps
        if guide_scale is not None:
            params["guide_scale"] = guide_scale
        if seed is not None:
            params["seed"] = seed
        if use_dmd is not None:
            params["use_dmd"] = _truthy(use_dmd)
        # export_ply is a forward-looking hook for Lyra-2's Step-2 3D reconstruction
        # (video -> Gaussian-splat .ply). No runner consumes it yet; see
        # inference/models/lyra-2/NOTES.md ("Future work"). Forwarded so wiring it
        # later needs no route change.
        if export_ply is not None:
            params["export_ply"] = _truthy(export_ply)
        jobs.submit(job_dir.name, params)
        return JSONResponse({"job_id": job_dir.name, "status": "queued"})

    @app.get("/jobs/{job_id}")
    @app.get("/status/{job_id}")
    def job_status(job_id: str):
        entry = jobs.get(job_id)
        if not entry:
            raise HTTPException(404)
        return {
            "job_id": job_id,
            "status": entry["status"],
            "submitted_at": entry["submitted_at"],
            "started_at": entry["started_at"],
            "completed_at": entry["completed_at"],
            "error": entry["error"],
        }

    @app.get("/jobs/{job_id}/output")
    @app.get("/result/{job_id}")
    def job_output(job_id: str):
        entry = jobs.get(job_id)
        if not entry:
            raise HTTPException(404)
        if entry["status"] != "complete":
            raise HTTPException(409, f"Job status: {entry['status']}")
        path = entry["result_path"]
        if not path or not Path(path).exists():
            raise HTTPException(500, "Result file missing")
        return FileResponse(path, media_type="application/octet-stream", filename=f"{job_id}.mp4")

    @app.get("/examples/{example_id}/image")
    def example_image(example_id: str):
        # Validate + bounds-check the id so it cannot escape EXAMPLES_DIR.
        examples_dir = safe_subdir(EXAMPLES_DIR, example_id)
        for ext in ("image.jpg", "image.png"):
            p = examples_dir / ext
            if p.exists():
                return FileResponse(p)
        raise HTTPException(404, f"No image for example {example_id}")
