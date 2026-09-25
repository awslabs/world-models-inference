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
import concurrent.futures
import json
import logging
import os
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

# How many frames may be in flight before the writer waits for the client to
# acknowledge one. Sized as a bandwidth-delay product: sustaining F fps over a
# link with round-trip time R needs F*R frames unacknowledged, so 16 covers
# 64 fps at 250 ms. Too small throttles throughput; too large is the standing
# queue this exists to prevent. 0 disables flow control entirely.
FLOW_WINDOW = int(os.environ.get("WORLD_MODEL_FLOW_WINDOW", "16"))

# How long the writer waits for a frame acknowledgement before deciding the
# client does not speak flow control and reverting to unbounded sending.
FLOW_ACK_TIMEOUT = 2.0

# How often a connection waiting for the session gate wakes up to re-preempt
# whatever session is currently published, and how long it keeps trying before
# giving up. The retry matters for correctness, not just latency: a session
# that publishes its buffer *after* a waiter's preempt check would otherwise
# never be told to stop, and the waiter would sleep on a gate nobody is going
# to release. The overall cap turns "wedged forever" into a clean 1013 close.
GATE_RETRY_INTERVAL = 0.5
GATE_ACQUIRE_TIMEOUT = 15.0


def _int(value: object) -> int:
    """Best-effort int from untrusted client JSON; 0 when unusable."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


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
    # Only ONE session can run at a time. A real-time session drives the whole
    # GPU cluster (rank 0 leads, ranks 1..N-1 follow in lockstep), so two
    # overlapping CMD_STREAM broadcasts desync the ranks and wedge the cluster;
    # single-GPU runners keep per-session state (frame context, KV cache) that
    # a second session would reset underneath the first. A browser in React
    # StrictMode opens two sockets on mount, which is exactly how this
    # surfaces. Serialise with a lock, and let a new connection preempt the
    # current one (last player wins) by closing its action buffer, which makes
    # the runner's loop end and release every rank cleanly.
    #
    # threading.Lock, not asyncio.Lock: connections are not guaranteed to share
    # an event loop (starlette's TestClient gives every websocket session its
    # own portal loop), and an asyncio.Lock released on one loop never wakes a
    # waiter parked on another — the second session hangs forever. A thread
    # lock acquired off the loop (run_in_executor) is correct on a single
    # shared loop and across loops alike.
    session_gate = threading.Lock()
    active_actions: dict[str, Optional[ActionBuffer]] = {"buf": None}

    # Blocking work (gate waits, runner.reset, generator teardown) gets its own
    # small executor. It must NOT share the default pool: write_to_client pulls
    # every frame through the default executor, and that pool is a fixed size
    # (min(32, cpus+4)). A few dozen connections parked in it waiting for the
    # gate would starve the frame pump — and the gate only releases when the
    # frame pump finishes, so the whole server deadlocks. On this pool the
    # worst a pile-up can do is queue behind other waiters, all of which are
    # bounded by GATE_ACQUIRE_TIMEOUT.
    gate_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="session-gate"
    )

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

        # Preempt any session already running before we take the cluster, then
        # wait for the gate in bounded slices, re-preempting on every timeout.
        # A single preempt-then-block-forever sequence has a hole: a connection
        # arriving while a preemption is already in flight sees no published
        # buffer (the outgoing session cleared it, the incoming one hasn't
        # published yet), closes nothing, and would block on the gate for a
        # session it never asked to stop — an accepted socket that silently
        # never serves a frame. Retrying the preempt from inside the wait loop
        # closes that hole, and the deadline guarantees no waiter outlives its
        # welcome: past it we tell the client to try again later (1013) instead
        # of holding a dead connection open.
        loop = asyncio.get_event_loop()
        deadline = loop.time() + GATE_ACQUIRE_TIMEOUT
        while True:
            if active_actions["buf"] is not None:
                logger.info("New session preempting the active one")
                active_actions["buf"].close()
            acquired = await loop.run_in_executor(
                gate_executor, session_gate.acquire, True, GATE_RETRY_INTERVAL
            )
            if acquired:
                break
            if loop.time() >= deadline:
                logger.warning(
                    "Session gate not acquired within %.0fs; refusing connection",
                    GATE_ACQUIRE_TIMEOUT,
                )
                try:
                    await ws.close(code=1013)  # try again later
                except Exception:
                    pass
                return
        try:
            logger.info("Stream session started")
            # The catalogue frontend waits for this before sending its 'start'
            # message (which may carry a seed image for the session). Advertising
            # the flow-control window here is what tells the client to acknowledge
            # frames; clients that predate it simply do not, and the writer notices.
            await ws.send_json({"type": "connected", "flow_window": FLOW_WINDOW})
            # reset() blocks until the runner's device thread drains its current
            # work (waypoint submits to a single-threaded device executor and
            # waits on the result). Called inline it would stall this event loop
            # — and /ping with it — for as long as teardown takes.
            await loop.run_in_executor(gate_executor, runner.reset)
            broadcast_command({"cmd": CMD_STREAM})

            actions = ActionBuffer()
            active_actions["buf"] = actions
            window = FLOW_WINDOW
            sent = acked = 0
            client_acks = False
            credit = asyncio.Event()

            async def read_from_client():
                # Browsers send JSON control messages as *text* frames and (some
                # clients) low-level actions as binary frames. Accept both: text
                # protocol chatter (ping/ack/start) is answered here, and every
                # payload is forwarded to the runner, which owns the action
                # format. Legacy binary-only clients are unaffected.
                try:
                    while True:
                        msg = await ws.receive()
                        if msg["type"] == "websocket.disconnect":
                            break
                        data = msg.get("bytes")
                        if data is None:
                            text = msg.get("text") or ""
                            data = text.encode()
                            try:
                                obj = json.loads(text)
                            except (json.JSONDecodeError, ValueError):
                                obj = None
                            if isinstance(obj, dict):
                                if obj.get("type") == "ping":
                                    await ws.send_json(
                                        {"type": "pong", "timestamp": obj.get("timestamp")}
                                    )
                                    continue
                                if obj.get("type") == "ack":
                                    # Cumulative count of frames the client has
                                    # received. Must not reach the action buffer:
                                    # the runner reads an unknown message type as a
                                    # neutral action, which would cancel held keys.
                                    nonlocal acked, client_acks
                                    acked = max(acked, _int(obj.get("n")))
                                    client_acks = True
                                    credit.set()
                                    continue
                                if obj.get("type") == "start":
                                    actions.put(data)
                                    await ws.send_json({"type": "started"})
                                    continue
                        actions.put(data)
                except WebSocketDisconnect:
                    pass
                finally:
                    actions.close()

            output = runner.stream(actions)

            async def write_to_client():
                nonlocal sent, window
                while actions.active:
                    chunk = await loop.run_in_executor(None, lambda: next(output, None))
                    if chunk is None:
                        break
                    # Bound the frames in flight. Every buffer between here and the
                    # browser — kernel send queue, load balancer, receive queue —
                    # will hold seconds of 720p JPEG. They fill during the opening
                    # burst, and because generation then settles to whatever the
                    # link carries, in-rate equals out-rate and that queue never
                    # drains: the player sees a picture seconds behind their input
                    # for the rest of the session. Waiting on acknowledgements keeps
                    # the standing queue at `window` frames instead of at whatever
                    # the path happens to hold.
                    while window:
                        credit.clear()
                        if sent - acked < window:
                            break
                        try:
                            await asyncio.wait_for(credit.wait(), FLOW_ACK_TIMEOUT)
                        except asyncio.TimeoutError:
                            if not client_acks:
                                logger.info(
                                    "Client does not acknowledge frames; "
                                    "sending unbounded"
                                )
                                window = 0
                            break
                    await ws.send_bytes(chunk)
                    sent += 1

            reader = asyncio.create_task(read_from_client())
            try:
                await write_to_client()
            except WebSocketDisconnect:
                pass
            finally:
                actions.close()
                reader.cancel()
                # Close the frame generator explicitly, off the event loop. A
                # preempted session leaves write_to_client via `while
                # actions.active` without exhausting the generator; if we left
                # it for the GC, GeneratorExit would run stream()'s finally on
                # this loop thread, where waypoint's runner waits up to 30 s
                # for the in-flight device call — freezing every session and
                # /ping (the ALB health check kills the instance) with it.
                # close() on an already-finished generator is a no-op, so the
                # clean-exhaustion path is unaffected.
                try:
                    await loop.run_in_executor(gate_executor, output.close)
                except Exception:
                    logger.exception("Stream generator close failed")
                # Only clear the shared slot if it is still ours: a preempting
                # connection has already replaced it with its own buffer.
                if active_actions["buf"] is actions:
                    active_actions["buf"] = None
                try:
                    await ws.close()
                except Exception:
                    pass
            logger.info("Stream session ended")
        finally:
            session_gate.release()

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
