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
import base64
import binascii
import concurrent.futures
import json
import logging
import math
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
    WS_BEARER_PROTOCOL,
    SecurityHeadersMiddleware,
    extract_token,
    safe_subdir,
    token_valid,
    ws_bearer_token,
    ws_offered_protocols,
)

logger = logging.getLogger(__name__)

JOB_OUTPUT_DIR = Path(tempfile.gettempdir()) / "world-model" / "jobs"
EXAMPLES_DIR = Path("/opt/ml/code/examples")

# Largest /invocations body accepted, in bytes. Requests carry a JSON parameter
# object (inputs arrive via /generate) or a real-time action packet, both tiny;
# the cap stops one caller buffering an arbitrarily large body into the GPU
# host's memory. Override with WORLD_MODEL_MAX_BODY_BYTES.
_DEFAULT_MAX_BODY_BYTES = 8 * 1024 * 1024
try:
    MAX_BODY_BYTES = max(1, int(os.environ.get("WORLD_MODEL_MAX_BODY_BYTES", _DEFAULT_MAX_BODY_BYTES)))
except ValueError:
    logger.warning("Invalid WORLD_MODEL_MAX_BODY_BYTES; using %d.", _DEFAULT_MAX_BODY_BYTES)
    MAX_BODY_BYTES = _DEFAULT_MAX_BODY_BYTES


class BodyTooLarge(Exception):
    pass


async def _read_body_capped(request: Request, limit: int) -> bytes:
    """Read the request body, refusing it as soon as it exceeds ``limit`` bytes."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise BodyTooLarge()
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise BodyTooLarge()
        chunks.append(chunk)
    return b"".join(chunks)


# The generation parameters a client may send to /invocations — the knobs the
# cartridges' generate() methods document. Anything else is refused, rather than
# forwarded as a keyword argument: in particular filesystem paths (image,
# image_path, action_path, output_path — an input image arrives as example_id or
# image_b64 and is staged server-side, see INVOCATION_IMAGE_KEYS; outputs go to a
# server-chosen file) and server-side settings: lyra-2's config name (`experiment`)
# and lingbot-fast's `max_attention_size`, a GPU-memory knob, are left to the
# cartridge defaults on purpose.
GENERATE_PARAM_KEYS = frozenset({
    "prompt", "seed", "size", "resolution", "height", "width", "fps",
    "frame_num", "num_frames", "sampling_steps", "num_inference_steps",
    "guide_scale", "guidance_scale", "use_dmd", "export_ply",
    "chunk_size", "sample_shift",
    "zoom_in_strength", "zoom_out_strength", "num_frames_zoom_in", "num_frames_zoom_out",
})

# Upper bounds on the numeric knobs, so one request cannot ask every GPU rank
# (they run each job in lockstep) for an effectively unbounded amount of work.
# Deliberately generous — several times the largest value any cartridge uses
# (lyra-2 renders 322 frames; step counts are single or low double digits) — so
# they only ever refuse abuse, never a real request. Values keep their type;
# runners coerce as before.
NUMERIC_PARAM_MAX = {
    "frame_num": 1024, "num_frames": 1024, "num_frames_zoom_in": 1024, "num_frames_zoom_out": 1024,
    "sampling_steps": 200, "num_inference_steps": 200,
    "height": 4096, "width": 4096, "fps": 120, "chunk_size": 64,
    "guide_scale": 100, "guidance_scale": 100, "sample_shift": 100,
    "zoom_in_strength": 100, "zoom_out_strength": 100,
    "seed": 2**63 - 1,
}
# Lower bounds are 0 except where a negative value is meaningful: the UI sends
# seed=-1 to mean "random".
NUMERIC_PARAM_MIN = {"seed": -(2**63)}
STRING_PARAM_MAX_CHARS = {"prompt": 8000, "size": 32, "resolution": 32}
# Counts and dimensions the runners int() — reject "1e3", "12.5" etc. up front
# with a 400 instead of letting the runner fail with a 500.
INTEGER_PARAM_KEYS = frozenset({
    "frame_num", "num_frames", "num_frames_zoom_in", "num_frames_zoom_out",
    "sampling_steps", "num_inference_steps", "height", "width", "chunk_size", "seed",
})


# How a client gives /invocations an input image without naming a server path:
# the id of a bundled example, or the image itself, base64-encoded (bounded by
# MAX_BODY_BYTES). Either is staged to a server-chosen file and handed to the
# runner as image_path. PNG and JPEG only, recognised by their magic bytes.
INVOCATION_IMAGE_KEYS = ("example_id", "image_b64")
_IMAGE_MAGIC = {b"\x89PNG\r\n\x1a\n": "png", b"\xff\xd8\xff": "jpg"}


def _stage_invocation_image(params: dict) -> Optional[str]:
    """Replace example_id/image_b64 in ``params`` with a staged image_path.

    Returns a reason the input is unusable, or None.
    """
    example_id = params.pop("example_id", None)
    image_b64 = params.pop("image_b64", None)
    if example_id is not None and image_b64 is not None:
        return "Send example_id or image_b64, not both"
    if example_id is not None:
        if not isinstance(example_id, str):
            return "example_id must be a string"
        candidate = safe_subdir(EXAMPLES_DIR, example_id) / "image.jpg"
        if not candidate.exists():
            return f"No bundled example {example_id!r}"
        params["image_path"] = str(candidate)
    elif image_b64 is not None:
        if not isinstance(image_b64, str):
            return "image_b64 must be a base64 string"
        try:
            data = base64.b64decode(image_b64, validate=True)
        except (binascii.Error, ValueError):
            return "image_b64 is not valid base64"
        ext = next((e for magic, e in _IMAGE_MAGIC.items() if data.startswith(magic)), None)
        if ext is None:
            return "image_b64 must be a PNG or JPEG image"
        job_dir = JOB_OUTPUT_DIR / uuid.uuid4().hex
        job_dir.mkdir(parents=True, exist_ok=True)
        path = job_dir / f"input.{ext}"
        path.write_bytes(data)
        params["image_path"] = str(path)
    return None


def invalid_generation_value(params: dict) -> Optional[str]:
    """A reason ``params`` is out of bounds, or None when every value is acceptable."""
    for key, limit in NUMERIC_PARAM_MAX.items():
        if key not in params or params[key] is None:
            continue
        value = params[key]
        if isinstance(value, bool):
            return f"{key} must be a number"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return f"{key} must be a number"
        low = NUMERIC_PARAM_MIN.get(key, 0)
        if not math.isfinite(number) or number < low or number > limit:
            return f"{key} must be between {low} and {limit}"
        if key in INTEGER_PARAM_KEYS and not (
            isinstance(value, int)
            or (isinstance(value, float) and value.is_integer())
            or (isinstance(value, str) and value.strip().lstrip("-").isdigit())
        ):
            return f"{key} must be an integer"
    for key, limit in STRING_PARAM_MAX_CHARS.items():
        if key in params and params[key] is not None:
            if not isinstance(params[key], str) or len(params[key]) > limit:
                return f"{key} must be a string of at most {limit} characters"
    return None

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
        try:
            body = await _read_body_capped(request, MAX_BODY_BYTES)
        except BodyTooLarge:
            return StarletteJSON(
                {"detail": f"Request body exceeds {MAX_BODY_BYTES} bytes"}, status_code=413,
            )

        if runner.supports_streaming:
            actions = ActionBuffer()
            actions.put(body)
            actions.close()
            output = next(runner.stream(actions), b"")
            return StreamingResponse(iter([output]), media_type="application/octet-stream")

        try:
            params = json.loads(body)
        except ValueError:
            return StarletteJSON({"detail": "Body must be a JSON object"}, status_code=400)
        if not isinstance(params, dict):
            return StarletteJSON({"detail": "Body must be a JSON object"}, status_code=400)
        # Only documented generation parameters reach runner.generate(**params);
        # see GENERATE_PARAM_KEYS. The output always goes to a server-chosen file.
        unsupported = sorted(
            k for k in params if k not in GENERATE_PARAM_KEYS and k not in INVOCATION_IMAGE_KEYS
        )
        if unsupported:
            return StarletteJSON(
                {"detail": f"Unsupported parameters: {', '.join(unsupported)}"},
                status_code=400,
            )
        staged_upload = "image_b64" in params
        problem = invalid_generation_value(params) or _stage_invocation_image(params)
        if problem:
            return StarletteJSON({"detail": problem}, status_code=400)
        params["output_path"] = str(JOB_OUTPUT_DIR / f"{uuid.uuid4().hex}.mp4")
        broadcast_command({"cmd": CMD_GENERATE, "args": params})
        try:
            output = runner.generate(**params)
        finally:
            # A decoded upload is only needed for this one call.
            if staged_upload and params.get("image_path"):
                shutil.rmtree(Path(params["image_path"]).parent, ignore_errors=True)
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
        # Enforce auth before accepting the socket. Non-browser clients send an
        # Authorization header; the browser WebSocket API cannot, so it offers
        # the token as a subprotocol instead (see WS_BEARER_PROTOCOL). A token in
        # the URL is deliberately not accepted.
        offered = ws_offered_protocols(ws.headers.get("sec-websocket-protocol"))
        if config.auth_enabled:
            presented = extract_token(ws.headers.get("authorization"), ws_bearer_token(offered))
            if not token_valid(config, presented):
                await ws.close(code=1008)  # policy violation
                logger.info("Rejected unauthenticated WebSocket connection")
                return

        # A browser that offered subprotocols aborts unless one is selected.
        await ws.accept(subprotocol=WS_BEARER_PROTOCOL if WS_BEARER_PROTOCOL in offered else None)

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


def _principal(request: Request) -> Optional[str]:
    """The authenticated caller, as set by AuthMiddleware (None if it never ran)."""
    return getattr(request.state, "principal", None)


def _register_jobs(app: FastAPI, jobs: JobStore):
    @app.post("/generate")
    async def submit_job(
        request: Request,
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
        problem = invalid_generation_value(params)
        if problem:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(400, problem)
        jobs.submit(job_dir.name, params, owner=_principal(request))
        return JSONResponse({"job_id": job_dir.name, "status": "queued"})

    # Jobs are scoped to the principal that submitted them: another caller's job
    # id resolves to 404, exactly like an id that does not exist.
    @app.get("/jobs/{job_id}")
    @app.get("/status/{job_id}")
    def job_status(job_id: str, request: Request):
        entry = jobs.get_owned(job_id, _principal(request))
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
    def job_output(job_id: str, request: Request):
        entry = jobs.get_owned(job_id, _principal(request))
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
