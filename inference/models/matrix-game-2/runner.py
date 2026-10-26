# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matrix Game 2 cartridge — Skywork's real-time interactive world model.

Single-GPU by construction: the 1.8B causal DiT, VAE and CLIP encoder fit on
one card with room to spare, and the model has no sequence-parallel path, so
extra GPUs cannot speed it up. The fastest single GPU is the whole lever — it
does 10-15 fps on an L40S and 17-20 fps on an H100 at 640x352.

Threading: every engine call runs on one dedicated device thread, because the
compiled VAE decoder belongs to the thread that compiled it. The shared server
calls reset() on the asyncio side and stream() on an executor thread; both
trampoline onto the device thread. While the GPU generates chunk N+1, stream()
JPEG-encodes chunk N and paces its frames out over that time (frames.Pacer).

Runtime knobs (container env):
  MG2_REPO           path to the upstream Matrix-Game-2 checkout
                     (default /opt/ml/code/Matrix-Game/Matrix-Game-2)
  MG2_SEED           seed image used when a client does not send one
                     (default: upstream demo_images/universal/0009.png)
  MG2_MAX_LATENTS    conditioning length before an automatic re-seed (360 ≈ 80 s)
  MG2_WARMUP         1 to compile and fill caches at startup (default 1)
  MG2_WARMUP_SESSIONS / MG2_WARMUP_CHUNKS
                     how much warmup runs (default 2 x 10); the decoder keeps
                     recompiling for several chunks, so fewer leaves the first
                     player waiting
  MG2_JPEG_QUALITY   JPEG quality for streamed frames (default 80)
  MG2_SEED_WAIT_S    how long a new session waits for a client seed before
                     starting from the default one (default 0.5)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

import torch

from lib.runner import ActionBuffer, Runner

import actions as mg2_actions
import frames as mg2_frames

logger = logging.getLogger(__name__)

DEFAULT_REPO = "/opt/ml/code/Matrix-Game/Matrix-Game-2"


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        value = float("nan")
    if value != value or value in (float("inf"), float("-inf")):
        logger.warning("ignoring %s=%r; using %s", name, os.environ.get(name), default)
        return default
    return value


def _env_int(name: str, default: int, minimum: int) -> int:
    value = max(minimum, int(_env_float(name, default)))
    return value


class MatrixGame2Runner(Runner):
    engine = None

    def setup(self, ckpt_dir: str, device: torch.device, rank: int, world_size: int) -> None:
        self.device = device
        configured = os.environ.get("MG2_JPEG_QUALITY")
        self.jpeg_quality = mg2_frames.resolve_quality(configured)
        if configured and configured.strip() != str(self.jpeg_quality):
            logger.warning("ignoring MG2_JPEG_QUALITY=%r; using %d", configured, self.jpeg_quality)
        self.seed_wait = _env_float("MG2_SEED_WAIT_S", 0.5)

        if world_size > 1 and rank != 0:
            # The image pins NPROC_PER_NODE=1, so this should not happen; if it
            # does, followers idle in the command loop holding nothing.
            logger.warning("matrix-game-2 is single-GPU; rank %d loads nothing", rank)
            return

        self._dev = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mg2-device")
        repo = os.environ.get("MG2_REPO", DEFAULT_REPO)
        # At least 3 chunks, or every step would re-seed.
        max_latents = _env_int("MG2_MAX_LATENTS", 360, minimum=9)

        def _load():
            from engine import MatrixGame2Engine

            return MatrixGame2Engine(ckpt_dir, repo, device, max_latents=max_latents)

        self.engine = self._dev.submit(_load).result()
        self._default_seed = self._load_default_seed()

        if os.environ.get("MG2_WARMUP", "1") == "1":
            # torch.compile max-autotune takes minutes on the first decode. Pay
            # it here, before /ping goes green, rather than on the first player.
            t0 = time.perf_counter()

            # A few chunks are not enough: the decoder's causal caches change
            # stride chunk to chunk, and each new stride is another compile.
            # Measured on an L40S, a 3-chunk warmup left the first player
            # waiting >60 s for frames while it recompiled. Run sessions until
            # those compiles are spent, as the booth's warm service did.
            sessions = _env_int("MG2_WARMUP_SESSIONS", 2, minimum=1)
            chunks = _env_int("MG2_WARMUP_CHUNKS", 10, minimum=1)

            def _warm():
                for s in range(sessions):
                    self.engine.start(self._default_seed)
                    for i in range(chunks):
                        self.engine.step((1, 0, 0, 0), (-1, 0, 1)[(s + i) % 3])
                    self.engine.end()

            self._dev.submit(_warm).result()
            logger.info("matrix-game-2 warm in %.0f s", time.perf_counter() - t0)

    def _load_default_seed(self):
        from PIL import Image

        path = os.environ.get("MG2_SEED")
        if path:
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                logger.exception("could not read MG2_SEED=%s; using upstream default", path)
        return self.engine.default_seed()

    @staticmethod
    def _image_from_b64(b64: str):
        from PIL import Image

        try:
            return Image.open(io.BytesIO(base64.b64decode(b64, validate=False))).convert("RGB")
        except Exception:
            logger.warning("bad client seed image; keeping the current world")
            return None

    # ------------------------------------------------------------- lifecycle

    def generate(self, **params) -> str:
        raise NotImplementedError("matrix-game-2 is a real-time cartridge; connect to /ws")

    @property
    def supports_streaming(self) -> bool:
        return True

    def reset(self) -> None:
        """End the previous world. The next one starts on the first step, once
        the client has had a moment to send its own seed image."""
        if self.engine is None:
            return
        self._dev.submit(self.engine.end).result()

    # --------------------------------------------------------------- stream

    def stream(self, actions: ActionBuffer):
        if self.engine is None:
            return

        state = {"ctrl": mg2_actions.NEUTRAL}
        got_input = Event()
        closed = Event()

        def pump():
            # ActionBuffer.get() re-returns the latest payload every <=0.1 s even
            # when nothing new arrived; act only when it changes. Seeds do not
            # come through here at all: the server parks `start` in its own
            # slot (take_seed) so a key tick right behind it cannot drop it.
            last = None
            try:
                while True:
                    payload = actions.get()
                    if payload is None:
                        if not actions.active:
                            break
                        continue
                    if payload == last:
                        continue
                    last = payload
                    ctrl = mg2_actions.decode(payload)
                    if ctrl is not None and not ctrl.is_seed:
                        state["ctrl"] = ctrl
                    got_input.set()
            except Exception:  # noqa: BLE001 — never let the pump die silently
                logger.exception("action pump failed; ending the session")
            finally:
                closed.set()
                got_input.set()

        Thread(target=pump, name="mg2-actions", daemon=True).start()

        # A client that seeds the world sends `start` right after it connects.
        # Waiting a moment for it saves encoding the default world only to
        # throw it away.
        got_input.wait(self.seed_wait)

        engine = self.engine
        reseeds_before = engine.reseeds

        def step():
            if closed.is_set():
                return [], 0.0
            seed = actions.take_seed()
            if seed is not None:
                ctrl = mg2_actions.decode(seed)
                image = self._image_from_b64(ctrl.seed_b64) if ctrl is not None and ctrl.is_seed else None
                if image is not None:
                    engine.start(image)
            if not engine.started:
                engine.start(self._default_seed)
            ctrl = state["ctrl"]
            t0 = time.perf_counter()
            out = engine.step(ctrl.keyboard, ctrl.yaw)
            return out, time.perf_counter() - t0

        pacer = mg2_frames.Pacer()
        pending = self._dev.submit(step)
        n_frames = 0
        t_start = time.perf_counter()
        try:
            while not closed.is_set():
                try:
                    frames, chunk_s = pending.result()
                except Exception:
                    # End this session cleanly (a normal close, the session gate
                    # released) rather than letting the error escape into the
                    # server's socket loop; the next player gets a fresh world.
                    logger.exception("generation step failed; ending the session")
                    return
                pending = self._dev.submit(step)
                interval = pacer.observe(chunk_s, len(frames))
                next_at = time.perf_counter()
                for jpeg in mg2_frames.encode_batch(frames, self.jpeg_quality):
                    if closed.is_set():
                        break
                    delay = next_at - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    yield jpeg
                    n_frames += 1
                    next_at += interval
        finally:
            closed.set()
            try:
                pending.result(timeout=30)
            except Exception:
                pass
            elapsed = time.perf_counter() - t_start
            if n_frames:
                logger.info("session ended: %d frames, %.1f fps, %d re-seeds",
                            n_frames, n_frames / max(elapsed, 1e-6), engine.reseeds - reseeds_before)


def create_runner() -> Runner:
    """Required factory — the shared server imports this by name."""
    return MatrixGame2Runner()
