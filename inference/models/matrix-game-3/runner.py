# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matrix Game 3.0 — real-time interactive world model (Skywork).

Upstream ships a batch script: `generate.py --interactive` prompts on stdin for
one keyboard and one mouse key per chunk, generates every chunk, then writes a
single MP4 at the end. We need the opposite shape — actions arriving from a
browser over a WebSocket, frames leaving as they are produced.

Rather than fork their pipeline, this runner drives it through two seams:

  1. `get_current_action` is a module-level function in their interactive
     pipeline, called once per chunk. We replace it with a closure that reads
     the latest action out of our ActionBuffer, so their loop keeps its own
     control flow and we keep ours.

  2. `vae.stream_decode` is the single point where latents become pixels in
     their synchronous path. We wrap it so each decoded chunk is copied to a
     queue on its way back, then split into frames and JPEG-encoded here.

Their generate() therefore runs on a worker thread while stream() drains the
queue, which is what turns a batch script into a session.

The seams are the single-GPU path only. On multiple GPUs the session runs the
lockstep chunk loop in lockstep.py instead, because upstream's loop reads the
action on rank 0 alone and cannot end a session without stranding the other
ranks at a barrier.

Two deliberate limits for this first cut:

  * Synchronous VAE only. Their `--use_async_vae` path decodes in a separate
    worker and never calls stream_decode on this process, so seam 2 would see
    nothing. Async VAE is a throughput optimisation to revisit once the
    baseline is measured.

  * Actions apply per chunk, not per frame. Upstream generates 57 frames for
    the first chunk and 40 thereafter, so input is sampled at chunk boundaries.
    That is a property of the model, not of this adapter.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import torch

from lib.distributed import broadcast_object
from lib.runner import ActionBuffer, Runner

import actions as action_codec
import frames as frame_codec
import lockstep

logger = logging.getLogger(__name__)

# Upstream's repo is cloned into the image; see this cartridge's Dockerfile.
UPSTREAM_REPO = os.environ.get("MATRIX_GAME_REPO", "/opt/ml/code/Matrix-Game/Matrix-Game-3")

# Their reference run: 704x1280, 3-step distilled schedule, int8 DiT, LightVAE.
# Upstream parses this as HEIGHT*WIDTH (see generate(): height = size.split("*")[0]),
# so "480*832" is landscape 832x480 — a quarter of the pixels of 704x1280 and the
# setting a colleague measured ~20fps with on 8 GPUs.
DEFAULT_SIZE = os.environ.get("MG3_SIZE_DEFAULT", "480*832")
DEFAULT_STEPS = 3
# generate.py defaults: sample_shift 5.0, sample_guide_scale 5.0.
DEFAULT_SHIFT = 5.0
DEFAULT_GUIDE_SCALE = 5.0
DEFAULT_PROMPT = (
    "A colorful, animated cityscape with a gas station and various buildings."
)

# Chunks in flight before we start dropping. A slow client must not be able to
# make the GPU accumulate unbounded decoded video in host memory; dropping the
# oldest chunk keeps the session live at the cost of a visual jump.
MAX_PENDING_CHUNKS = 4

# Upstream's loop is bounded by num_iterations, so an interactive session asks
# for far more chunks than anyone will play through and stops early when the
# client disconnects.
SESSION_ITERATIONS = 100_000

# Multi-GPU sessions must be bounded: memory selection indexes latents from
# session start, so the latent list grows for the life of a session (~150KB per
# latent frame at 480x832, ~1.5MB per chunk). 1800 chunks is roughly half an
# hour of play at one chunk per second, for under 3GB of an 80GB H100.
MAX_SESSION_CHUNKS = int(os.environ.get("MG3_MAX_CHUNKS", "1800"))


def block_broken_flash_attn() -> bool:
    """Make a broken flash-attn look absent, so upstream's own guard fires.

    Adapted from the world-models-infra `rahul-mg3` engine. Upstream's
    wan/modules/attention.py does `try: import flash_attn / except
    ModuleNotFoundError`. A wheel built against a different torch raises
    ImportError instead, which that clause does not catch, so the whole import
    chain dies. Installing a finder that raises ModuleNotFoundError converts the
    failure into the one upstream handles.

    Note this only buys a clean *import*. Matrix Game 3 still cannot generate
    without flash-attn, because wan/modules/model.py calls flash_attention()
    directly and that asserts a backend is available. The shim's value is a clear
    failure at load time rather than a confusing one mid-session.
    """
    import importlib.abc

    try:
        import flash_attn  # noqa: F401
        return False
    except ModuleNotFoundError:
        return False
    except ImportError as exc:
        logger.warning("flash_attn present but unusable (%s); masking it", exc)

    for name in [k for k in sys.modules if k == "flash_attn" or k.startswith("flash_attn.")]:
        del sys.modules[name]

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "flash_attn" or fullname.startswith("flash_attn."):
                raise ModuleNotFoundError(
                    "flash_attn is masked: ABI mismatch with the installed torch"
                )
            return None

    sys.meta_path.insert(0, _Blocker())
    os.environ.setdefault("WAN_FA_VERSION", "0")
    return True


def _importable(module: str) -> bool:
    """True only if the module imports cleanly.

    Deliberately catches Exception, not ImportError: a compiled extension built
    against the wrong torch can fail in several ways, and every one of them
    means "do not ask upstream to use this backend".
    """
    try:
        __import__(module)
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means unusable
        logger.info("%s unusable: %s", module, exc)
        return False


def relocate_broadcast_objects_to_local_device() -> None:
    """Make dist.broadcast_object_list land CUDA tensors on the *local* device.

    Upstream's second multi-GPU bug, and the one that actually breaks 8-GPU runs.
    broadcast_object_list pickles a tensor together with its device, so when rank 0
    sends the action/pose conditions (which it has already moved to cuda:0), every
    follower deserialises tensors tagged cuda:0. Later, deep in the DiT:

        wan/modules/action_module.py:222
        group_mouse = torch.cat([hidden_states, group_mouse], dim=-1)
        RuntimeError: Expected all tensors to be on the same device, but got
        tensors is on cuda:0, different from other tensors on cuda:5

    Rank 0 never sees it because cuda:0 is its own device. Ranks 1..N-1 all die,
    rank 0 finishes its loop alone and blocks forever on the final dist.barrier()
    — which is the BROADCAST-vs-ALLREDUCE deadlock we spent a while chasing.

    Reproduced with upstream's own generate.py --interactive --ulysses_size 8, with
    no code of ours in the process, so it is theirs and not an artefact of this
    cartridge.

    Wrapping the collective here rather than sed-ing their source keeps the fix
    working across upstream revisions. Only CUDA tensors on the wrong device are
    touched; CPU tensors and plain objects are passed through untouched, so our own
    gloo command channel is unaffected.
    """
    import torch.distributed as dist

    if getattr(dist.broadcast_object_list, "_relocates_to_local_device", False):
        return

    original = dist.broadcast_object_list

    def _relocate(obj, target):
        if isinstance(obj, torch.Tensor):
            # Only CUDA tensors that landed on another GPU need moving.
            if obj.is_cuda and obj.device != target:
                return obj.to(target)
            return obj
        if isinstance(obj, list):
            return [_relocate(o, target) for o in obj]
        if isinstance(obj, tuple):
            return tuple(_relocate(o, target) for o in obj)
        if isinstance(obj, dict):
            return {k: _relocate(v, target) for k, v in obj.items()}
        return obj

    def wrapper(object_list, src=0, group=None, device=None):
        result = original(object_list, src=src, group=group, device=device)
        if torch.cuda.is_available():
            target = torch.device(f"cuda:{torch.cuda.current_device()}")
            for i, obj in enumerate(object_list):
                object_list[i] = _relocate(obj, target)
        return result

    wrapper._relocates_to_local_device = True
    dist.broadcast_object_list = wrapper
    logger.info("broadcast_object_list patched to relocate CUDA tensors locally")


class _EndOfSession(Exception):
    """Raised inside the patched action hook to unwind upstream's loop."""


class MatrixGameRunner(Runner):
    def __init__(self) -> None:
        self.pipe = None
        self.device = None
        self.rank = 0
        self.world_size = 1
        self._args = None
        self._text_cond = None
        self._chunks: queue.Queue = queue.Queue()
        self._error: BaseException | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ setup

    def setup(self, model_dir, device, rank, world_size) -> None:
        self.device = device
        self.rank = rank
        self.world_size = world_size

        if UPSTREAM_REPO not in sys.path:
            sys.path.insert(0, UPSTREAM_REPO)

        block_broken_flash_attn()

        use_sp = world_size > 1
        if use_sp:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend="nccl", init_method="env://", rank=rank, world_size=world_size
                )
            torch.cuda.set_device(device)

        self._args = self._build_args(model_dir)

        if use_sp:
            # Upstream sends conditions as pickled cuda:0 tensors; relocate them
            # before anything can cat them against local activations.
            relocate_broadcast_objects_to_local_device()

            # THE step that actually enables 8-GPU inference. generate.py calls
            # this whenever ulysses_size > 1, and it builds upstream's Ulysses
            # sequence-parallel groups. Without it the ranks all start, the model
            # loads on each of them, and then rank 0 does every bit of the work
            # alone while ranks 1..N-1 sit at 0% — a chunk took ~70s instead of
            # streaming.
            from wan.distributed.util import init_distributed_group  # noqa: E402

            init_distributed_group()
            logger.info("Ulysses sequence parallelism initialised across %d GPUs", world_size)

        from pipeline.inference_interactive_pipeline import (  # noqa: E402
            MatrixGame3Pipeline as InteractivePipeline,
        )
        from wan.configs import WAN_CONFIGS  # noqa: E402

        logger.info(
            "loading Matrix Game 3.0 from %s (rank %d/%d, fa_version=%s)",
            model_dir, rank, world_size, self._args.fa_version,
        )
        # Mirror generate.py's construction exactly. Guessing this signature is
        # what broke the first bring-up: it takes `config` and `checkpoint_dir`,
        # not `ckpt_dir`, and it needs the args namespace and fa_version too.
        self.pipe = InteractivePipeline(
            config=WAN_CONFIGS["matrix_game3"],
            checkpoint_dir=str(model_dir),
            device_id=device.index if getattr(device, "index", None) is not None else 0,
            rank=rank,
            t5_fsdp=self._args.t5_fsdp,
            dit_fsdp=self._args.dit_fsdp,
            use_sp=self._args.ulysses_size > 1,
            t5_cpu=self._args.t5_cpu,
            # Keep the DiT off CPU when sequence-parallel, per generate.py.
            init_on_cpu=self._args.ulysses_size <= 1,
            convert_model_dtype=self._args.convert_model_dtype,
            # Left at the constructor default. Upstream derives camera poses from
            # the action stream (compute_all_poses_from_actions), so mouse-look
            # works without it, and turning it on changed the conditioning path
            # for no observed benefit.
            args=self._args,
            fa_version=self._args.fa_version,
            use_base_model=self._args.use_base_model,
        )

        if use_sp:
            # The prompt is fixed per deployment, so encode it once at boot.
            # T5 runs on the CPU (t5_cpu=True) and costs seconds per encode;
            # paying that here moves it out of every session's first frame.
            # Same call upstream's generate() makes, identical on every rank.
            self._text_cond = self.pipe.text_encoder(
                [os.environ.get("MG3_PROMPT", DEFAULT_PROMPT)], device=self.device
            )
            logger.info("text conditions cached for lockstep sessions")

    def _build_args(self, model_dir) -> SimpleNamespace:
        """Mirror upstream's argparse namespace, which their pipeline reads directly."""
        out_dir = os.environ.get("MG3_OUTPUT_DIR", "/tmp/mg3-output")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            ckpt_dir=str(model_dir),
            size=os.environ.get("MG3_SIZE", DEFAULT_SIZE),
            num_inference_steps=int(os.environ.get("MG3_STEPS", DEFAULT_STEPS)),
            fa_version=self._pick_flash_attention(),
            use_int8=os.environ.get("MG3_INT8", "1") == "1",
            # ON by default, and not optional in practice: with compile_vae off,
            # upstream takes a different decode path that never calls
            # vae.stream_decode, so the seam this runner hooks to collect frames
            # sees nothing and a session streams zero frames while looking
            # perfectly healthy. Measured 44 fps with it on, nothing at all with
            # it off. The first chunk pays a JIT cost; that is the trade.
            compile_vae=os.environ.get("MG3_COMPILE_VAE", "1") == "1",
            vae_type=os.environ.get("MG3_VAE_TYPE", "mg_lightvae"),
            lightvae_pruning_rate=float(os.environ.get("MG3_VAE_PRUNE", "0.5")),
            # Async VAE bypasses our decode seam — see the module docstring.
            use_async_vae=False,
            async_vae_warmup_iters=0,
            ulysses_size=self.world_size,
            # FSDP off, Ulysses on. The working engine in world-models-infra
            # (rahul-mg3) runs 8 GPUs with both FSDP flags False and relies on
            # sequence parallelism alone; enabling FSDP as well shards a 5B model
            # that already fits, for no gain.
            dit_fsdp=False,
            t5_fsdp=False,
            # T5 lives on CPU and moves to GPU on demand — it is used once per
            # chunk for the prompt and would otherwise hold VRAM permanently.
            t5_cpu=True,
            # bf16, not fp32. With this off, per-GPU memory went from 6.8GB to
            # 23GB — a 5B model in fp32 — and generation slowed enough that no
            # frame arrived inside 240s.
            convert_model_dtype=True,
            use_base_model=False,
            verify_quant=False,
            interactive=True,
            seed=int(os.environ.get("MG3_SEED", "42")),
            # Upstream's interactive pipeline also reads these two off args when
            # it writes its own MP4 at the end of a run. We stream frames instead
            # and never use the file, but they have to exist or generate() dies
            # with AttributeError mid-session. /tmp is the only writable path for
            # the non-root container user.
            output_dir=out_dir,
            save_name="session",
            # Upstream's pipeline reads these off args in places; take the shift
            # from the model config rather than inventing one.
            sample_shift=self._config_shift(),
            sample_guide_scale=DEFAULT_GUIDE_SCALE,
            prompt=os.environ.get("MG3_PROMPT", DEFAULT_PROMPT),
            image=None,
        )

    @staticmethod
    def _config_shift() -> float:
        """Noise-schedule shift from the model config, with a documented default."""
        try:
            from wan.configs import WAN_CONFIGS

            return float(WAN_CONFIGS["matrix_game3"].sample_shift)
        except Exception:
            return DEFAULT_SHIFT

    def _max_area(self) -> int:
        """Pixel budget for the current size.

        Upstream's MAX_AREA_CONFIGS only lists 704*1280, so deriving the area
        from the size string is what makes MG3_SIZE do anything at all.
        """
        height, _, width = self._args.size.partition("*")
        return int(height) * int(width)

    @staticmethod
    def _pick_flash_attention() -> str:
        """Pick the attention backend upstream should use: '3', '2' or '0' (SDPA).

        GPU capability alone is not enough. A flash-attn built against a
        different torch imports with an ImportError rather than a
        ModuleNotFoundError, and upstream only catches the latter — so a broken
        wheel kills the server instead of degrading it. Probe the import here and
        ask for SDPA unless flash-attn genuinely loads.
        """
        override = os.environ.get("MG3_FA_VERSION")
        if override:
            return override
        if not torch.cuda.is_available():
            return "0"

        major, _ = torch.cuda.get_device_capability()
        if major >= 9 and _importable("flash_attn_interface"):
            return "3"  # Hopper and newer, FA3 package present
        if major >= 8 and _importable("flash_attn"):
            return "2"
        logger.info("flash-attention unavailable; falling back to SDPA")
        return "0"

    # ------------------------------------------------------------------ async

    def generate(self, **params):
        """Matrix Game 3.0 is a session model; there is no one-shot job form."""
        raise NotImplementedError(
            "matrix-game-3 is real-time only — open a WebSocket session at /ws "
            "instead of POSTing a job"
        )

    @property
    def supports_streaming(self) -> bool:
        return True

    def reset(self) -> None:
        """Drop any state left by a previous session."""
        self._stop.clear()
        self._error = None
        while not self._chunks.empty():
            try:
                self._chunks.get_nowait()
            except queue.Empty:
                break

    # ----------------------------------------------------------------- stream

    @torch.no_grad()
    def stream(self, client_actions: ActionBuffer) -> Iterator[bytes]:
        """Run one interactive session, yielding JPEG frames as they decode.

        Single GPU drives upstream's own generate() through the two seams
        below. Multi-GPU takes the lockstep path instead: upstream's loop reads
        the action only on rank 0 and lets rank 0 unwind alone on disconnect,
        which strands ranks 1..N-1 at a barrier — see lockstep.py.
        """
        if self.pipe is None:
            raise RuntimeError("setup() must run before stream()")

        if self.world_size > 1:
            yield from self._stream_lockstep(client_actions)
            return

        self.reset()

        with self._patched_action_source(client_actions), self._patched_decoder():
            worker = threading.Thread(
                target=self._run_upstream, name="mg3-generate", daemon=True
            )
            worker.start()

            if self.rank != 0:
                worker.join()
                self._reraise()
                return

            frame_index = 0
            while True:
                if not client_actions.active and self._chunks.empty():
                    break
                try:
                    chunk = self._chunks.get(timeout=0.25)
                except queue.Empty:
                    if not worker.is_alive():
                        break
                    continue

                if chunk is None:  # worker finished
                    break

                for frame in frame_codec.split_chunk(chunk):
                    frame_index += 1
                    yield frame_codec.encode_jpeg(frame)

            self._stop.set()
            client_actions.close()
            worker.join(timeout=30)
            self._reraise()
            logger.info("session ended after %d frames", frame_index)

    @torch.no_grad()
    def _stream_lockstep(self, client_actions: ActionBuffer) -> Iterator[bytes]:
        """Multi-GPU session: every rank runs one identical chunk loop.

        Rank 0 reads the newest client action and broadcasts it — a dozen CPU
        floats on the gloo command channel — then all ranks compute the chunk
        together. The stop flag rides the same message, so the ranks always
        leave together; there is no path where rank 0 exits and the others
        keep waiting on a collective.
        """
        self.reset()
        session = lockstep.LockstepSession(
            pipe=self.pipe,
            cfg=self._wan_config(),
            args=self._args,
            device=self.device,
            rank=self.rank,
            anchor_image=self._anchor_image(),
            text_cond=self._text_cond,
        )

        frame_index = 0
        stopped = False
        try:
            while True:
                if self.rank == 0:
                    payload = client_actions.get()
                    if (
                        payload is None
                        or not client_actions.active
                        or session.chunk_idx >= MAX_SESSION_CHUNKS
                    ):
                        msg = {"stop": True}
                    else:
                        keyboard, mouse = action_codec.decode(payload)
                        msg = {"stop": False, "keyboard": keyboard, "mouse": mouse}
                else:
                    msg = None
                msg = broadcast_object(msg)
                if not isinstance(msg, dict) or msg.get("stop", True):
                    stopped = True
                    break

                frames = session.run_chunk(msg["keyboard"], msg["mouse"])
                if self.rank == 0 and frames is not None:
                    # Later chunks re-decode the 16-frame overlap with the
                    # previous window; emit only frames the client hasn't seen.
                    fresh = lockstep.new_frames_in_chunk(session.chunk_idx - 1)
                    for frame in frames[-fresh:]:
                        frame_index += 1
                        yield frame_codec.encode_jpeg(frame)
        finally:
            # Reached on clean stop, on error, and on GeneratorExit when the
            # client vanishes mid-yield. Without this, followers would sit in
            # broadcast_object forever and swallow the NEXT session's command.
            if self.rank == 0 and not stopped:
                try:
                    broadcast_object({"stop": True})
                except Exception:
                    logger.exception("failed to broadcast session stop")
            if self.rank == 0:
                logger.info("lockstep session ended after %d frames", frame_index)

    def _wan_config(self):
        from wan.configs import WAN_CONFIGS

        return WAN_CONFIGS["matrix_game3"]

    def _run_upstream(self) -> None:
        """Drive upstream's generate() until the client goes away."""
        try:
            from PIL import Image

            anchor = self._anchor_image()
            if not isinstance(anchor, Image.Image):
                raise RuntimeError("anchor image did not load as a PIL image")

            # generate() is positional for prompt and image (see generate.py) and
            # reads the rest off the args namespace, including num_iterations,
            # which is how we ask for a session rather than a fixed-length clip.
            session_args = SimpleNamespace(
                **vars(self._args), num_iterations=SESSION_ITERATIONS
            )
            self.pipe.generate(
                os.environ.get("MG3_PROMPT", DEFAULT_PROMPT),
                anchor,
                max_area=self._max_area(),
                shift=self._config_shift(),
                num_inference_steps=self._args.num_inference_steps,
                guide_scale=DEFAULT_GUIDE_SCALE,
                seed=self._args.seed,
                use_base_model=self._args.use_base_model,
                args=session_args,
            )
        except _EndOfSession:
            logger.info("upstream loop unwound on client disconnect")
        except BaseException as exc:  # surfaced to the client by _reraise
            logger.exception("matrix-game-3 generation failed")
            self._error = exc
        finally:
            self._chunks.put(None)

    def _anchor_image(self):
        """The scene the session starts from.

        Upstream needs an anchor frame. We use one of their bundled demo images
        so a session can start with no client-supplied scene; a future revision
        should accept the anchor over the socket.
        """
        from PIL import Image

        override = os.environ.get("MG3_ANCHOR_IMAGE")
        candidates = [Path(override)] if override else []
        candidates.append(Path(UPSTREAM_REPO) / "demo_images" / "001" / "image.png")
        for path in candidates:
            if path.is_file():
                return Image.open(path).convert("RGB")
        raise FileNotFoundError(
            "no anchor image found; set MG3_ANCHOR_IMAGE to a readable file"
        )

    def _reraise(self) -> None:
        if self._error is not None:
            error, self._error = self._error, None
            raise error

    # ------------------------------------------------------------------ seams

    def _patched_action_source(self, client_actions: ActionBuffer):
        """Replace upstream's stdin prompt with our ActionBuffer."""
        from pipeline import inference_interactive_pipeline as upstream

        original = upstream.get_current_action

        def from_client():
            if self._stop.is_set():
                raise _EndOfSession
            payload = client_actions.get()
            if payload is None:
                raise _EndOfSession
            keyboard, mouse = action_codec.decode(payload)
            return {
                "keyboard": torch.tensor(keyboard),
                "mouse": torch.tensor(mouse),
            }

        return _Patch(upstream, "get_current_action", from_client, original)

    def _patched_decoder(self):
        """Copy every decoded chunk to our queue on its way back to upstream."""
        vae = getattr(self.pipe, "vae", None)
        if vae is None or not hasattr(vae, "stream_decode"):
            # Nothing to hook: let the session run and fail loudly rather than
            # silently streaming nothing.
            logger.warning("vae.stream_decode not found — no frames will stream")
            return _NullPatch()

        original = vae.stream_decode

        def decode_and_tee(*args, **kwargs):
            result = original(*args, **kwargs)
            if self.rank == 0:
                video = result[0] if isinstance(result, tuple) else result
                if self._chunks.qsize() >= MAX_PENDING_CHUNKS:
                    try:
                        self._chunks.get_nowait()  # drop oldest
                    except queue.Empty:
                        pass
                self._chunks.put(video)
            return result

        return _Patch(vae, "stream_decode", decode_and_tee, original)


class _Patch:
    """Context manager that swaps an attribute and always puts it back."""

    def __init__(self, target, name, replacement, original):
        self._target = target
        self._name = name
        self._replacement = replacement
        self._original = original

    def __enter__(self):
        setattr(self._target, self._name, self._replacement)
        return self

    def __exit__(self, *exc_info):
        setattr(self._target, self._name, self._original)
        return False


class _NullPatch:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def create_runner():
    return MatrixGameRunner()
