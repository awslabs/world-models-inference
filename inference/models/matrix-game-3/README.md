# Matrix Game 3.0

Real-time interactive world generation: send a keyboard and mouse action, get the
next frames back. Upstream is [Matrix-Game-3.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-3)
from Skywork AI, served here through the shared real-time (WebSocket) path.

This is the catalogue's first *generative* real-time cartridge. V-JEPA 2 also runs
on the real-time path, but it produces embeddings rather than frames.

## At a glance

- **Upstream repo:** https://github.com/SkyworkAI/Matrix-Game (`Matrix-Game-3/`)
- **Weights:** `Skywork/Matrix-Game-3.0` (~57 GB — distilled 5B DiT, T5 encoder, VAE)
- **Licence:** Apache 2.0, code and weights
- **Compute:** `p5.48xlarge` (8× H100 80GB), matching upstream's reference run
- **Resolution:** 832×480 by default, 3-step distilled schedule, int8 DiT, LightVAE
- **Throughput:** measured through the deployed endpoint (ALB, remote client) at
  832×480: **~35 fps sustained on 8 GPUs** with every inter-frame gap under 1s.
  A single GPU reaches ~44 fps inside a chunk but freezes ~7s between chunks, so
  8 GPUs is what makes it playable rather than merely fast in bursts.
- **Cost:** ~$55.04/hr on-demand in us-east-1. Use a Capacity Block for `p5`.

## Status

Validated on a real `p5.48xlarge` deployment on both one and eight GPUs. The
first frame of the first session costs extra (~15s: session init plus the int8
and VAE JIT warm-up); later chunks stream steadily. Offline pieces are covered
by `tests/test_matrix_game_3.py`.

## What you invoke

```
GET  /ping                      health
WS   /ws                        interactive session (EC2)
WS   /invocations-bidirectional-stream   interactive session (SageMaker)
```

The socket needs the same bearer token as every other endpoint, passed either as
an `Authorization` header or a `token` query parameter.

### Session protocol

Client sends one message per tick, in any of three forms:

| Form | Example | When to use |
| --- | --- | --- |
| JSON keys | `{"keys": "wl"}` | simplest; WASD to move, IJKL for the camera |
| JSON vectors | `{"keyboard": [1,0,0,0,0,0], "mouse": [0.1, 0.0]}` | driving the model directly |
| Binary | `<uint32 key bitflags><float32 pitch><float32 yaw>` | lowest overhead in a game loop |

Server sends one binary message per frame, each a bare JPEG. That is what the
catalogue UI expects (`frontend/src/hooks/useFrameBuffer.ts` renders the Blob
straight to a canvas), so no JSON envelope is added.

Camera deltas are clamped to one 0.1 step per axis per tick. Without that a
client can teleport the camera and break the pose continuity the model's scene
memory depends on.

### Input granularity

Actions apply **per chunk, not per frame**. Upstream generates 57 frames for the
first chunk and 40 for each one after, so input is sampled at chunk boundaries
and you steer the world several times a second rather than every frame. This is
a property of the model, not of the serving layer.

## How it works

Upstream ships a batch script — `generate.py --interactive` prompts on stdin for
one action per chunk, generates them all, then writes a single MP4. On a single
GPU the runner drives that same pipeline through two seams instead of forking it:

1. **`get_current_action`** — a module-level function in their interactive
   pipeline, replaced with a closure that reads our `ActionBuffer`.
2. **`vae.stream_decode`** — the one place latents become pixels on the
   synchronous path, wrapped so each decoded chunk is copied to a queue on its
   way back, then split into frames and JPEG-encoded.

Their `generate()` runs on a worker thread while `stream()` drains the queue.

On multiple GPUs the session runs the lockstep chunk loop in `lockstep.py`
instead. Upstream's own loop reads the action only on rank 0 and lets rank 0
unwind alone when a session ends, which strands the other ranks at a barrier;
the lockstep loop broadcasts the action (and the stop decision) as a few CPU
floats at the top of every chunk, so all ranks always advance and exit
together. See the module docstring for the full reasoning.

Two deliberate limits in this first cut:

- **Synchronous VAE only.** `--use_async_vae` decodes in a separate worker and
  never calls `stream_decode` in this process, so the decode seam would see
  nothing. It is a throughput optimisation to revisit after the baseline.
- **Fixed anchor scene.** A session starts from one of upstream's bundled demo
  images. Accepting a client-supplied anchor over the socket is the next step.

## Environment overrides

The runner reads these so a deployment can be retuned without a rebuild:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MG3_SIZE` | `480*832` | output resolution, HEIGHT*WIDTH |
| `MG3_STEPS` | `3` | denoising steps (distilled schedule) |
| `MG3_COMPILE_VAE` | `1` | torch.compile the VAE decoder; required for the single-GPU decode seam |
| `MG3_MAX_CHUNKS` | `1800` | multi-GPU session length bound (latent memory grows per chunk) |
| `MG3_FA_VERSION` | auto | force FlashAttention `3`, `2` or `0` (SDPA) |
| `MG3_INT8` | `1` | int8 quantised DiT |
| `MG3_VAE_TYPE` | `mg_lightvae` | `mg_lightvae_v2` is faster |
| `MG3_VAE_PRUNE` | `0.5` | LightVAE pruning rate |
| `MG3_PROMPT` | cityscape | scene description |
| `MG3_ANCHOR_IMAGE` | bundled demo | starting frame |
| `MG3_SEED` | `42` | generation seed |

FlashAttention 3 is Hopper-only. The runner picks 3 on H100 and newer, 2 on
Ampere, and SDPA otherwise, so the cartridge boots on cheaper hardware for a
smoke test instead of failing outright.

## Hardware notes

Upstream's `test.sh` runs 8 GPUs (7 when async VAE is on) with `ulysses_size`
matched to the GPU count. The previous version of this manifest targeted
`g5.2xlarge` on EC2, a single Ampere A10G, which cannot run this model: FA3 is
unavailable and 24 GB is far short. The manifest now targets `p5.48xlarge` on
both deploy paths.
