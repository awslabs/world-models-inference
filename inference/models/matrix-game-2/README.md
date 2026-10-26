# Matrix Game 2.0

Real-time interactive world generation on one GPU: seed a world with an image, hold
WASD to walk and turn the camera, and get JPEG frames back over a WebSocket. Upstream
is [Matrix-Game-2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2)
from Skywork AI.

## At a glance

- **Upstream repo:** https://github.com/SkyworkAI/Matrix-Game (`Matrix-Game-2/`),
  pinned to `71c3cd7`
- **Weights:** `Skywork/Matrix-Game-2.0` — ~20 GB repo, ~11 GB read (distilled
  universal DiT, Wan 2.1 VAE, CLIP encoder and tokenizer)
- **Licence:** MIT, code and weights
- **Compute:** `g6e.4xlarge` (1× L40S 48GB), on-demand. Single-GPU by construction —
  the model has no sequence-parallel path, so more GPUs cannot help it
- **Resolution:** 640×352, 3-step distilled schedule, 12 frames per chunk
- **Throughput:** **14.5 fps at 640×352 on one L40S, first frame in 1.9 s, 147 ms input
  lag** — this image on a `g6e.4xlarge`, measured through the endpoint (2026-10-04).
  The same engine ran 17-20 fps on one H100 at the AWS Summit Dubai booth
- **Cost:** ~$3.00/hr on-demand (`g6e.4xlarge`)

## Status

Validated end to end on a `g6e.4xlarge` on 2026-10-04: image built in CodeBuild,
weights staged, deployed with `./deploy.sh`, played from the catalogue UI over the
Cognito-authenticated proxy, and measured with `scripts/benchmark.py` over 60 s:

```
frame rate        14.5 fps at 640x352
first frame       1.9 s   (first session after boot)
input lag p50     147 ms
bitrate           4.0 Mbit/s, 33 KiB/frame
```

Not yet measured with this image on an H100 (`p5.4xlarge` had no on-demand capacity
in us-west-2 that day). The SageMaker path (`ml.g6e.2xlarge`) has not been deployed,
so the catalogue lists EC2 only. The adapter is also covered offline by
`tests/test_matrix_game_2.py` and `tests/test_matrix_game_2_session.py`.

## Deploy

```bash
python scripts/stage-weights.py matrix-game-2 --source hf:Skywork/Matrix-Game-2.0
./deploy.sh matrix-game-2                              # g6e.4xlarge, 1x L40S
INSTANCE_TYPE=p5.4xlarge ./deploy.sh matrix-game-2     # 1x H100, faster
```

Cold start takes about 12 minutes on a fresh instance: ~3-4 minutes of launch, image
pull and weight sync, then ~8 minutes of warmup (490 s measured on an L40S). The VAE decoder is `torch.compile`d with max-autotune, and its
causal caches change stride from chunk to chunk, so it recompiles until dynamo's
recompile limit. The runner spends those compiles during startup — two warm sessions
of ten chunks — so `/ping` only turns healthy once the first player gets a frame in
about 2 s.

## What you invoke

```
GET  /ping                                health
WS   /ws                                  interactive session (EC2)
WS   /invocations-bidirectional-stream    interactive session (SageMaker — untested)
```

The socket needs a Cognito access token, as an `Authorization` header or, from a
browser, the `world-model.bearer` subprotocol — see
[docs/SECURITY.md](../../../docs/SECURITY.md#authenticating).

### Session protocol

Client messages, any mix of:

| Form | Example | Notes |
| --- | --- | --- |
| Binary | `<uint32 wsad bitflags><float32 pitch><float32 yaw>` | what the catalogue player sends; any non-zero yaw turns |
| JSON control | `{"type":"control","buttons":["W","L"],"mouse_dx":100}` | booth kiosk / gamepad; stick yaw only turns past ±30 of 100 |
| Seed | `{"type":"start","image_data":"<base64 or data: URL>"}` | starts a new world from the image |
| JSON keys | `{"keys":"wl"}` | WASD moves, J/L turns |

Server sends one binary message per frame, each a bare JPEG.

The server keeps the last action, so a client must send an explicit release
(empty `buttons`, or a zero binary action) when the last key goes up, or the world
keeps walking. Pings and acks never count as a release.

### What the model can and cannot do

- **Turning is ternary.** The model was trained on one yaw magnitude, so input maps
  to left / none / right — there is no proportional turn.
- **No vertical look.** Pitch is forced to zero. The model handles it badly, and
  the Paris booth found it better disabled than shown misbehaving.
- **Input applies per chunk** (~0.8 s on an L40S, ~0.7 s on an H100), not per frame.
- **A sub-second pause about every 80 s.** A session has 360 latent frames of
  conditioning. When they run out the engine re-seeds from the last frame it made
  and carries on. Before that fix the world froze outright at ~80 s.

## How it works

`engine.py` is the booth engine reduced to a call-per-chunk API: `start(image)`
encodes the seed, `step(keys, yaw)` denoises the next 3 latents and decodes 12
frames. Two of its booth fixes matter for latency:

1. **Seed encode in ~1.2 s, not 13.5 s.** Upstream encodes the seed plus 1,437
   all-zero padding frames through the causal VAE. Those latents go constant a few
   frames past the seed (exact over the prefix and flat to one bf16 ulp beyond,
   measured on an H100), so the engine encodes a 32-latent prefix and repeats the
   last one.
2. **Conditioning cache.** Seeds are keyed by content, so a revisited world starts
   in ~0.7 s.

`runner.py` runs every engine call on one device thread (the compiled decoder
belongs to the thread that compiled it). While the GPU makes chunk N+1, `stream()`
encodes chunk N and spaces its frames over the time the next chunk takes. Sent as
a burst, most of a chunk's 12 frames would be dropped by the catalogue player, which
only keeps the newest frame it has not decoded yet.

One session at a time is the shared server's session gate: a new connection
preempts the old one. The booth server needed its own fix for this (a closing tab
killed the session on screen); here it is the framework's job.

## Tuning knobs

| Env var | Default | Effect |
| --- | --- | --- |
| `MG2_SEED` | upstream `demo_images/universal/0009.png` | world used when the client sends no seed |
| `MG2_MAX_LATENTS` | `360` | conditioning length; sets the re-seed interval (360 ≈ 80 s) |
| `MG2_WARMUP` | `1` | compile and warm at startup |
| `MG2_WARMUP_SESSIONS` / `MG2_WARMUP_CHUNKS` | `2` / `10` | how much warmup runs |
| `MG2_JPEG_QUALITY` | `80` | streamed frame quality |
| `MG2_SEED_WAIT_S` | `0.5` | how long a new session waits for a client seed |
| `MG2_REPO` | `/opt/ml/code/Matrix-Game/Matrix-Game-2` | upstream checkout |

## Toolchain

torch 2.8.0 (cu128), Python 3.10 and the prebuilt
`flash_attn-2.8.3.post1+cu12torch2.8cxx11abiFALSE-cp310` wheel. torch 2.6 lacks
`torch._dynamo.config.recompile_limit`; torch 2.9 has no matching Python 3.10
flash-attn wheel. The Dockerfile asserts torch is still 2.8.0 after the last
`pip install`, so a dependency that swaps it breaks the build, not the endpoint.

## Credits

The engine is ported from Aleksei Iancheruk's `world-models-infra` (the AWS Summit
Paris demo), with the Dubai booth fixes on top.
