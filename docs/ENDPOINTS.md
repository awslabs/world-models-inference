# Endpoints

One directory per cartridge under `inference/models/`. "Wired" means the server code
exists, the deploy path works, and an end-to-end invoke has been verified.

## Models

| ID | Type | Mode | EC2 instance | GPUs | Weights | Wired? |
|----|------|------|--------------|------|---------|--------|
| `lingbot-fast` | video generation | async | `p5.48xlarge` | 8× H100 80GB | ~234 GB | ✅ end-to-end |
| `cosmos3-nano` | video generation | async | `g6e.12xlarge` | 4× L40S | ~35 GB | ✅ end-to-end |
| `lyra-2` | image→3D-world video | async | `p5.4xlarge` | 1× H100 80GB | gated | ✅ end-to-end |
| `vjepa2` | representation | real-time | `g5.2xlarge` | 1× A10G | ~4 GB | ⚠️ encoder only |
| `matrix-game-3` | real-time | real-time | `p5.48xlarge` | 8× H100 80GB | ~57 GB | ✅ end-to-end |
| `waypoint-1-5` | real-time | real-time | `p5.4xlarge` | 1× H100 80GB | ~12 GB | ✅ end-to-end |

### Smoke-test cartridges

| ID | Mode | EC2 instance | Purpose |
|----|------|--------------|---------|
| `echo-async` | async | `g5.2xlarge` | Tiny synthetic-video cartridge — no weights, seconds per job. Returns a real MP4 |
| `echo-realtime` | real-time | `g5.12xlarge` | Streaming equivalent |

Neither is a world model. Use them to confirm the deploy path, auth and UI wiring
work before spending on a large GPU: `./deploy.sh echo-async` costs ~$1.20/hr.

`_template` is a scaffold, hidden from `./deploy.sh list`. See
[Adding a model](ADDING_A_MODEL.md).

## Per-model notes

### `lingbot-fast`

Upstream: [LingBot-World-Fast](https://github.com/robbyant/lingbot-world). The
flagship — camera-controlled video generation across 8 GPUs, matching upstream's
`--nproc_per_node=8 --ulysses_size 8`.

Built from the **shared** `Dockerfile.default`, with `base_image` pinned to an NGC
image carrying torch 2.4 / CUDA 12.6 / Python 3.10 and a matching prebuilt flash-attn
wheel.

Weights come from two Hugging Face repos (`lingbot-world-base-cam` plus the KV-cached
fast head). Both are **full models**, not a base plus a small head — hence ~234 GB
rather than the ~60 GB you might expect. The 500 GB volume accommodates this.

### `cosmos3-nano`

Upstream: [NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos). Omnimodal — video
generation, world simulation, action policy. Uses a custom `Dockerfile`.

### `lyra-2`

Upstream: [Lyra 2.0](https://github.com/nv-tlabs/lyra). Image → 3D-consistent camera
fly-through video.

Two caveats:

- **Single-GPU.** The runner wraps upstream's script entry points as a subprocess
  that claims the local GPU, so it runs `WORLD_SIZE=1` rather than using torchrun
  parallelism. It needs one H100-class GPU — a 46 GB L40S OOMs at load. An 8-GPU box
  would leave 7 idle, so `p5.4xlarge` is the supported target.
- **Restricted licence.** Weights are under the NVIDIA Internal Scientific Research
  and Development Model License — internal R&D only, not for production, public
  deployment or redistribution. `deploy.sh` gates deployment on acknowledging this.
  You are responsible for compliant use in your own account.

Only Step 1 (video) is exposed. The Step-2 3D Gaussian-splat reconstruction is not
wired up; `export_ply` is accepted by the API as a forward-looking hook but no runner
consumes it.

### `vjepa2`

Upstream: [V-JEPA 2](https://github.com/facebookresearch/vjepa2). Real-time video
encoder, streamed over WebSocket with no job queue. `max_concurrent: 1` — one session
per container, so scale horizontally rather than per-instance.

What runs is `facebook/vjepa2-vitg-fpc64-384` — the plain encoder — over a 16-frame
sliding window, emitting a pooled embedding plus a cosine-distance alert per frame.
The action-conditioned **V-JEPA 2-AC** wrapper is *not* hooked up, which is why this
row reads "encoder only" rather than end-to-end and the card carries
`build: 'scaffold'`. Treat the fps figure as untested.

The **data flow is inverted** relative to every other real-time cartridge: the client
sends video and the server answers with vectors, not frames. So the catalogue UI has no
player for it (`Catalogue.tsx` routes `type: 'representation'` to an explanatory panel
rather than to `CartridgePlayer`, which would open a session and then paint nothing),
and no GPU has run it. The runner is covered offline by
[`tests/test_vjepa2_session.py`](../tests/test_vjepa2_session.py) — wire format,
warm-up window, baseline, alert threshold and session reset, against the real `/ws`
route with a stubbed encoder. See
[`inference/models/vjepa2/README.md`](../inference/models/vjepa2/README.md).

### `matrix-game-3`

Upstream: [Matrix Game 3.0](https://github.com/SkyworkAI/Matrix-Game-3.0). Validated
end to end on a real `p5.48xlarge`: **~35 fps sustained at 832×480 on 8 GPUs**,
measured through the endpoint from a remote client. A single GPU reaches ~44 fps
inside a chunk but freezes ~7 s between chunks, so 8 GPUs is what makes it playable
rather than merely fast in bursts.

Actions apply **per chunk, not per frame** (57 frames for the first chunk, 40 after),
so at ~35 fps you steer the world roughly once a second, not per keypress. That is a
property of the model.

Full detail — the two seams the runner hooks into upstream's batch pipeline, the
lockstep multi-GPU loop, and every `MG3_*` tuning knob — is in
[`inference/models/matrix-game-3/README.md`](../inference/models/matrix-game-3/README.md).

### `waypoint-1-5`

Upstream: [Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B). A 1.2B
autoregressive diffusion transformer with a TAEHV autoencoder, driven through
Overworld's `world_engine`. Native 720p, 4-step denoise, and four frames yielded per
engine step. WASD moves, IJKL (or a right stick / mouse delta) turns the camera.

**Measured throughput at 720p**, taken over the WebSocket from inside the container so
no network sits in the path:

| GPU | Instance | Quantization | fps |
|-----|----------|--------------|-----|
| 1× H100 80GB | `p5.4xlarge` | `fp8w8a8` | 82 |
| 1× A10G 24GB | `g5.2xlarge` | `intw8a8` | 25 |

The H100 run held 82.3 fps over 7408 frames with no clock throttling, at 65% GPU
utilisation and **under 5 GB of VRAM** — the model is far smaller than the card. fp8
needs compute capability ≥ 8.9, so on the A10G (8.6) the runner falls back to `intw8a8`
on its own.

Three things matter more than the fps number when planning a deployment:

- **Bandwidth, not GPU, limits what a remote player sees.** Frames go out as bare
  JPEGs — ~73 KiB each at the shipped defaults (quality 60, 4:2:0 chroma) — so
  `delivered fps = link Mbit/s ÷ (bytes-per-frame × 8 / 1e6)`, and 60 fps at 720p
  needs **~36 Mbit/s** of sustained downstream per session. Streaming from us-west-2
  to a laptop over the public internet delivered anywhere from 12.6 to 62.1 fps purely
  as a function of the link, while the server produced 82 throughout. A dependable
  remote 60 fps demo wants H.264/WebRTC in front of the socket (15.7 KiB per frame
  measured on real output, ~4.7× fewer bytes), not a faster GPU.
- **Frames in flight are capped, and that is what fixed the lag.** Without flow
  control every buffer between encoder and browser fills during the opening burst and
  never drains: in-session ping measured 1799 ms. The server sends at most
  `WORLD_MODEL_FLOW_WINDOW` frames ahead of what the client has acknowledged, which
  brought that to 201 ms — the floor for a London → us-west-2 path — at a ~14%
  throughput cost, since a window of W frames caps throughput at `W / RTT`. Both
  figures were measured at a window of 12; the shipped default is 16, raised
  afterwards to recover the throughput.
- **The world drifts.** The frame context is ~512 frames — about **6 s of generated
  video** at 82 fps, or roughly ten seconds of play at the rate a remote link
  delivers. A held, unchanging input walks the world off-distribution and it washes
  out; a contact sheet of the highway capture shows the decay (kept outside the
  repo, see `EVIDENCE.md`).

`torch.compile(fullgraph=True)` with max-autotune plus CUDA graphs means the first
session pays a warm-up: **645 s on the H100**, ~27 min on the A10G. `/health` answers
long before then, so an early client sees a socket that accepts and stays quiet. Every
engine call must happen on one dedicated thread, which the runner enforces. Mount
`TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` on the host if you plan to recreate the
container.

Weights are Apache-2.0; `world_engine` itself is GPL-3.0, so the built image carries a
GPL obligation — fine internally, check before redistributing it.

Full detail — session protocol, the measured encoding menu, every `WAYPOINT_*` knob —
is in [`inference/models/waypoint-1-5/README.md`](../inference/models/waypoint-1-5/README.md).

## Async vs real-time

The mode is derived from whether `sagemaker.output` is present in `endpoint.yaml`,
**not** from a `mode:` key:

- **async** — `POST /generate` enqueues a job and returns a `job_id`; poll
  `GET /jobs/{id}`, then fetch `GET /jobs/{id}/output`. Every rank runs the job in
  lockstep.
- **real-time** — a WebSocket session on `/ws` streams frames continuously. No job
  queue.

See [Architecture](ARCHITECTURE.md#run).

## Catalogue UI

`./deploy.sh ui [model]` starts the catalogue at `localhost:3000`, discovering the
endpoint URL from CloudFormation and injecting a Cognito access token server-side so
nothing secret reaches the bundle.

Every card carries a status badge. Two independent facts feed it — what the code in
this repo supports, and whether a GPU is answering right now:

| Badge | Meaning |
|---|---|
| `live` | `/health` answered just now. Click to play. |
| `down` | This UI is pointed here, but the endpoint is not answering. |
| `deployable` | Code is wired; no endpoint behind it. Run `./deploy.sh <id>`. |
| `not wired` | Scaffold only — the detail view points at the upstream repo. |

Liveness cannot live in a checked-in file: deployments cost money and get torn down.
So `cartridges.ts` records only the **build** state (`wired` / `scaffold`), and the UI
probes `/health` at runtime. Only the cartridge named by `config.js`'s `deployedModel`
can show `live`, because that is the one this UI is pointed at.

The card list (`frontend/src/data/cartridges.ts`) is maintained by hand alongside
`inference/models/*/endpoint.yaml`, so `tests/test_cartridge_parity.py` fails the build
if the two disagree on ids, instance types, deploy targets or licence flags.
