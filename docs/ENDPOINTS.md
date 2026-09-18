# Endpoints

One directory per cartridge under `inference/models/`. "Wired" means the server code
exists, the deploy path works, and an end-to-end invoke has been verified.

## Models

| ID | Type | Mode | EC2 instance | GPUs | Weights | Wired? |
|----|------|------|--------------|------|---------|--------|
| `lingbot-fast` | video generation | async | `p5.48xlarge` | 8× H100 80GB | ~234 GB | ✅ end-to-end |
| `cosmos3-nano` | video generation | async | `g6e.12xlarge` | 4× L40S | ~35 GB | ✅ end-to-end |
| `lyra-2` | image→3D-world video | async | `p5.4xlarge` | 1× H100 80GB | gated | ✅ end-to-end |
| `vjepa2` | representation | real-time | `g5.2xlarge` | 1× A10G | ~4 GB | ✅ end-to-end |
| `matrix-game-3` | real-time | real-time | `g5.2xlarge` | placeholder | — | ❌ scaffold only |

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

### `matrix-game-3`

Upstream: [Matrix Game 3.0](https://github.com/SkyworkAI/Matrix-Game-3.0).
**Scaffold only** — `endpoint.yaml` and a real-time handler exist, but the upstream
inference code is not wired into the runner.

Its `endpoint.yaml` still carries a cheap placeholder instance (`g5.2xlarge`). The
real model needs 8× H100 (`p5.48xlarge`) — set that when you wire the runner up.

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

`./deploy.sh ui` starts the catalogue at `localhost:3000`. Only `ready` cartridges are
clickable; the rest render a "Not wired up yet" view pointing at the upstream repo.

> The UI's card list (`frontend/src/data/cartridges.ts`) is maintained separately from
> `inference/models/`, so the two can drift. It currently lists `vjepa2-ac` where the
> models directory has `vjepa2`, and omits `echo-realtime`.
