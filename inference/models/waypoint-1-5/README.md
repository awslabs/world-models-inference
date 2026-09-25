# Waypoint-1.5-1B

Real-time interactive world model: seed it with one image, then hold WASD and look
around while it generates 720p frames as fast as your link can carry them.
Upstream is [Overworld](https://overworld.ai)'s
[`world_engine`](https://github.com/Overworldai/world_engine) with the
[`Overworld/Waypoint-1.5-1B`](https://huggingface.co/Overworld/Waypoint-1.5-1B)
weights, served through the shared real-time (WebSocket) path.

The fastest cartridge in the catalogue, and the only one that is real-time on a
single GPU — Matrix Game 3 needs eight to stay playable.

## At a glance

- **Upstream:** https://github.com/Overworldai/world_engine (also vendored as
  Overworld's `Biome` demo server)
- **Weights:** `Overworld/Waypoint-1.5-1B` (~12 GB) + `Overworld-Models/taehv1_5`
  (the TAEHV autoencoder, ~10 MB). The DiT itself is 1.2B params, ~4 GB in bf16;
  the repo is large because it also ships the weights in diffusers layout
  (`transformer/`, `vae/`), which this runner does not use.
- **Licence:** weights **Apache-2.0**. The `world_engine` inference library
  installed into this cartridge's image is **GPL-3.0** — fine for an internal
  spike or demo, but the *image* is not redistributable without meeting GPL
  obligations. Nothing gates this in `deploy.sh`; it is on you before you publish
  an image or offer the endpoint to third parties.
- **Compute:** `p5.4xlarge` (1× H100 80GB) — see [Hardware](#hardware) for
  cheaper shapes and their measured cost in fps.
- **Resolution:** 1280×720 native, 4-step denoise, fp8w8a8 quantisation
- **Generation rate:** **82 fps measured** on one H100, over the endpoint's own
  WebSocket from the instance itself (loopback, so no WAN in the path).
- **Delivered rate:** whatever your link carries — see
  [Frame rate is bandwidth, not GPU](#frame-rate-is-bandwidth-not-gpu). This is
  the single most important thing to understand before quoting a number.
- **Cost:** p5.4xlarge is ~$8/hr on-demand (us-west-2, subject to change) and
  normally requires a **Capacity Block**; on-demand p5 capacity is rarely
  available. `g5.2xlarge` at ~$1.20/hr runs the same cartridge at ~25 fps.

## Status

Validated end to end on a real `p5.4xlarge`: weights staged from Hugging Face,
image built by CodeBuild, ALB in front of the instance, catalogue UI driving a
session from a browser, and `./deploy.sh bench` writing a performance report.
Recordings, screenshots and the measurements behind every number here are kept
outside the repo — see [`docs/EVIDENCE.md`](../../../docs/EVIDENCE.md) — and are
available on request.

Also validated on `g5.2xlarge` (1× A10G), which has no fp8 — the runner detects
compute capability < 8.9 and falls back to int8 rather than failing in warmup.

Offline coverage:

| Test file | What it holds |
| --- | --- |
| `tests/test_waypoint_1_5.py` | manifest shape, every action wire format, JPEG encoding including the chroma default, and both env knobs falling back rather than raising on a typo |
| `tests/test_waypoint_1_5_session.py` | a whole session against the real `/ws` route and the real `stream()`, with a fake engine: handshake, four frames per engine step in order, held keys surviving flow-control acks, client seed → reset + append (and applied *once* when the client then goes quiet), single-session refusal, `setup()` reading the encoding knobs off the environment, and the flow window being configurable and disengaging at `0` |
| `tests/test_cartridge_parity.py` | this manifest, the catalogue card and the `docs/ENDPOINTS.md` row cannot drift on instance type, deploy targets, licence flags or the wired/`build` claim |

`world_engine` and CUDA are not needed for any of them: `uv sync --group dev && uv run pytest tests`.

## Deploy

Weights first — this is a **separate step** from deploying, and `./deploy.sh`
will not do it for you. (The `hf:` block in `endpoint.yaml` documents what to
stage; nothing reads it.) Two repos land under one S3 prefix:

```bash
python scripts/stage-weights.py waypoint-1-5 --source hf:Overworld/Waypoint-1.5-1B
python scripts/stage-weights.py waypoint-1-5 --source hf:Overworld-Models/taehv1_5 --subdir taehv1_5
```

The `--subdir taehv1_5` matters: the runner looks for the autoencoder at
`<ckpt_dir>/taehv1_5` and only then avoids reaching out to the Hub at boot,
which a private-subnet instance cannot do.

Then:

```bash
./deploy.sh waypoint-1-5              # build image + deploy on EC2
./deploy.sh ui waypoint-1-5           # wire the catalogue UI to it and open it
./deploy.sh bench waypoint-1-5 60     # 60s performance report → docs/evidence/
./deploy.sh destroy waypoint-1-5      # tear it down
```

To use different hardware, override the instance type — no manifest edit needed:

```bash
INSTANCE_TYPE=g5.2xlarge ./deploy.sh waypoint-1-5
```

**The first session pays for warmup.** `world_engine` compiles CUDA graphs on
first use; a cold container took **645 s** before its first frame. `/health`
answers long before then, so a client that connects early sees a socket that
accepts and then goes quiet. Wait for `Warmup (torch.compile) done` in the
container log before connecting.

`WAYPOINT_WARMUP=0` does **not** avoid that wait — it moves it. `torch.compile`
is lazy, so all it skips is the pre-emptive `gen_frame` at startup; the first
player then pays the same compile inside their session, with no log line to tell
them why nothing is happening. Steady-state throughput is identical either way.
Use it when you want the container to reach `/health` quickly and do not care
who absorbs the compile, not to make a session start sooner.

## Hardware

Every figure below was measured on the deployed endpoint at 720p, holding W, with
the client on the instance itself so the number is generation rate rather than
link rate.

| Instance | GPU | Quantisation | Generated fps | Notes |
| --- | --- | --- | --- | --- |
| `p5.4xlarge` | 1× H100 80GB | fp8w8a8 | **82** | the validated target; uses <5 GB VRAM at ~65% utilisation |
| `g5.2xlarge` | 1× A10G 24GB | int8 (auto) | **25** | Ampere has no fp8; the runner downgrades rather than failing |
| `g6e.2xlarge` | 1× L40S 48GB | fp8w8a8 | untested | Ada, so fp8 works; expect between the two |
| `g7e.2xlarge` | 1× RTX PRO 6000 | fp8w8a8 | untested | closest shape to upstream's RTX 5090 reference (they quote 72 fps fp8), but capacity was unobtainable in us-west-2 during this spike |

The H100 is the target because it is the one that was proven, **not** because the
model needs it. Under 5 GB of an 80 GB card is a cost problem worth revisiting:
if the L40S lands near 60 fps it is a much better shape for this workload.

**Quantisation does nothing on the H100.** bf16 and fp8w8a8 both measured 82.3
fps; GPU utilisation actually *fell* 65% → 51% with fp8 while board power rose
391 → 462 W, so the pipeline is not GEMM-bound on that card. `fp8w8a8` is the
image default because it is what makes the smaller cards viable, not because it
buys anything here — set `WAYPOINT_QUANT=none` on an H100 and nothing changes.
Measured, not estimated; the raw captures live outside the repo
([`docs/EVIDENCE.md`](../../../docs/EVIDENCE.md)).

### Frame rate is bandwidth, not GPU

The GPU generates ~82 fps regardless. What a player sees is:

```
delivered fps = link Mbit/s ÷ (bytes-per-frame × 8 / 1e6)
```

At 720p, quality 60, 4:2:0 chroma a frame is ~73 KiB, so that divisor is ~0.60 and
**60 fps needs ~36 Mbit/s sustained** to wherever the player is sitting. Measured
over the same London → us-west-2 path, identical server config, differing only in
the link the client happened to get:

| Chroma | bytes/frame | Link | Delivered fps |
| --- | --- | --- | --- |
| 4:4:4 (shipped first) | 85.6 KiB | 42.5 Mbit/s | 62.1 |
| 4:4:4 | 85.4 KiB | 29.4 Mbit/s | 43.0 |
| 4:4:4 | 85.6 KiB | 13.3 Mbit/s | 19.4 |
| **4:2:0 (current default)** | 74.7 KiB | 27.3 Mbit/s | 44.6 |
| **4:2:0** | 69.2 KiB | 7.1 Mbit/s | 12.6 |

Every row lands on the formula within 3%. No run happened to get a link fast
enough to show 60 fps at 4:2:0 — that needs ~36 Mbit/s and the best 4:2:0 sample
got 27.3, which is why the 60 fps figure is arithmetic from the measured frame
size rather than an observed number. The three 4:4:4 rows are the ones that
bracket 60 fps, at the higher bandwidth their larger frames demand.

So a low fps number from a remote client is a statement about the network. To
measure the *model*, run the benchmark on the instance over loopback — the report
labels which of the two it captured.

There is also a latency/throughput trade-off in the shared serving layer. Without
flow control every buffer between the encoder and the browser fills during the
opening burst and never drains, and in-session ping measured **1799 ms**.
`WORLD_MODEL_FLOW_WINDOW` bounds frames in flight, which brought that to
**201 ms** — the floor for the path, since a bare TCP connect to the ALB measures
174–250 ms — at a ~14% throughput cost, because a window of W frames caps
throughput at `W / RTT`. Both of those numbers were measured at a window of
**12**; the shipped default is 16, raised afterwards to win the throughput back,
which at a 220 ms round-trip puts the ceiling near 72 fps. Deploying in the
player's region raises that ceiling far more than any GPU change would.

## What you invoke

```
GET  /ping                                health
GET  /health                              readiness + model info
WS   /ws                                  interactive session (EC2)
WS   /invocations-bidirectional-stream     interactive session (SageMaker)
```

The socket takes the same bearer token as every other endpoint, as an
`Authorization` header or a `token` query parameter.

This cartridge has no batch mode, so the shared server's async routes exist but
cannot succeed: `POST /generate` accepts the request and returns a `queued`
`job_id` as usual, then the job fails when the worker calls `generate()` — poll
`GET /jobs/{id}` to see the `NotImplementedError`. `POST /invocations` calls
`generate()` inline, so it fails immediately with a 500 instead. Use `/ws`.

`endpoint.yaml` declares `sagemaker.instance: ml.g6e.2xlarge`, but only the EC2
path has been exercised, which is why the catalogue card offers EC2 only.

### Session protocol

One session at a time (`max_concurrent: 1`) — a real-time runner holds one
engine, one KV cache and one GPU worker, so a second socket is refused rather
than allowed to reset the world under the player.

Server → client on connect:

```json
{"type": "connected", "flow_window": 16}
```

Client → server, any of these forms per tick:

| Form | Example | When to use |
| --- | --- | --- |
| Catalogue UI JSON | `{"type":"control","buttons":["W","L"],"mouse_dx":40,"mouse_dy":0}` | what `frontend/` sends |
| Matrix Game 3 keys | `{"keys":"wl"}` | reuse an existing client |
| Vectors | `{"keyboard":[1,0,0,0,0,0],"mouse":[0.1,0.0]}` | driving the model directly |
| Binary | `<uint32 key bitflags><float32 pitch><float32 yaw>` | lowest overhead in a game loop |

`buttons` are single characters; WASD moves, IJKL looks (they are folded into
mouse deltas, not passed through as keys), and `mouse_dx`/`mouse_dy` are stick
positions in ±100 scaled by `WAYPOINT_MOUSE_SCALE`. Anything unparseable decodes
to a neutral action rather than dropping the session.

Two message types are handled by the serving layer and never reach the runner:

- `{"type":"ack","n":<count>}` — frames received so far. Required to keep the
  flow window open; a client that never acks gets one stall and then unbounded
  sending (and the lag that comes with it).
- `{"type":"ping","timestamp":t}` → `{"type":"pong","timestamp":t}`. The pong is
  queued behind the frames already on the connection, which is exactly why it is
  an honest measure of what the player feels.

`{"type":"start","image_data":"data:image/png;base64,..."}` seeds the world from
a client image: the engine is reset and that frame appended, so the world starts
from precisely that picture. Without an image, `start` is just a handshake and
the boot seed stands. An undecodable image is logged and ignored — the world the
player is already in is not blanked.

Server → client frames are **bare JPEGs**, one binary message each, no envelope
(`frontend/src/hooks/useFrameBuffer.ts` renders the Blob straight to a canvas).
Each engine step yields `temporal_compression` = **4** frames.

## How it works

`world_engine` is an autoregressive diffusion transformer. `reset()` clears the
frame context, `append_frame(seed)` establishes the world, and each
`gen_frame(ctrl)` denoises the next latent conditioned on the held buttons and
mouse deltas; the TAEHV autoencoder decodes that latent into 4 RGB frames.

Two things in the adapter are load-bearing:

**One thread owns the GPU.** Construction, warmup, reset and every `gen_frame`
run on a single dedicated device thread. `world_engine` leans on `torch.compile`
with CUDA graphs, and a compiled graph must execute on the thread that compiled
it — the same discipline Overworld's own server applies. The shared framework
calls `reset()` on the asyncio thread and `stream()` on an executor thread, so
both trampoline onto the device thread instead of touching the engine.

**Generation and encoding overlap.** `stream()` submits step N+1 to the GPU
before JPEG-encoding and yielding step N. At 720p that overlap is what keeps 60
fps reachable; without it the CPU encode time is dead GPU time.

Actions are read by a separate pump thread into a latest-wins slot, so input rate
and generation rate are decoupled: a key takes effect on the next engine step
that has not already been submitted, not on the next message. Because `stream()`
runs one step ahead, that is one to two steps of 4 frames each — **~50 to ~100 ms
of generated video** at 82 fps. On top of that a player waits out whatever frames
are already in flight (up to `WORLD_MODEL_FLOW_WINDOW`, 16 frames), which is why
the benched end-to-end input lag is 263 ms on a 220 ms round-trip path rather
than anything near 50 ms.

## Environment overrides

Set on the container so a deployment can be retuned without a rebuild.

| Variable | Default | Purpose |
| --- | --- | --- |
| `WAYPOINT_QUANT` | `fp8w8a8` | `none`, `intw8a8`, `fp8w8a8` or `nvfp4`. Only `fp8w8a8` auto-downgrades (to `intw8a8`) below compute capability 8.9; ask for `nvfp4` on a card without it and warmup fails. |
| `WAYPOINT_JPEG_QUALITY` | `60` | JPEG quality for delivered frames — the bandwidth dial |
| `WAYPOINT_JPEG_SUBSAMPLING` | `420` | chroma subsampling: `420`, `422` or `444`. Anything else logs a warning at startup and uses the default |
| `WAYPOINT_MOUSE_SCALE` | `15` | mouse counts per step at full stick deflection |
| `WAYPOINT_SEED` | demo video frame | path to a boot seed image |
| `WAYPOINT_WARMUP` | `1` | `0` skips the startup `gen_frame`, so the first player pays the `torch.compile` instead. Same steady state; see [Deploy](#deploy) |
| `WORLD_MODEL_FLOW_WINDOW` | `16` | frames in flight before the server waits for an ack; `0` disables |

## Making it faster

Every lever priced on 180 real frames sampled from the live endpoint, since
delivered fps is bytes-bound. Bytes per delivered frame, and the link 60 fps
would then need:

Relative fps is against the current 4:2:0 default, so a row's number is what you
would gain by switching to it today:

| Encoding | Bytes/frame | Link for 60 fps | Relative fps |
| --- | --- | --- | --- |
| JPEG q60 **4:4:4** (what shipped first) | 86.2 KiB | 42.4 Mbit/s | 0.85× |
| JPEG q60 **4:2:0** (current default) | 73.1 KiB | 35.9 Mbit/s | 1.00× |
| JPEG q45 4:2:0 | 67.3 KiB | 33.1 Mbit/s | 1.09× |
| JPEG q60 4:2:0 at 960×540 | 50.5 KiB | 24.8 Mbit/s | 1.45× |
| **H.264 crf23** (x264, zerolatency) | 15.7 KiB | 7.7 Mbit/s | 4.7× |
| H.264 crf28 | 7.3 KiB | 3.6 Mbit/s | 10.0× |
| H.265 crf28 | 7.0 KiB | 3.4 Mbit/s | 10.4× |

4:2:0 was free and is now the default — full-resolution chroma buys nothing a
player can see in a moving world. The H.264/H.265 rows are x264 on a CPU; NVENC
on the H100 needs roughly 10–20% more bitrate for the same quality, so treat
them as the optimistic end. **A video codec over WebRTC is the real unlock** —
roughly a day of work for 4.7–10× fewer bytes — and no GPU upgrade comes close.

Non-encoding levers: deploy in the player's region (removes ~150 ms RTT, so the
`W / RTT` flow ceiling goes from ~72 fps at a 220 ms round-trip to ~230 fps at
70 ms — high enough that flow control stops being the binding constraint at all,
since the GPU only generates 82), and drop to 960×540 if 1.45× fps is worth the
pixels.

## Caveats

- **Coherence decays with frames generated, not seconds elapsed.** The context
  window is ~512 frames — about **6 s of generated video** at 82 fps — after which
  the world visibly softens and then dissolves. Because a remote player receives
  frames at their link rate, that same budget stretches over ~10 s of play at 45
  fps and ~20 s at 25 fps: a slow connection buys you a longer-lasting world, not
  a better one. Upstream model property, not a deployment defect —
  a contact sheet of the 897-frame highway capture shows it frame by frame
  (kept outside the repo, see `docs/EVIDENCE.md`).
- **A stale session looks like a dead endpoint.** A client that vanishes without
  closing cleanly (browser tab, Ctrl-C) leaves the socket open through the ALB,
  so the endpoint counts itself busy until the ~60 s idle timeout tears the dead
  socket down. During that window every new connection is refused the moment it
  is accepted — `{"type":"error"}` then close 1013 — and the catalogue UI only
  logs that to the browser console, leaving a black canvas that looks like a
  broken endpoint. Wait it out and retry.
- **The spike's ALB had no TLS listener**, so the token travelled in the clear
  over HTTP:80. Acceptable for an IP-restricted spike; add HTTPS before leaving
  anything running.
