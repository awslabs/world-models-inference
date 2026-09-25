# V-JEPA 2

The only cartridge in the catalogue that **consumes** video instead of producing
it. You stream frames at it; it answers with a vector per frame and a flag that
goes up when the scene stops resembling the one the session opened with.

Upstream is Meta FAIR's [V-JEPA 2](https://github.com/facebookresearch/vjepa2),
served through the shared real-time (WebSocket) path.

## What it is for

V-JEPA 2 is a self-supervised video encoder: it was trained to predict masked
regions of video *in representation space* rather than in pixels, so what it
learns is a compact description of what a clip contains and how it is moving.
There is no decoder — you cannot ask it for a picture. What you get is the
description.

That makes it the right tool for questions of the form *"is what I am looking at
now still the thing I was looking at before?"*, which is what this runner
implements:

- **Novelty / anomaly detection on a live feed.** The first full 16-frame window
  of a session becomes the baseline. Every window after it is scored by cosine
  distance from that baseline, and past `0.5` the reply carries
  `"alert": true`. Dashcam, CCTV, a production line, a drone feed — anywhere you
  want "tell me when this changes" without training a classifier for the specific
  thing you are afraid of, and without shipping every frame somewhere for review.
- **Embeddings for something downstream.** The vector is in the reply whether or
  not you care about the alert: cluster it, index it for retrieval, feed it to a
  small classifier head, or store it as a searchable summary of a long recording.

Two things it is deliberately *not*:

- **Not a world model.** It does not generate, and it cannot be played. If you
  want to steer a world with WASD, that is `waypoint-1-5` or `matrix-game-3`.
- **Not V-JEPA 2-AC.** The action-conditioned variant — the one you would use for
  robot planning, where you roll the representation forward under a candidate
  action — is a different checkpoint and is not wired up here. This runner loads
  the plain encoder. The catalogue card used to claim otherwise; it no longer
  does.

## At a glance

- **Upstream:** https://github.com/facebookresearch/vjepa2
- **Weights:** `facebook/vjepa2-vitg-fpc64-384` (ViT-g, 384px, ~4 GB), pinned in
  `runner.py` to revision `12ca9169…` for the Hub-download fallback
- **Licence:** upstream weights and code. Check the model card before shipping —
  nothing in `deploy.sh` gates this cartridge.
- **Compute:** `g5.2xlarge` (1× A10G 24GB) declared. **Untested** — no GPU has
  run this cartridge, so the `~30 fps` on its card is a design target and not a
  measurement.
- **Window:** 16 frames (`WINDOW_SIZE`), alert at cosine distance `> 0.5`
  (`ALERT_THRESHOLD`). Both are module constants, not environment knobs.
- **Concurrency:** `max_concurrent: 1`. One window and one baseline per runner,
  so a second client would interleave two cameras into one window. Scale
  horizontally — one container per feed.

## Status

**Scaffold.** The runner is written and tested offline; what it has never had is
a GPU and a client.

| | |
| --- | --- |
| Runner implemented | ✅ |
| Offline coverage | ✅ `tests/test_vjepa2_session.py` |
| Deployed to a GPU | ❌ never |
| Measured fps | ❌ the card's 30 is a target |
| Browser client | ❌ see [No UI path](#no-ui-path) |

`tests/test_vjepa2_session.py` drives whole sessions through the real FastAPI
`/ws` route and the real `stream()`, with torch and `transformers` stubbed and a
deterministic fake encoder whose output depends on the pixels it is handed — so
the assertions on distance and alerting are about the runner's logic rather than
about a constant. What it holds: the wire format; no embedding until the window
fills; the frame counter matching the frames actually sent; the first full window
becoming the baseline and surviving a later scene change; an alert on a changed
scene and no alert on drift; the window sliding, so one odd frame among fifteen
normal ones does not alert; a clean window per session; the second client being
refused; an undecodable frame being skipped rather than fatal; and both branches
of `setup()`, including the pinned Hub revision.

No GPU, no CUDA and no `transformers` install needed:
`uv sync --group dev && uv run pytest tests/test_vjepa2_session.py`.

## Deploy

Weights first — a separate step from deploying, which `./deploy.sh` will not do
for you:

```bash
python scripts/stage-weights.py vjepa2 --source hf:facebook/vjepa2-vitg-fpc64-384
```

Staging matters more than usual here: without it the runner falls back to
downloading from the Hub at boot, which an instance in a private subnet cannot
do. `setup()` decides by looking for `config.json` in the model directory.

Then:

```bash
./deploy.sh vjepa2                 # build image + deploy on EC2
./deploy.sh destroy vjepa2         # tear it down
```

`./deploy.sh ui vjepa2` will point the catalogue at it, but see below for what
you will see.

## No UI path

The catalogue cannot drive this cartridge, and that is a missing feature rather
than a bug to hunt. Every other card assumes the server sends pictures:
`CartridgePlayer` seeds a world from an image and takes keyboard input, and
`useFrameBuffer` decodes each binary message with `createImageBitmap` and drops
whatever fails. Point either at this endpoint and you get a canvas that never
paints.

`Catalogue.tsx` therefore routes `type: 'representation'` to an explanatory panel
before it reaches the player, so a live endpoint says what it is instead of
looking broken. Driving it needs a different client — a camera or file picker on
the way in, a distance readout and an alert light on the way out — which nobody
has built yet.

Until then, use the socket directly (see the protocol below); it is a handful of
lines in any language with a WebSocket and a JPEG encoder.

## What you invoke

```
GET  /ping                                health
GET  /health                              readiness + model info
WS   /ws                                  streaming session (EC2)
WS   /invocations-bidirectional-stream    streaming session (SageMaker)
```

The socket takes the same bearer token as every other endpoint, as an
`Authorization` header or a `token` query parameter.

There is no batch mode. `generate()` raises `NotImplementedError`, so
`POST /generate` queues a job that then fails — poll `GET /jobs/{id}` to see it —
and `POST /invocations` fails inline with a 500. Use `/ws`.

### Session protocol

**Client → server:** one JPEG per binary message. Nothing else; there is no
control channel. Send frames in lockstep with the replies — the shared
`ActionBuffer` is a latest-wins slot rather than a queue, so frames pushed faster
than the encoder consumes them are dropped rather than buffered.

**Server → client:** one binary message per accepted frame.

```
[4B frame_num LE][4B meta_len LE][JSON meta][float16 embedding]
```

While the window is filling (frames 1–15) there is no embedding and the meta is:

```json
{"frame": 3, "warming_up": true, "frames_buffered": 3}
```

From frame 16 on:

```json
{"frame": 16, "alert": false, "distance": 0.0, "embed_dim": 1408, "dtype": "float16"}
```

`distance` is `1 - cosine_similarity(this window, the first full window)`, so the
frame that establishes the baseline always reports `0.0`. `embed_dim` is the
encoder's width; read it from the message rather than hard-coding it.

Two message types are handled by the serving layer and never reach the runner:
`{"type":"ack","n":N}`, which keeps the flow window open (`WORLD_MODEL_FLOW_WINDOW`,
default 16 messages in flight), and `{"type":"ping","timestamp":t}` → `pong`.

## How it works

`stream()` keeps a `deque(maxlen=16)` of decoded frames. Each new frame is
appended and, once the deque is full, the whole window is passed through the
video processor and `get_vision_features`, mean-pooled over the patch/time axis
into one vector. The first such vector is kept as `self.reference`; every later
one is compared against it. `reset()` clears all three per-session (window,
reference, counter), which the route calls when a client connects.

One thing in the adapter is load-bearing: **a re-delivered payload is not a new
frame.** `ActionBuffer.get()` re-returns the payload it already gave out every
≤0.1 s whether or not the client sent anything, which for a cartridge reading
held keyboard input is harmless and here would not be: a camera that paused would
have its last frame appended ~10× a second, filling the window with copies of one
frame, racing `frame_count` ahead of the frames actually sent, and taking a
duplicate as the session baseline if the pause landed during warm-up. The loop
skips a payload that is the *same object* as the last one. Identity and not
equality, so a synthetic source whose frames encode byte-identically — a test
pattern, a screen capture of a still, a looping clip — is still processed.

## Caveats

- **Nothing here is measured.** No GPU has run it. Expect the first deploy to
  surface something.
- **The window and threshold are constants.** `0.5` cosine distance was chosen a
  priori, not calibrated against a feed; it is the first thing to tune once you
  have real footage, and it belongs in the environment rather than in the source
  once you do.
- **The baseline is whatever the session opened with.** If the first 16 frames
  are themselves the anomaly, everything after them reads as normal. A real
  deployment probably wants a baseline captured deliberately rather than by
  arriving first.
- **Coherence of the alert depends on frame cadence, not wall clock.** The window
  is 16 frames, so at 30 fps it spans half a second and at 5 fps it spans three.
