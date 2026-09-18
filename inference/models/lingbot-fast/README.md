# LingBot-World-Fast

Async video generation via the upstream [LingBot-World-Fast](https://github.com/robbyant/lingbot-world) pipeline, deployed to a single EC2 instance running 8-GPU `torchrun` behind a FastAPI server.

## At a glance

- **Upstream repo:** https://github.com/robbyant/lingbot-world
- **Weights:** `robbyant/lingbot-world-base-cam` + `robbyant/lingbot-world-fast` (~60 GB total)
- **Compute:** `p5.48xlarge` (8× H100 80GB)
- **Inference time:** ~60 s per 81-frame clip (~5 s of video @ 16 fps) once warm
- **Cold start:** ~20 min for first deploy (weight download + flash-attn wheel + model init)
- **Cost:** ~$55/hr on-demand. Use a **Capacity Block** reservation for predictable pricing — see the top-level CLI for guidance.

## What you invoke

```
GET  /lingbot/health            liveness + GPU info
GET  /lingbot/examples          list bundled examples (copied from upstream)
GET  /lingbot/examples/{id}/image
POST /lingbot/generate          submit job
GET  /lingbot/status/{id}       poll
GET  /lingbot/result/{id}       download MP4
GET  /ping                      generic alias for health checks
```

`POST /lingbot/generate` accepts multipart form:
- `prompt` (required)
- Either `image` (upload) OR `example_id` (one of `00`..`05` shipped alongside)
- `frame_num` (default 81, must be 4n+1)
- `size` (default `480*832`)
- `seed` (default 42)

## Architecture

```
Single EC2 p5.48xlarge
├── ./lib/serve → torchrun --nproc_per_node=8 -m lib.serve
│     ├── rank 0: FastAPI on :8080, worker thread drains job queue,
│     │           broadcasts generate commands to followers, saves MP4
│     └── ranks 1-7: block on dist.broadcast_object_list, run
│                    pipeline.generate(...) in lockstep with rank 0
└── /opt/checkpoints/lingbot-fast/          (base + fast head)
```

All ranks load `WanI2VFast` once at boot; the model stays on GPU for the lifetime of the instance. Per-request latency is bounded by the upstream pipeline itself, not startup.

## Deploy

Use the top-level `lingbot` CLI:

```bash
./deploy              # launch on-demand p5.48xlarge
./deploy --capacity-block cr-abc123    # use reserved capacity
./deploy status
./deploy invoke --example 03 --prompt "..." --output out.mp4
./deploy stop
```

## Local dev

Single-GPU mode (no Ulysses SP, expect OOM on anything smaller than H100 80GB):
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
# Run the shared entrypoint from the inference/ dir with one process:
NPROC_PER_NODE=1 CKPT_DIR=/path/to/lingbot-world-base-cam ./lib/serve
```

## Vendored code

`wan/` and `examples/` are copied verbatim from upstream. See `UPSTREAM_LICENSE.txt` for the Apache-2.0 license.
