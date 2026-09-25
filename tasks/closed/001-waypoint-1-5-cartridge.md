---
id: 001
title: Add Overworld Waypoint-1.5-1B as a real-time cartridge and validate on AWS
priority: P0
assignee: Babs Khalidson
created: 2026-09-10
due: 2026-09-11
tags: [waypoint, realtime, spike, h100]
---

# Add Overworld Waypoint-1.5-1B as a real-time cartridge

## Why
Overworld's Waypoint-1.5 family targets 60 fps at 720p on a single GPU — a far
cheaper real-time cartridge than matrix-game-3's 8×H100. Validating it proves
the accelerator's single-GPU real-time path and gives us a playable demo.

## What
- New cartridge `inference/models/waypoint-1-5/` wrapping `world_engine`
  (WorldEngine.gen_frame loop, VK-code buttons, mouse deltas).
- Shared `lib/app.py` WS handler: accept text frames + speak the frontend's
  connected/start/started/ping handshake (backwards compatible).
- Deploy on a single GPU in embark/us-west-2 and benchmark fps at 720p.
- Demo through the catalogue UI.

## Acceptance Criteria
- [x] Offline tests for actions/frames/manifest pass (145 passed, 9 skipped)
- [x] Weights staged to S3 (model + taehv1_5 AE, 11 GB)
- [x] Image builds via CodeBuild, stack WorldModel-waypoint-1-5 deploys
- [x] WebSocket session streams JPEG frames; measured fps recorded
      (H100 fp8: 82 fps @720p; A10G int8: 25 fps @720p)
- [x] Demo through frontend with WASD + camera control
      (captured; evidence kept outside the repo, see `docs/EVIDENCE.md`)

## Outcome
60 fps @ 720p is met by **one H100** — measured 82.3 fps over 7408 frames in
`fp8w8a8`, at 65% GPU utilisation and under 5 GB VRAM. Full numbers, the
warm-up cost, and the caveats are in
the waypoint-1-5 measurement write-up, kept with the rest of the evidence outside
the repo ([docs/EVIDENCE.md](../../docs/EVIDENCE.md)).

Two findings worth carrying forward:

- **A remote 60 fps demo is bandwidth-bound, not GPU-bound.** Bare JPEG at 720p
  is ~127 KiB/frame, so 60 fps needs ~62 Mbit/s per session. Over the public
  internet we measured 24–42 fps delivered, exactly tracking available Mbit/s,
  while the server produced 82. Video-encoding the stream (H.264/WebRTC) is the
  fix; a faster GPU is not.
- **Sessions drift.** The ~512-frame context is only ~6 s at 82 fps, so held
  unchanging input washes the world out within ~30 s. Upstream model property.

## Notes
- world_engine is GPL-3.0 (image-internal only; weights are Apache-2.0).
- First session pays 645 s of single-threaded `torch.compile` autotune on the
  H100 (~27 min on A10G). Mount `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR`
  on the host before recreating the container.
- `p5.4xlarge` is the smallest H100 instance, not a sizing conclusion — the
  model needs under 5 GB. A single Blackwell card (`g7e`, RTX PRO 6000) is the
  better price/performance target but no capacity was obtainable in us-west-2
  during this spike.
