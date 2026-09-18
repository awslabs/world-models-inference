# Lyra-2 — verification notes & findings

Status of the lyra-2 cartridge, what has been verified on real hardware, measured
latency, and known gaps. Written after an end-to-end bring-up on a single H100.

## What this runner does (and doesn't)

Lyra-2 is a **two-step** pipeline:

1. **Step 1 — video generation** (image → camera-controlled walkthrough `.mp4`).
   This is what our runner implements: in-process, load-once, single-GPU.
2. **Step 2 — 3D reconstruction** (video → navigable 3D Gaussian Splat `.ply`,
   via VIPE pose estimation + DA3 depth). **This is Lyra-2's actual product** —
   the walkable scene you can drop into a renderer/simulator. **Our runner does
   NOT implement Step 2** (see "Future work" below). The `.mp4` from Step 1 is an
   intermediate artifact.

So today the deployed endpoint produces the **intermediate video only**, not the
3D scene.

## Verified end-to-end (1× H100 80GB, p5.4xlarge)

Real photograph input, 480×832, 322 frames (81 zoom-in + 241 zoom-out):

| Stage | Sampling | Wall time | Output | Peak GPU |
|---|---|---|---|---|
| Step 1 — video | 50-step (default) | **~32.8 min** | intermediate `.mp4` | 74 / 80 GB |
| Step 1 — video | 4-step **DMD** | **~3.0 min** (~11× faster) | intermediate `.mp4` | ~74 GB |
| Step 2 — reconstruction | — | **~2 min** | `reconstructed_scene.ply` (758 MB, 11.15M gaussians) | needs full GPU |
| **Full (DMD + Step 2)** | | **~5 min** | image → navigable 3D scene | |

### Comparison to upstream published figures

Upstream README (1× H100 80GB) states: video ~9 min / 80 frames standard
(~35s with `--use_dmd`); reconstruction ~1 min. Frame counts must be `1 + 80k`.
Our 322 frames = 81 + 241 ≈ 4 × 80-frame chunks:

| Stage | Upstream implied (4×80f) | Ours |
|---|---|---|
| Video, standard | 4 × 9 min = ~36 min | 32.8 min |
| Video, DMD | 4 × 35s = ~2.3 min | 3.0 min |
| Reconstruction | ~1 min | ~2 min |

Caveat: single runs, one machine, one scene; upstream doesn't state its exact
frame count or whether model-load is included — treat this as a sanity check,
not a controlled benchmark.

Notes:
- **Memory:** Lyra-2 is an **80GB-class model** — the diffusion net + activations
  peak at ~74 GB. It does **not** fit a 46 GB L40S; context parallelism doesn't
  help because it shards activations (a small slice), not the ~40 GB resident net.
  Single-GPU on H100 (the p5 target) is the supported path.
- **Input matters:** feed a real photo. Synthetic/noise inputs give DA3 no depth
  structure and the output degenerates (a flat plane scaling to solid colour).

## DMD fast path (`use_dmd`)

- ~11× end-to-end speedup (32.8 → 3.0 min) with minor quality trade-off.
- Reduces diffusion sampling 50→4 steps and distills away CFG, via the DMD LoRA
  (`checkpoints/lora/dmd_distillation.safetensors`).
- Wiring: the `/generate` route forwards `use_dmd`; the runner calls upstream's
  `_apply_dmd_defaults(args)` which injects the LoRA + switches to the 4-step
  scheduler. Confirm it engaged by the log line `[DMD] Enabled: lora=...`.
- **Gotcha (fixed):** the route originally dropped `use_dmd`, so it silently ran
  the 50-step path. If you ever see two runs with near-identical wall time, check
  for the `[DMD] Enabled` line before trusting a "no speedup" conclusion.

## Future work — Step 2 (3D reconstruction → `.ply`)

Not wired into the runner. The `/generate` route accepts an `export_ply` form
field and forwards it as a param, but **the runner ignores it today** (dead stub,
kept intentionally as the hook for this work).

Step 2 was proven **manually** on the box (produced the 758 MB `.ply`). To wire it:

1. **Deps** (Dockerfile): pin `gdown==5.2.0` (VIPE calls `gdown.download(...,
   fuzzy=True)`; gdown ≥6 removed `fuzzy` → `TypeError`). For the optional preview
   render (`gs_trajectory.mp4`) also install `gsplat` (the `[gs]` extra, currently
   dropped). The `.ply` export itself does NOT need gsplat — only DA3 + VIPE.
2. **Invocation:** run as a **separate stage / subprocess**, not inside the
   in-process `generate()`:
   ```
   python -m lyra_2._src.inference.vipe_da3_gs_recon \
     --input_video_path <step1_output.mp4> --output_dir <out>
   ```
   Produces `reconstructed_scene.ply`, `cameras.npz`, `vipe_predictions.npz`.
3. **Memory:** Step 2's DA3 dense pass needs the **whole GPU** (~60 GB). It OOMs
   if the Step-1 inference server is still resident. So Step 2 cannot be a
   per-request flag on the live video server — it must run with the model
   unloaded (a distinct job type, or a post-process stage).

## Reproduce (manual Step 2, from a Step-1 video on an H100 box)

```bash
docker stop world-model                      # free the GPU
docker run --rm --gpus all --ipc host --shm-size=32g \
  -v /opt/checkpoints/lyra-2:/opt/ml/model -v /out:/out \
  -v /path/to/step1.mp4:/in.mp4 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash <lyra-2-image> -c '
    pip install -q "gdown==5.2.0"
    cd /opt/ml/code/lyra/Lyra-2 && ln -sfn /opt/ml/model/checkpoints checkpoints
    python -m lyra_2._src.inference.vipe_da3_gs_recon \
      --input_video_path /in.mp4 --output_dir /out'
# -> /out/reconstructed_scene.ply  (view in SuperSplat / Nerfstudio / any GS viewer)
```

## Runtime scaling (not a fixed 33 min)

Wall time scales ~linearly with **sampling steps × frames** — "33 min" is only the
default config (50 steps × 322 frames). From two measured points (same 322-frame
job): 50 steps → 1960 s, 4 steps (DMD) → 179 s ⇒ ~38.7 s/step + ~24 s fixed
overhead. So diffusion sampling dominates (not DA3/decode), which is why DMD's
50→4 step cut gives the ~11× speedup.

Frames are a free parameter (only constrained to the `1 + 80k` pattern: 81, 161,
241, …); Lyra-2's strength is long-horizon trajectories, so runs can go well past
33 min at 50 steps (e.g. ~2× frames ≈ ~65 min). Two unmeasured ceilings: the exact
time-vs-frames curve beyond 322f (2-point extrapolation only), and the frame count
that OOMs an 80 GB H100 — the autoregressive history buffer grows with trajectory
length (we saw 74/80 GB at 322f), so **memory, not time, is the real limit**.

## Deployment target — recommendation & gaps (NEXT STEPS)

`target` (ec2 | sagemaker) and `mode` (async) are independent axes: async batch
serves the same `/generate` + `/jobs/{id}` container API on either target. Only the
**EC2/container path has been exercised** (raw `docker run` on a p5 box). The
**SageMaker Async deployment is wired in CDK but untested.**

Recommendation for Lyra-2 (slow, bursty, expensive): **SageMaker Async** — it is
natively scale-to-zero and request-triggered (idle → 0 instances; a submitted job
lands in the managed queue → autoscaling spins up 1 → runs → scales back to 0).
That is exactly the "scale to zero, scale to 1 on trigger" behaviour you'd want,
already managed — no need to hand-build it on EC2.

Gaps to resolve before calling either production-ready:
- **[NEXT STEP] Test the SageMaker Async deployment end-to-end.** Deploy via
  `./deploy.sh lyra-2 sagemaker`, submit an async job (S3 in → S3 out), confirm
  scale-to-zero → spin-up-on-trigger → result. Nothing about this path has been
  run yet.
- **SageMaker Async has a 60-min invocation ceiling.** Fine for typical configs
  (DMD ~3 min; default 50-step ~33 min), but long 50-step trajectories can exceed
  it — those need EC2 (no ceiling) or DMD. Set `InvocationTimeoutSeconds` high for
  the 50-step path and verify.
- **EC2 construct cannot scale to zero today.** `cdk/lib/constructs/ec2.ts` hard-codes
  the ASG to `minSize/maxSize/desiredCapacity = 1` and ignores the manifest
  `scale: { min: 0, max: 2 }` block — so on EC2 a Lyra-2 deploy pins an H100 24/7.
  If EC2 is ever the target for this model, wire the ASG to honour `scale` and add
  queue-depth-based scale-to-zero (i.e. reimplement what SageMaker Async gives free).
- **Cold start** (~10–15 min: instance launch + 97 GB weight sync + model load)
  applies to both targets on scale-from-zero; acceptable for batch, note it for UX.
