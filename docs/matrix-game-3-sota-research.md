# State of the art for real-time interactive video world models — applied to Matrix Game 3

Deep research survey, 2026-09-10.
Baseline under discussion: MG3 at 832×480, 3 UniPC steps, int8 DiT, LightVAE,
Ulysses-8 on 8×H100 ≈ 30 fps / 1.34 s chunk.

All fps/latency figures are author-self-reported unless marked
"measured"/MLPerf. 2026 preprints (WorldKV, MoGAN, OPSD-V, Matrix-Game-3.5)
have no third-party validation yet. Marketing-grade numbers are flagged.

## TL;DR

1. The fastest wins are config-level — Skywork's own 40 fps number is a 9-GPU
   "8+1" async setup (8 DiT + 1 dedicated VAE GPU) with FA3, int8,
   LightVAE-v2@0.75 and GPU-side memory retrieval. Their ablation says GPU
   retrieval alone is worth 6.6→40 fps.
2. 60 fps has two credible paths: RIFE-TensorRT frame interpolation (~1–3
   ms/frame at 832×480, +$1.86/hr L40S sidecar) or a 2-step re-distillation —
   NVIDIA LongLive-2.0 (Apache-2.0, same Wan2.2-TI2V-5B base as MG3) ships a
   2-step NVFP4 checkpoint at 45.7 FPS on H100 (README verified).
3. Input latency is an architecture problem (input once per 40-frame chunk =
   up to 1.34 s); the industry norm is per-frame or 3-latent-chunk
   conditioning — fixable only by re-distillation, though mid-chunk action
   injection is a cheap hack.
4. Skip: H200 (+6.5% MLPerf SDXL for +15% cost), TeaCache/FBCache (useless at
   3 steps), SVDQuant/Nunchaku on H100 (no Wan support), full TensorRT port
   (~10–25% over compiled PyTorch for weeks of work).

## 1. Config-level / drop-in (verify before anything else)

Matrix-Game-3 tech report (github.com/SkyworkAI/Matrix-Game →
Matrix-Game-3/assets/pdf/report.pdf; README flags verified directly):

- The "40 fps @ 720p" claim = async 8+1 setup: 8 GPUs DiT + 1 GPU dedicated
  VAE decode, `--fa_version 3`, `--use_int8`,
  `--vae_type mg_lightvae_v2 --lightvae_pruning_rate 0.75`,
  `--use_async_vae --async_vae_warmup_iters 1`, GPU-based camera-frustum
  memory retrieval. "7+1 on one node is slightly lower." Our 8-GPU ~30 fps at
  832×480 is roughly consistent with their numbers — we're not misconfigured
  by 33%, we're comparing against a 9-GPU async pipeline.
- Ablation (Table 1, FPS after removing each piece from ~40): −int8 → 27.4;
  −LightVAE → 25.8; −GPU retrieval → 6.6. (Our lockstep path does use the GPU
  retrieval path, with sync overhead — see audit (b)/(d).)
- LightVAE-v2@0.75 decodes 17f@720p in 0.13 s vs 0.76 s stock Wan VAE (5.2×,
  PSNR 33.8→31.1). We run v1@0.5 — bump it (audit fix #4).
- Pre-warm the VAE torch.compile at server start (repo issue #21: the ~1-min
  first-input stall).
- Upstream issues confirm our known bugs: #80 is exactly our SP TypeError
  (ulysses.py passes `fa_version=` to a kwarg named `version`); #79 open PR
  matches our teardown desync; a community 1-step LoRA card independently
  documents the per-block `context_noise` KV-refresh pass as un-distillable
  overhead. Repo is effectively unmaintained since the 3.0 drop.

## 2. Few-step distillation — 3→2 steps is proven and is the real 60 fps path

- **LongLive-2.0** (github.com/NVlabs/LongLive, Apache-2.0, base =
  Wan2.2-TI2V-5B — the SAME base as MG3). Verified from README: 24.8 FPS bf16
  → 29.7 FPS NVFP4 4-step → **45.7 FPS NVFP4 2-step** (VBench
  85.06→84.51→83.14) on H100. Recipe: TorchAO FP8 W8A8 row-wise PTQ, NVFP4
  W4A4 + NVFP4 KV cache via TransformerEngine, sequence parallel, async
  decode, torch.compile. Existence proof that our exact model family runs
  2 steps at ~46 fps. LongLive-1.0 (arXiv 2509.22622) trained long-video
  capability as a rank-256 LoRA in 32 H100-days.
- Re-distillation compute is small: Self-Forcing (arXiv 2506.08009,
  Apache-2.0) = 600 iters in <2 h on 64 H100s (~5.3 H100-days), data-free
  DMD. MG3's distillation is a descendant of this pipeline; re-running with a
  2-entry denoising_step_list is a fine-tune, not a retrain.
- Cheapest experiment: `kimhyunwoo/matrix-game-gta-1step-lora` (HF) — rank-16
  LoRA takes MG2's 3-step to 1-step: 2.08× measured speedup, cost = −16%
  sharpness, +56% temporal jitter; trained in 2.5 h on one 3090. A 2-step
  LoRA on MG3 is a days-scale experiment.
- **FastVideo** (hao-ai-lab/FastVideo) ships Matrix-Game 2.0 AND 3.0 Diffusers
  pipelines, a full DMD distillation recipe for MG2, and FastWan-QAD
  (quantization-aware DMD: 3-step CFG-free FP8/FP4 + SageAttention + TAEHV).
  A QAD re-distillation of MG3 is the most credible retrain-class route to
  60 fps at equal quality; all tooling exists in one framework.
- 1-step (Seaweed-APT, arXiv 2501.08316): works but −31–38% structural
  integrity and O(10^3) H100-days adversarial training. Not recommended.
- Quality-at-same-speed post-training: MoGAN (arXiv 2511.21592 —
  optical-flow discriminator on a 3-step DMD Wan student, literally our
  configuration; +13.3% VBench motion; no code). OPSD-V (arXiv 2607.08766,
  Apache-2.0) fixes long-rollout error accumulation without changing sampler,
  steps, or cache. Self-Forcing++ (arXiv 2510.02283) for minute-scale drift.
  Tencent WorldCompass (RL post-training for action-following, code released).

## 3. Kernels / quantization on H100

Priority order:

1. **FA3 `flash_attn_with_kvcache`** (Dao-AILab/flash-attention, hopper/):
   the only fast kernel with first-class KV-cache semantics — in-place cache
   update, paged KV, tensor-valued `cache_seqlens` (CUDA-graph-friendly), FP8
   e4m3 with descale. BF16 exact, 1.5–2× FA2 kernel-level; FP8 ~1.2 PFLOPs.
   Upstream MG3 already sets `--fa_version 3`. (Note: our audit sizes plain
   FA3 at only ~8–12 ms/chunk at 480p — the kvcache variant matters more as
   part of the compile/graph stack and at 720p.)
2. **torch.compile(fullgraph, max-autotune) + CUDA Graphs** over the whole
   3-step chunk denoise, with a statically-allocated max-size KV cache
   (ring-buffer writes, no concat). flux-fast (pytorch.org blog) got ~2.5×
   total on a DiT on H100 from compile+CUDAGraphs+FP8 linears+FA3-FP8.
3. **FP8 the FFN** (torchao per-row W8A8 on all linears): our int8 covers
   only attention projections. Tencent Yan measured 1.18× e2e from
   fp8-everything; ~parity with int8 on raw speed but unlocks cleaner compile
   fusion and often better quality.
4. **Tiny VAE: TAEHV `taew2_2`** (madebyollin/taehv) decodes the same Wan2.2
   latent space — drop-in A/B vs LightVAE-v2. MotionStream (arXiv
   2511.01266) went 16.7→29.5 FPS at 480p on one H100 just from a tiny-VAE
   swap.
5. **Hybrid Ulysses×Ring** (xDiT USP): measured 8×H100 Ulysses scaling is
   5.64×/8 (~30% comm overhead); ring degree 2 overlaps comm with compute.
6. **SageAttention2++**: on H100 only matches FA3-FP8 speed. Hard blockers in
   issues: wrong causal masks for decode shapes (#131), sm90 kernels escape
   CUDA-graph capture → corrupt replays (#371), pip 2.2.0 NaNs on H100 for
   Wan (#288/#320). Only if not on FA3-FP8; pinned source build; never under
   CUDA graphs.

SKIP: TeaCache/First-Block-Cache (needs 30–50 steps of redundancy; ceiling
≤1.33× at 3 steps); SVDQuant/Nunchaku (no Wan/H100 story); full TensorRT port
(TRT-LLM visual_gen covers stock Wan2.2, not custom AR/KV/action DiT; expect
10–25% over compiled PyTorch for weeks-months — harvest their ModelOpt FP8
calibration recipes instead). LightX2V (ModelTC) supports Matrix-Game-2.0 +
Self-Forcing with fp8/nvfp4 variants — worth mining configs.

## 4. Input latency & KV/context

- Root cause is chunk size: input applied once per 40-frame chunk = up to
  1.34 s input-to-effect. Industry norm: MirageLSD per-frame (<40 ms
  claimed), Odyssey ~40–50 ms/frame loop, Yan per-frame with shift-window
  denoising, MG2 chunk=3 latents. Fixes, ascending cost: (a) inject updated
  action embeddings into remaining denoise calls mid-chunk — days, hacky, no
  retraining; (b) LongLive-style "KV re-cache" on control change — ~200 ms
  per switch; (c) re-distill at smaller chunk (MG2 ran chunk=3 at 25 fps
  single-GPU) — retrain-class; (d) shift-window/diagonal denoising à la Yan —
  retrain.
- Per-chunk latency growth: consensus fix is fixed sliding window +
  attention sink. LongLive: 3-latent sink + 9-latent window = constant
  per-chunk cost, −28% compute. Consistent finding: SHORT windows beat long
  ones for drift (MG2's own ablation: 6-latent cache beats 9 — a free
  quality knob).
- Long-horizon memory without latency growth: **WorldKV** (arXiv 2605.22718,
  KAIST/NAVER) — training-free: evicted KV retained and re-inserted by
  camera/action correspondence; ~2× throughput vs full-KV; evaluated on
  Matrix-Game-2.0 and LingBot-World-Fast. Days-to-2-weeks, inference-only.
- Block Cascading (arXiv 2511.20426, training-free): ~2× throughput via
  temporal pipeline parallelism, but spends GPUs we give to Ulysses and
  worsens newest-input latency.

## 5. Frame interpolation + SR (60 fps without generating more frames)

- RIFE 4.x TensorRT: measured 426.8 fps @1080p (RIFE 4.6, RTX 4090,
  VSGAN-tensorrt-docker) → ~1–3 ms/frame at 832×480 (pixel-scaled estimate).
  Practical-RIFE 4.22.lite is explicitly recommended for diffusion-generated
  video.
- Latency accounting: interpolation normally adds ~1 frame of delay, but
  chunked generation means 39 of 40 next-frames already sit in the buffer —
  chunk-aware scheduling makes the added latency ~one 60 fps slot (~8 ms).
  Caveat: never improves input-to-photon latency; action response still
  quantizes to the generation clock.
- Deployment: g6e.xlarge sidecar (L40S, $1.86/hr = +3.4% on p5) runs RIFE +
  2× SR + NVENC (H100 has no NVENC). Or a few % of one H100 with MPS/stream
  pinning.
- SR 480p→960p: Real-ESRGAN-compact / SPAN via TRT ≈3–4 ms/frame at our
  input size.
- Ruled out: NVIDIA FRUC on H100 (no hardware OFA on Hopper), DLSS-FG
  (GeForce-only), FILM (not real-time).
- Quality risks: panning wobble on TRT RIFE variants, UI/HUD ghosting
  (composite the HUD after interpolation), chunk-seam judder (frame-repeat
  fallback).

## 6. Hardware

- **H200** (p5e $5.97/GPU-hr CB, p5en $6.87): MLPerf v4.1 SDXL 8×H200 vs
  8×H100 = +6.5% — diffusion here is compute-bound. Negative perf/$ for us;
  only relevant if VRAM-bound (141 GB for more concurrent sessions).
- **B200** (p6-b200.48xlarge, $12.36/GPU-hr CB = 2.38× p5's $5.19): dense
  FLOPs 2.27× H100 at bf16 AND fp8; MLPerf v5.0 SDXL 8×B200 ≈ 1.7–1.9× H100.
  FP4 up to 3–3.7× on LLMs but needs NVFP4 work. Software: CUDA 12.8+/sm_100,
  FlashAttention-4 (FA3 is Hopper-only), PyTorch ≥2.7. Chunk 1.34 s → ~0.7 s
  at bf16/fp8 recompile — **native 60 fps plausible on B200 without
  retraining**, at 2.38× cost. Recommendation: 1-day capacity block (~$2.4k)
  to measure real chunk time; gate a fleet move on ≥1.9×. Available in
  us-east-1/2, us-west-2 + Mumbai/Hyderabad via capacity blocks.

## 7. Production systems — what they publish

- **Decart MirageLSD**: <40 ms/frame, 24 fps, per-frame; Diffusion Forcing,
  history augmentation, Hopper megakernels with fused GPU-GPU comms,
  architecture-aware pruning, shortcut distillation. Lucy 2 (Jan 2026):
  30 fps 1080p claimed. Etched Sohu "4K": marketing, no benchmarks.
- **Tencent Yan** (arXiv 2508.08601): strongest published 1080p claims —
  1080p@50–60 FPS, 0.07 s latency via extreme-compression VAE (2×32×32 —
  collapses DiT token count; the biggest architectural lever in the
  literature, full retrain), 4-step + shift-window denoising, fp8 all GEMMs,
  CUDA graphs+compile, DiT and VAE on separate GPUs.
- **Tencent HY-WorldPlay** (Dec 2025, open-source): 24 FPS @480p 4-step AR;
  "Context Forcing" (aligns memory context between teacher/student during
  distillation — relevant when we re-distill MG3 with memory). Long-term
  PSNR 18.94 vs MG2's 9.57.
- **Genie 3** (DeepMind): 720p24, minutes-scale consistency — zero serving
  details, quality bar only. **World Labs RTFM**: single-H100 interactive,
  per-frame retrieval of posed frames from spatial memory. **Odyssey**:
  30 FPS, ~40 ms loop, $1–2/user-hour (only public unit-economics
  datapoint). **GameNGen**: context-4 ≈ context-64 in-domain — history
  length has sharply diminishing returns; 1-step DMD took them 20→50 fps.
- **Matrix-Game-3.5** (github.com/Riemann-Dynamics/Matrix-Game-3.5, arXiv
  2608.29910 — appears to be the MG team continued outside Skywork; NOT
  independently verified): 5B Wan2.2, 720p 3-step, single ≥40 GB GPU,
  parameter-free geometry memory ("Patch Memory" + warped RoPE). Head-to-head
  eval is a days-scale task and possibly the fastest quality upgrade. Also:
  Skywork 28B-MoE (2×14B viewpoint-specialized) "released soon."

## Ranked table (gain × certainty ÷ effort)

| # | Technique | Expected gain | Effort | Risk |
|---|---|---|---|---|
| 1 | Audit vs Skywork's 40 fps config: GPU retrieval + async VAE + LightVAE-v2@0.75 + FA3 flags | up to +50% fps if any piece is off | hours | none |
| 2 | TAEHV taew2_2 tiny-VAE A/B vs LightVAE-v2 | decode 4–6× faster | hours–days | mild softness; A/B |
| 3 | Static-max KV cache + torch.compile fullgraph + CUDA Graphs; kill CPU syncs | 1.2–1.5× | days | compile warmup mgmt |
| 4 | FA3 flash_attn_with_kvcache BF16→FP8 | attention 1.5–2×; e2e ~1.1–1.3× | days | FP8 artifacts — gate on eval |
| 5 | FP8 (torchao per-row) on FFN + all linears | ~1.15–1.2× e2e | days | low |
| 6 | RIFE 4.22.lite TRT interpolation 30→60 (+optional 2× SR), L40S sidecar | 2× displayed fps, +3.4% cost | days–2 wks | wobble/ghosting; input latency unchanged |
| 7 | 2-step re-distillation (Self-Forcing/DMD or LoRA; LongLive-2.0 = proof at 45.7 FPS) | native ~45 fps; NVFP4 path beyond | ~5–50 H100-days + eval | jitter/sharpness loss; mitigate w/ MoGAN/OPSD-V |
| 8 | Mid-chunk action injection + KV re-cache on control change | input-to-effect 1340 → ~200–450 ms | days | hacky; eval |
| 9 | Hybrid Ulysses×Ring SP (xDiT) | ~1.1–1.25× | days | low |
| 10 | WorldKV camera-keyed memory | ~2× vs full-KV; long-horizon consistency | days–2 wks | unvalidated preprint |
| 11 | KV window tuning (6 latents > 9 per MG2 ablation) | drift/quality ↑ free | days | low |
| 12 | B200 p6-b200 pilot | 1.7–2× recompile-only; ~3× w/ NVFP4 | days–2 wks | 2.38× $/GPU-hr; sm100 maturity |
| 13 | Quality post-trains (MoGAN/OPSD-V/Self-Forcing++/WorldCompass); eval MG3.5 | quality ↑ same speed | small retrain / days | MoGAN no code; MG3.5 unverified |
| 14 | Block Cascading temporal pipelining | ~2× throughput | weeks | worsens input latency |
| 15 | SKIP: H200, TeaCache@3steps, SVDQuant on H100, full TRT port, SageAttention (if FA3-FP8 lands), DLSS-FG/FRUC | — | — | documented above |

Suggested first sprint: items 1+2+11 (~2–3 days, zero training) while
spinning up item 7's 2-step LoRA experiment in parallel; item 6
(interpolation sidecar) as the guaranteed-60fps fallback if distillation
quality disappoints.
