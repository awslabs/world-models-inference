# Praxis-1

Runway's **world action model**: a video-action robot policy. Each call predicts the
next chunk of robot actions *and* the video of what those actions will do. This
cartridge runs the released checkpoints in closed loop in MuJoCo (upstream's
`praxis-sim`) and returns what happened.

- **Upstream:** https://github.com/runwayml/praxis-1-oss (Apache-2.0), pinned to
  `2042edf6f5512294f9242636ee1ce235153b96f0` in the `Dockerfile`
- **Weights:** `runwayml/praxis-1-droid` (single arm), `runwayml/praxis-1-yam`
  (bimanual, optional) — Apache-2.0, **gated** on Hugging Face (accept the terms)
- **Frozen deps:** `Wan-AI/Wan2.2-TI2V-5B` (`Wan2.2_VAE.pth`), `Qwen/Qwen3.5-2B` —
  Apache-2.0
- **GPU:** one, with at least 80 GB (H100 80 GB, RTX PRO 6000 96 GB)

> **Status: deploy-tested end to end on EC2 (2026-11-25).** `./deploy.sh` → image
> 0.1.2 → a g7e.12xlarge (RTX PRO 6000, one of its two GPUs used) → jobs run from the
> catalogue page on both rigs. SageMaker has not been run. See [Open items](#open-items).

## What a job does

An async job: one task, one episode per seed.

| Parameter | Default | Meaning |
|---|---|---|
| `rig` | `droid` | `droid` (Franka + Robotiq, `scripts/run_sim.py`) or `yam` (two YAM arms, `scripts/run_sim_yam.py`) |
| `task` | `lift_cube` / `dice_in_bag` | a task of the rig's suite (below) |
| `prompt` | task default | replaces the instruction; the task's success check still runs |
| `seeds` | `[0]` | scene seeds (object placement; the sampler seed follows), up to 8; `"0,1"` on the form route |

DROID tasks: `lift_cube`, `cube_on_plate`, `block_in_bowl`, `block_out_of_bowl`,
`move_block`, `stack_block`, `press_button`, and the harder `cube_in_tray`,
`cube_on_color`, `pick_color_block`, `color_block_in_bowl`, `stack_color_blocks`,
`peg_in_hole_6mm`, `block_in_small_bowl`, `cube_on_small_plate`, `stack_small_block`,
`two_blocks_in_bowl`, `stack_three_blocks`. Each runs with its own default time limit
(as upstream's suite runner does).

YAM tasks: `dice_in_bag`, `tennis_ball_in_bag`, `apple_in_bag`, `bell_in_bag`.

**Output.** The job result (`/jobs/{id}/output`, or the SageMaker async output) is
**one MP4**: every episode back to back. Each frame shows the rendered sim views
(DROID: overview, exterior, wrist; YAM: top and both wrists) with the video the
model predicted underneath and a status band.

Next to it, in the container, the runner leaves `<job>/output/`:

```
episode_000.mp4 ...    one per seed (YAM: <task>/episode_000.mp4)
summary.json           {"rig", "task", "prompt", "seeds", "successes",
                        "per_episode": [{"seed", "success", "video"}]}
upstream_summary.json  upstream's own summary (call times, lift/reach metrics)
rollout.log            the child process's log
```

`GET /result/<job_id>/summary` (alias `/jobs/<job_id>/summary`) returns that
`summary.json` once the job is complete.

## Example

```bash
# EC2 (/generate is a multipart form; seeds comma-separated)
curl -sS -H "Authorization: Bearer $TOKEN" \
  -F task=block_in_bowl -F seeds=0,1 https://<endpoint>/generate
# -> {"job_id": "...", "status": "queued"}; poll /jobs/<id>, then GET /jobs/<id>/output

# /invocations (SageMaker async) takes JSON
{"rig": "droid", "task": "stack_block", "seeds": [0, 1, 2]}
```

## UI

The catalogue opens Praxis-1 on its own page (`frontend/src/components/PraxisRunner.tsx`):
pick the rig (single arm or two arms), a task from the suite and up to 8 seeds, optionally
replace the instruction, and run. The page shows the concatenated episodes (the sim
cameras above the video the model imagined) and a success/missed badge per seed, read
from `GET /result/<job_id>/summary` (the `summary.json` the runner writes beside the MP4;
same auth and owner scoping as `/result`).

## Deploy

```bash
# 1. Accept the terms of runwayml/praxis-1-droid (and -yam) on Hugging Face.
# 2. Stage the weights (HF_TOKEN is passed to CodeBuild through Secrets Manager
#    and cleared afterwards; it is never written to the repo or the image).
export HF_TOKEN=<your Hugging Face read token>
python scripts/stage-weights.py praxis-1 --source hf:runwayml/praxis-1-droid --subdir praxis-1-droid \
  --revision 6bb70750663e53592cc38f46e531a5a01a12d291
python scripts/stage-weights.py praxis-1 --source hf:Wan-AI/Wan2.2-TI2V-5B --subdir wan2_2 \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e
python scripts/stage-weights.py praxis-1 --source hf:Qwen/Qwen3.5-2B --subdir qwen3_5-2b \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc
# optional, for rig=yam:
python scripts/stage-weights.py praxis-1 --source hf:runwayml/praxis-1-yam --subdir praxis-1-yam \
  --revision b72c4bbc0430d40ef9bc26c49ae6d9773298ee07
unset HF_TOKEN

# 3. Deploy (p5.4xlarge, 1x H100 80 GB)
./deploy.sh praxis-1
```

Staged size is ~148 GB with the YAM checkpoint (~94 GB without). The whole
Wan2.2-TI2V-5B repo (~34 GB) is staged although only `Wan2.2_VAE.pth` (2.8 GB) is
read, because `stage-weights.py` has no per-file filter yet.

The Launch Stack has no Hugging Face token parameter, so it cannot stage these gated
weights.

## How it runs

`setup()` only checks that the upstream checkout and the weights are in place; it
never touches the GPU. Each job starts upstream's own script as a child process with
fixed arguments (no shell; every user value is validated, and a prompt is passed as a
JSON string so it can only ever be a string), the torchrun variables removed, and
`MUJOCO_GL=egl`. The child owns the GPU for the job and exits when it finishes, so a
crash or a CUDA error never outlives the job. Upstream's kernel cache
(`PRAXIS_CACHE_DIR`, under `/tmp`) keeps later jobs from recompiling; the first job
on a fresh container pays for compile and CUDA-graph capture.

Timeouts: `PRAXIS_BASE_TIMEOUT_S` (default 1200) plus `PRAXIS_EPISODE_TIMEOUT_S`
(default 600) per seed.

### Image patches

- **MoE GEMM shared memory** (`patch_moe_smem.py`). Upstream's Triton MoE GEMMs use
  three pipeline stages, which need more shared memory per block than GPUs with
  101,376 B have (RTX PRO 6000, L40S class), and fail with `OutOfResources`. Below
  147,456 B the image caps them at two. Only the pipelining depth changes; tiles and
  arithmetic are the same. H100 has room for three and is unaffected.
- **Robot models baked in.** The MuJoCo Menagerie files (Franka Panda Apache-2.0,
  Robotiq 2F-85 BSD-2-Clause, I2RT YAM MIT) are fetched at a pinned commit and
  sha256-checked at build time, and runs use `scene.download=false`.

## Measured (raw upstream sim, not this endpoint)

On a g7e.2xlarge (RTX PRO 6000 Blackwell 96 GB, CUDA 13 driver), DROID checkpoint,
two seeds per task, with `sim_droid.yaml`'s 20.6 s time limit: `lift_cube` 2/2,
`press_button` 2/2, `block_in_bowl` 1/2, `stack_block` 1/2, `move_block` 1/2 (7/10).
About 4-5 minutes per task including compile and CUDA-graph capture. This cartridge
uses each task's own default time limit instead, as upstream's suite runner does, so
results can differ (`lift_cube` and `press_button` default to 11 s).

## Open items

- **Host driver (handled).** The image is CUDA 13 (torch cu130) and needs an R580+
  driver, newer than the repo's default PyTorch 2.7 / Ubuntu 22.04 DLAMI, so
  `endpoint.yaml` sets `ec2.ami_ssm_parameter` to the Base OSS Ubuntu 24.04 AMI.
- **g7e.2xlarge** is where it was validated, but the catalogue's price table and
  `scripts/capacity.py` have no entry for it, so the default is p5.4xlarge.
- **SageMaker** (`./deploy.sh praxis-1 sagemaker`) has not been run; EC2 has.
- Deploy-test results, droid: lift_cube 2/2, block_in_bowl 1/2; yam: tennis_ball_in_bag
  2/2, apple_in_bag 2/2. Submit → result about 70–100 s for two seeds once warm.
