# Troubleshooting

## Quick reference

| Symptom | Fix |
|---|---|
| `InsufficientInstanceCapacity` | GPU capacity is scarce and AZ-specific — see [GPU capacity](#gpu-capacity) below |
| Stack stuck in `ROLLBACK_COMPLETE` | Delete the stack manually, then redeploy |
| Model OOMs on L40S | Use an H100-class instance; many models need >48 GB VRAM |
| Weight staging: `No space left on device` | Peak disk is ~2× the repo size — see [Weight staging](#weight-staging) |
| Deploy launched an instance that serves nothing | Shouldn't happen: `deploy.sh` refuses to launch without an ECR image. If you changed inference code, set `REBUILD_IMAGE=1` — an unchanged image tag is reused |
| `Cloud assembly schema version mismatch … found 54.0.0` | Your **global** CDK CLI is older than the pinned `aws-cdk-lib` — see [CDK version](#cdk-version) |
| Endpoint returns 401 / WebSocket closes with 1008 | Auth is on by default. Pass the bearer token from SSM — see [Security](SECURITY.md) |
| Endpoint unreachable, no error | ALB ingress is locked to the deployer's IP. Re-run `./deploy.sh` from the same network, or set `ALLOWED_CIDR` |
| Real-time socket accepts, then no frames | The first session pays a `torch.compile` warm-up — see [Real-time sessions](#real-time-sessions) |
| Real-time socket closes with 1013 (black canvas) | A previous client left the socket open; it frees on the ALB idle timeout — see [Real-time sessions](#real-time-sessions) |
| Real-time fps far below the quoted number | Almost always your link, not the GPU — see [Real-time sessions](#real-time-sessions) |

## GPU capacity

The most common blocker, and usually not something you can fix in code.

Large GPU types are frequently unavailable. We have observed `p5.48xlarge` **and**
`g6e.12xlarge` both refusing to launch in every AZ this VPC spans, with no p5
[Capacity Block](https://aws.amazon.com/ec2/capacityblocks/) offerings for over a
week.

Options, cheapest first:

1. **Prove the path with a small instance.** `./deploy.sh echo-async` runs on a
   `g5.2xlarge` (~$1.20/hr, no weights) and exercises the whole deploy, auth and UI
   chain.
2. **Retry later.** The ASG retries across the VPC's AZs automatically.
3. **Try a smaller type** — `g6e.2xlarge`, `g5.2xlarge`.
4. **Deploy in another Region.**
5. **Buy a Capacity Block** — see below.

The VPC uses `maxAzs: 2`, so AZs beyond those two are unreachable without recreating
it.

Trying a different type needs no manifest edit — `INSTANCE_TYPE` overrides
`ec2.instance` for one deploy:

```bash
INSTANCE_TYPE=g5.2xlarge ./deploy.sh waypoint-1-5
```

Whether the model still fits is on you: the cartridge READMEs list the shapes each
one has actually been measured on.

Single-GPU H100 is scarce too: on-demand `p5.4xlarge` was unavailable across us-west-2
throughout this project's spike, and a Capacity Block was the only way to get one.
`g7e` (RTX PRO 6000 Blackwell) had no capacity and no Capacity Block offerings at all.
Plan on a block for anything p5, and treat g7e as unavailable until proven otherwise.

## Capacity Block Reservations

For predictable `p5.48xlarge` or `p5.4xlarge` launches, buy a reservation:

```bash
# 1. Find offerings
aws ec2 describe-capacity-block-offerings \
  --instance-type p5.48xlarge --instance-count 1 \
  --capacity-duration-hours 24 \
  --start-date-range <iso> --end-date-range <iso>

# 2. Purchase one → yields cr-...
aws ec2 purchase-capacity-block \
  --capacity-block-offering-id <id> \
  --instance-platform 'Linux/UNIX'

# 3. Deploy into it
CAPACITY_BLOCK_ID=cr-... ./deploy.sh lingbot-fast
```

Or pass it directly to CDK:

```bash
cd cdk && npx cdk deploy WorldModel-lingbot-fast \
  -c model=lingbot-fast -c target=ec2 -c capacity=cr-...
```

**Blocks bill from the block's start time whether or not your instance is running.**
Launch promptly.

Terminating a Capacity Block instance does **not** release the block — you are still
billed for the remaining window. Release it with
`aws ec2 cancel-capacity-reservation` if you finish early.

### Multi-GPU instance options

If `p5.48xlarge` is unavailable, these also give 8 GPUs. `torchrun` detects the count
from `nvidia-smi`, so no configuration change is needed beyond `ec2.instance` in the
manifest.

| Instance | GPUs | Total VRAM | ~$/hr | Notes |
|---|---|---|---|---|
| `p5.48xlarge` | 8× H100 | 640 GB | $31–55 | Primary target |
| `p5e.48xlarge` | 8× H200 | 1128 GB | $65 | More memory, similar FLOPS |
| `p4de.24xlarge` | 8× A100 80GB | 640 GB | $40 | Slower, often more available |
| `g6e.48xlarge` | 8× L40S | 384 GB | $20 | Budget; roughly half speed |
| `g5.48xlarge` | 8× A10G | 192 GB | $16 | May OOM at 720p |

Rates are approximate and region-dependent — check current pricing.

## Real-time sessions

Applies to `waypoint-1-5`, `matrix-game-3` and `echo-realtime` — the WebSocket
cartridges.

**The socket accepts but no frames arrive.** The first session after a container start
pays a `torch.compile` + CUDA-graph warm-up: **645 s** for `waypoint-1-5` on an H100,
~27 min on an A10G. `/health` answers long before that finishes, so an early client
sees a socket that connects and then goes quiet. Watch for the warm-up line in
`docker logs world-model` before connecting. Mounting `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` on the host makes the cost one-off across container restarts — that
is the fix.

`WAYPOINT_WARMUP=0` does **not** help here and makes the symptom worse. `torch.compile`
is lazy, so all it skips is the pre-emptive `gen_frame` at startup; the first client
then pays the identical compile inside their own session, with no log line marking when
it ends. Steady-state throughput is the same either way.

**A second client is refused, for up to about a minute.** These cartridges are
`max_concurrent: 1`. A client that vanishes without closing cleanly — a closed browser
tab, a Ctrl-C'd script — leaves the socket open through the ALB, so the endpoint counts
itself busy until the ALB idle timeout reaps the dead socket. Until then every new
socket is accepted and then immediately closed with `{"type":"error"}` and code 1013.
The catalogue UI only logs that to the browser console, so what you see is a black
canvas that looks like a dead endpoint. Wait it out and retry; do not redeploy.

**Delivered fps is far below the quoted figure.** The quoted numbers are *generation*
rate, measured on the instance. What a remote client sees is
`link Mbit/s ÷ (bytes-per-frame × 8 / 1e6)`: a 720p JPEG is ~73 KiB, which makes that
divisor ~0.60, so 60 fps needs ~36 Mbit/s
sustained all the way to wherever you are sitting. Check `Measured bitrate` against
`Link needed for 60 fps` in `./deploy.sh bench` output before concluding anything about
the GPU, and run the bench on the instance over loopback to measure the model instead
of the path. Lowering `WAYPOINT_JPEG_QUALITY`, deploying in the player's Region, or
switching to a video codec all move this number; a bigger GPU does not.

**fp8 silently becomes int8.** `waypoint-1-5` requests `fp8w8a8`, which needs compute
capability ≥ 8.9. On Ampere (A10G, 8.6) the runner downgrades to `intw8a8` and logs it
rather than failing — so an unexpectedly low fps on a `g5` is expected, not a fault.

## Weight staging

`hf download --local-dir` keeps a copy in the Hugging Face cache **and** in the
target directory, so peak disk is roughly twice the repo size. `lingbot-fast` (~234
GB) exhausted a smaller CodeBuild volume partway through.

The `world-model-weights` project therefore runs on `X2_LARGE` with a 4-hour timeout,
and the buildspec prunes the cache as it goes. A model larger than ~200 GB may need a
bigger compute type — set it in `cdk/lib/shared.ts`.

Gated repos need `HF_TOKEN` exported before running `stage-weights.py`.

## CDK version

If a **global** `cdk` is older than the pinned `aws-cdk-lib`, synth fails with a
schema mismatch. Either use the repo-local CLI:

```bash
cd cdk && npx cdk synth      # what ./deploy.sh already does
```

…or upgrade the global one (`npm install -g aws-cdk@2.1134.0`, possibly with `sudo`).
Tools that shell out to a global `cdk` — e.g. `dsr assess` — need the upgrade.

## Debugging a running deployment

```bash
./deploy.sh status                    # endpoint URLs + CloudFormation stacks
```

On the instance (reachable via SSM Session Manager — it has no public IP):

```bash
cat /var/log/world-model-startup.log  # boot: docker pull, weight sync, docker run
docker logs world-model               # torchrun, rank startup, request logs
nvidia-smi                            # GPU visibility and memory
```

Common boot failures, in order of likelihood: `docker pull` denied (image missing or
ECR permissions), `aws s3 sync` finding no weights at the expected prefix, and the
model OOMing during `setup()` on a GPU too small for it.

For CodeBuild failures the deploy scripts print the log deep-link on exit; otherwise:

```bash
aws codebuild batch-get-builds --ids <build-id> \
  --query 'builds[0].logs.deepLink' --output text
```

## Cleanup

```bash
./deploy.sh destroy <model>
```

The artifacts and outputs buckets are `RETAIN`, so staged weights and generated
outputs survive teardown — a later redeploy syncs from S3 rather than re-downloading
from Hugging Face. Delete them by hand if you want the storage back.
