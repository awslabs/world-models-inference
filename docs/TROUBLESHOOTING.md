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

> Instance type is manifest-only today, so working around a shortage means editing
> `endpoint.yaml`. An `INSTANCE_TYPE` env override would be a useful addition.

## Capacity Block Reservations

For predictable `p5.48xlarge` launches, buy a reservation:

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
