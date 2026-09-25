# World Model Inference

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

One-click deployment for open-source **world models** on AWS. Each model is a
self-contained "cartridge" under `inference/models/<id>/` — the infrastructure is
one parameterised CDK stack, identical for every model, so adding a model means
adding a directory rather than writing new infrastructure.

The flagship endpoint is **`lingbot-fast`**, an 8-GPU video generator built on
[LingBot-World-Fast](https://github.com/robbyant/lingbot-world).

## Before you deploy

1. **AWS credentials** — however you normally configure them.
2. **Tooling** — `aws` CLI v2, `node` ≥ 18, `npm`.
3. **IAM permissions** — to create EC2 instances, IAM roles, S3 buckets, security
   groups, and (for SageMaker) endpoints. CDK creates everything via CloudFormation.
4. **GPU capacity** — the most common blocker, and it is AZ-specific. To prove the
   deploy path works first, start with `echo-async` on a `g5.2xlarge` (~$1.20/hr, no
   weights). See [Troubleshooting](docs/TROUBLESHOOTING.md#gpu-capacity).

## Quick start

```bash
git clone <this-repo> && cd world-model-inference

./deploy.sh                          # lingbot-fast on EC2 (default)
./deploy.sh cosmos3-nano             # a different model
./deploy.sh cosmos3-nano sagemaker   # deploy to SageMaker instead
```

`./deploy.sh` handles CDK bootstrap, dependency installation, image build and stack
creation.

```bash
./deploy.sh list                     # available models
./deploy.sh status                   # active deployments + endpoint URLs
./deploy.sh ui                       # catalogue UI at localhost:3000
./deploy.sh bench waypoint-1-5 60    # performance report for a real-time endpoint
./deploy.sh destroy lingbot-fast     # tear down — p5.48xlarge is ~$55/hr!
```

Models with weights need them staged to S3 once before the first deploy — see the
cartridge's own README, or `python scripts/stage-weights.py --help`.

Hitting an error? See **[Troubleshooting](docs/TROUBLESHOOTING.md)**.

## Endpoints

| ID | Type | Mode | Instance | Wired? |
|----|------|------|----------|--------|
| [`waypoint-1-5`](inference/models/waypoint-1-5/README.md) | real-time | real-time | `p5.4xlarge` · 1× H100 | ✅ end-to-end · **82 fps at 720p** |
| [`matrix-game-3`](inference/models/matrix-game-3/README.md) | real-time | real-time | `p5.48xlarge` · 8× H100 | ✅ end-to-end · ~35 fps at 832×480 |
| [`lingbot-fast`](https://github.com/robbyant/lingbot-world) | video generation | async | `p5.48xlarge` · 8× H100 | ✅ end-to-end |
| [`cosmos3-nano`](https://github.com/NVIDIA/Cosmos) | video generation | async | `g6e.12xlarge` · 4× L40S | ✅ end-to-end |
| [`lyra-2`](https://github.com/nv-tlabs/lyra) | image→3D-world video | async | `p5.4xlarge` · 1× H100 | ✅ end-to-end |
| [`vjepa2`](https://github.com/facebookresearch/vjepa2) | representation | real-time | `g5.2xlarge` · 1× A10G | ⚠️ encoder only |

The two real-time cartridges have their own READMEs covering the session
protocol, hardware trade-offs and tuning knobs. For both, **the fps a remote
player sees is set by their link, not by the GPU** — a 720p frame is ~73 KiB, so
60 fps needs ~36 Mbit/s sustained.

`echo-async` and `echo-realtime` are cheap no-weights cartridges for verifying a
deployment end to end. Details and caveats: [Endpoints](docs/ENDPOINTS.md).

## Architecture

![World Model Accelerator architecture on AWS. The developer runs ./deploy.sh, which builds the container image and stages model weights via two AWS CodeBuild projects, pushing the image to Amazon ECR and the weights to an Amazon S3 artifacts bucket. AWS CDK then deploys a VPC spanning two availability zones: an Application Load Balancer in the public subnet is the only public entry point, forwarding port 8080 to a single GPU instance in a private subnet with no public IP. The instance pulls its image from ECR, syncs weights from S3 over an S3 gateway endpoint, and runs torchrun — rank 0 serves FastAPI while ranks 1 to N-1 follow in lockstep. Requests are authenticated against an Amazon Cognito user pool. Six interchangeable model cartridges plug into the same stack. Amazon SageMaker is an optional alternate deploy target.](docs/diagrams/architecture-v1-cartridge-platform.png)

`./deploy.sh` builds the container image and stages weights via two CodeBuild
projects, then CDK deploys a two-AZ VPC: an Application Load Balancer is the only
public entry, forwarding `:8080` to a GPU instance in a private subnet. On the
instance, `torchrun` runs one process per GPU — rank 0 serves FastAPI and a job
queue, ranks 1..N-1 follow, and each job is broadcast so every GPU runs the same
step in lockstep.

Full walkthrough: **[Architecture](docs/ARCHITECTURE.md)**.

## Cost

| Instance | Rate | Notes |
|---|---|---|
| `p5.48xlarge` | ~$31–55/hr | 8× H100. **Destroy when done.** |
| `g6e.12xlarge` | ~$10/hr | 4× L40S |
| `p5.4xlarge` | ~$8/hr | 1× H100 — usually needs a Capacity Block |
| `g5.2xlarge` | ~$1.20/hr | 1× A10G — cheapest smoke test |

Weight storage is one-off: `lingbot-fast` ~234 GB (~$5.40/mo), `cosmos3-nano`
~35 GB, `waypoint-1-5` ~12 GB, `vjepa2` ~4 GB. Later launches sync from S3
instead of Hugging Face.

## Security

`./deploy.sh` restricts ALB ingress to your own public IP and provisions an Amazon
Cognito user pool, so the endpoint is authenticated by default — the service verifies
Cognito access tokens and holds no credential scheme of its own. The GPU instance
sits in a private subnet with no public IP. The ALB serves **plain HTTP** unless you
supply an ACM certificate via `CERTIFICATE_ARN`.

Before exposing anything to untrusted networks, read
**[Security](docs/SECURITY.md)**. To report a vulnerability, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

| Doc | Contents |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Build/serve/run walkthrough, project layout, runtime topology |
| [Adding a model](docs/ADDING_A_MODEL.md) | The cartridge contract, container build paths, `endpoint.yaml` keys |
| [Endpoints](docs/ENDPOINTS.md) | Per-model detail, instance sizing, known gaps |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Failure modes, GPU capacity, Capacity Block Reservations |
| [Security](docs/SECURITY.md) | Auth, network posture, hardening for production |
| [Model licences](docs/MODEL_LICENSES.md) | Per-cartridge weights/code licences and the deploy-time gate |

## License

Apache 2.0 — see [LICENSE](LICENSE). Includes code from
[LingBot-World](https://github.com/robbyant/lingbot-world) and
[SAM 2](https://github.com/facebookresearch/sam2), both Apache 2.0; see
[NOTICE](NOTICE) for attribution.

**The models are licensed separately, and not all of them permit commercial use.**
This repo being Apache 2.0 says nothing about the weights it downloads at deploy time.
Before you deploy anything, read **[Model licences](docs/MODEL_LICENSES.md)** — it
records the weights licence, the upstream code licence, and what each one restricts,
per cartridge. `lyra-2` in particular is internal research only, and `deploy.sh`
will refuse to deploy it until you acknowledge that.
