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
./deploy.sh destroy lingbot-fast     # tear down — p5.48xlarge is ~$55/hr!
```

Hitting an error? See **[Troubleshooting](docs/TROUBLESHOOTING.md)**.

## Endpoints

| ID | Type | Mode | Instance | Wired? |
|----|------|------|----------|--------|
| [`lingbot-fast`](https://github.com/robbyant/lingbot-world) | video generation | async | `p5.48xlarge` · 8× H100 | ✅ end-to-end |
| [`cosmos3-nano`](https://github.com/NVIDIA/Cosmos) | video generation | async | `g6e.12xlarge` · 4× L40S | ✅ end-to-end |
| [`lyra-2`](https://github.com/nv-tlabs/lyra) | image→3D-world video | async | `p5.4xlarge` · 1× H100 | ✅ end-to-end |
| [`vjepa2`](https://github.com/facebookresearch/vjepa2) | representation | real-time | `g5.2xlarge` · 1× A10G | ✅ end-to-end |
| [`matrix-game-3`](https://github.com/SkyworkAI/Matrix-Game-3.0) | real-time | real-time | `g5.2xlarge` (placeholder) | ❌ scaffold only |

`echo-async` and `echo-realtime` are cheap no-weights cartridges for verifying a
deployment end to end. Details and caveats: [Endpoints](docs/ENDPOINTS.md).

## Architecture

![World Model Accelerator architecture on AWS. The developer runs ./deploy.sh, which builds the container image and stages model weights via two AWS CodeBuild projects, pushing the image to Amazon ECR and the weights to an Amazon S3 artifacts bucket. AWS CDK then deploys a VPC spanning two availability zones: an Application Load Balancer in the public subnet is the only public entry point, forwarding port 8080 to a single GPU instance in a private subnet with no public IP. The instance pulls its image from ECR, syncs weights from S3 over an S3 gateway endpoint, reads its API token from AWS Systems Manager, and runs torchrun — rank 0 serves FastAPI while ranks 1 to N-1 follow in lockstep. Eight interchangeable model cartridges plug into the same stack. Amazon SageMaker is an optional alternate deploy target.](docs/diagrams/architecture-v1-cartridge-platform.png)

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
| `g5.2xlarge` | ~$1.20/hr | 1× A10G — cheapest smoke test |

Weight storage is one-off: `lingbot-fast` ~234 GB (~$5.40/mo), `cosmos3-nano`
~35 GB, `vjepa2` ~4 GB. Later launches sync from S3 instead of Hugging Face.

## Security

`./deploy.sh` restricts ALB ingress to your own public IP and provisions a shared
bearer token in SSM, so the endpoint is authenticated by default. The GPU instance
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
| [Pattern specification](docs/ACCELERATOR_PATTERNS.md) | Design spec (draft, April 2026) — aspirational, not shipped |
| [Platform vision](docs/WORLD_MODELS_PLATFORM.md) | Where this is heading beyond what ships today |

## License

Apache 2.0 — see [LICENSE](LICENSE). Includes code from
[LingBot-World](https://github.com/robbyant/lingbot-world) and
[SAM 2](https://github.com/facebookresearch/sam2), both Apache 2.0; see
[NOTICE](NOTICE) for attribution.
