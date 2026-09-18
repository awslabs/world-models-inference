# World Foundry — "Bedrock for World Models"

> **Status:** Design / vision doc. Reality today lives in `README.md` and
> `inference/models/`. Anything past §3 is forward-looking unless noted.

A reusable accelerator where any world model can be dropped in as a **cartridge**
(one directory under `inference/models/`) and served behind a uniform API on
Amazon SageMaker. Open-source friendly.

---

## 1. Motivation

Open-source world models are appearing faster than the tooling to run them. Each
ships its own launch script, dependencies, and API, and the larger ones need
several GPUs in lockstep — so most of the work between "here are some weights"
and "here is a running endpoint" gets redone per model. There is no managed
service for this, so a reusable deployment pattern is worth having.

**Goals:**
- Make it easy to deploy world models and run inference.
- Establish a pattern where a new model = a new directory, nothing more.
- Encourage collaboration across teams and partners.

**Target use cases:** Robotics, gaming, simulation, self-driving cars,
synthetic data generation.

---

## 2. What ships today

Production-ready mechanism, model bodies still being wired:

- **`inference/models/<id>/`** — one directory per endpoint.
  `endpoint.yaml` + `Dockerfile` + `handler.py` + `requirements.txt`. Directory
  name **is** the endpoint id. No central registry file.
- **`inference/lib/`** — two handler bases:
  - `AsyncHandler` for `mode: async` endpoints (HTTP request/response, JSON).
  - `RealtimeHandler` for `mode: real-time` endpoints (SageMaker AI bidi
    streaming on `ws://localhost:8080/invocations-bidirectional-stream`).
- **`cdk/`** — CDK stack that globs `inference/models/*/endpoint.yaml` and
  creates one SageMaker endpoint per file. No per-endpoint CDK code.
- **`deploy.sh` / `scripts/build-container.sh` / `scripts/test-endpoint.py`** —
  all data-driven off the same YAML. Adding an endpoint never touches these.

See [`ACCELERATOR_PATTERNS.md`](./ACCELERATOR_PATTERNS.md) for the full
inference-API spec (pattern 1 `real-time`, pattern 2 `async-3d`,
pattern 3 `async-video`).

### Model archetypes served today

The platform uses a two-mode split (`real-time` / `async`). Pattern-level
archetypes from `ACCELERATOR_PATTERNS.md` map onto the deployment modes as:

| Archetype | `mode:` | Interaction | Output |
|-----------|---------|-------------|--------|
| `real-time` (human) | `real-time` | WebSocket session, WASD/gamepad | JPEG frame stream |
| `real-time` (agent) | `real-time` | WebSocket session, per-step action payload | features / risk scores / frames |
| `async-video`       | `async`     | POST prompt + camera path, poll for MP4 | `.mp4` |
| `async-3d`          | `async`     | POST prompt + cameras, poll for artifacts | `.ply` / `.spz` + preview `.mp4` |

---

## 3. The cartridge contract

Every endpoint is a directory whose name is its id:

```
inference/models/<id>/
├── endpoint.yaml      # deployment policy (image, instance, mode, scale, model_data)
├── Dockerfile         # real-time: ENTRYPOINT ["python","-m","lib.serve"] + bidi label
│                      # async:     SageMaker Inference Toolkit defaults
├── handler.py         # subclass RealtimeHandler or AsyncHandler; call register_handler(me)
└── requirements.txt
```

**`endpoint.yaml` shape** (deployment policy only — model identity is the
directory name):

```yaml
image: world-model-<id>:0.1.0          # ECR repo:tag or full URI
instance: ml.p5.48xlarge
mode: real-time                         # real-time | async
scale: { min: 1, max: 4 }
model_data: s3://world-models-artifacts/<id>/   # S3Prefix ending in '/'

# required if mode: async
async:
  output: s3://world-models-output/<id>/
  max_concurrent_invocations: 2
```

Weights are staged in S3 (JumpStart pattern). SageMaker streams the S3Prefix
into `/opt/ml/model/` at cold start. Swapping weights doesn't rebuild the image.

---

## 4. Future work (aspirational)

The following are platform extensions — not yet built:

### 4.1 Routing layer
A **Fargate router** in front of the SageMaker endpoints to do:
- Cognito JWT validation.
- Session/quota tracking in DynamoDB.
- Matchmaking for real-time sessions (pin a session to a warm instance).
- A uniform client-facing API that abstracts over `real-time` vs `async`
  differences.

Without the router, clients call SageMaker endpoints directly
(`InvokeEndpointWithBidirectionalStream` / `InvokeEndpointAsync`).

### 4.2 EC2 tinkering backend
SageMaker endpoints are the production path. For research / rapid iteration
an **EC2 fleet** variant could mirror the same Dockerfile on a per-model ASG
with full SSM access. The image is already identical — this is a CDK addition
only.

### 4.3 Session recording
S3 + MediaConvert to let users save a `real-time` run as an MP4.

### 4.4 Fine-tuning / LoRA
Today the platform is inference-only. Hosting per-user LoRA adapters would
need its own mechanism (weight mount + routing).

### 4.5 Catalogue UI integration
`frontend/` currently lists models statically. It should read from
`inference/models/*/endpoint.yaml` (or a generated index) so the catalogue
stays in sync automatically.

---

## 5. Open questions

- **Cold start** — H100 endpoints take ~3–5 min to warm. `scale.min=0` (pay
  per invoke) vs `scale.min=1` (always warm) is a per-endpoint call.
- **Multi-tenant on one instance** — for small models we could shard multiple
  real-time sessions per GPU. Needs per-session CUDA stream isolation.
- **3DGS viewer** — Gaussian Splat rendering is client-side WebGL. How do we
  serve `.ply` files efficiently from the async endpoint output bucket?
- **Recording rights** — if a cartridge wraps third-party weights, who owns
  the generated output?
- **Open-source scope** — what's the boundary of open-source vs internal?

---

## 6. Files

- [`../README.md`](../README.md) — quick start and endpoint list.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — the shipping architecture, code-accurate.
- [`ADDING_A_MODEL.md`](./ADDING_A_MODEL.md) — the cartridge contract as implemented.
- [`ENDPOINTS.md`](./ENDPOINTS.md) — per-model detail and known gaps.
- [`ACCELERATOR_PATTERNS.md`](./ACCELERATOR_PATTERNS.md) — the April 2026 inference-API
  design spec (REST + WebSocket protocols, runner protocols). A draft that predates
  the implementation; its three-pattern taxonomy
  (`real-time` / `async-3d` / `async-video`) is **not** what ships — the code has two
  modes, `async` and `real-time`.
- `archive/world-foundry-vision-diagram.drawio` / `.png` — block diagram of the
  target platform (includes routing/EC2-fleet pieces from §4, not all shipped).
  This is the forward-looking vision, archived to avoid confusion with the
  shipping design.
