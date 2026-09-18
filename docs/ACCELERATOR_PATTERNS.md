# World Model Inference — Pattern Specification

> **Version:** 0.1.0 · **Date:** 2026-04-20 · **Status:** Draft
>
> ⚠️ **This is a design document, not a description of what ships.** It predates the
> implementation and diverges from it in ways that matter:
>
> - It defines three patterns (`real-time`, `async-3d`, `async-video`). The code has
>   **two** modes, `async` and `real-time`, derived from whether `sagemaker.output`
>   is set in `endpoint.yaml`. `async-3d` and `async-video` appear nowhere in the
>   codebase.
> - Its manifest schema (top-level `image:`, `instance:`, `pattern:`, `hf_repos:`)
>   is not the shipped schema. See [Adding a model](ADDING_A_MODEL.md).
> - Its model catalogue lists endpoints that do not exist in this repo
>   (`flashworld`, `multiworld-game`, `multiworld-robots`,
>   `interactive-world-sim`, `vjepa2-predict`).
>
> Kept for the API and runner-protocol thinking, which informed the design, and as a
> record of the intended direction. **For current behaviour see
> [Architecture](ARCHITECTURE.md), [Adding a model](ADDING_A_MODEL.md) and
> [Endpoints](ENDPOINTS.md).**

The accelerator serves any world model behind a uniform API. Every model declares which **pattern** it implements. The router dispatches accordingly. This document is the contract between model authors, the API layer, and client applications.

---

## Table of Contents

1. [Patterns Overview](#1-patterns-overview)
2. [Pattern 1: `real-time` — Stateful Step Loop](#2-pattern-1-real-time)
3. [Pattern 2: `async-3d` — Prompt → 3D Scene + Video](#3-pattern-2-async-3d)
4. [Pattern 3: `async-video` — Prompt → Video](#4-pattern-3-async-video)
5. [Model Registry Manifest](#5-model-registry-manifest)
6. [REST API Specification](#6-rest-api-specification)
7. [Runner Protocol (Python)](#7-runner-protocol-python)
8. [Model Catalogue](#8-model-catalogue)
9. [Use Case Matrix](#9-use-case-matrix)

---

## 1. Patterns Overview

| # | Pattern | Protocol | Lifecycle | Input | Output |
|---|---------|----------|-----------|-------|--------|
| 1 | **`real-time`** | WebSocket | Stateful session (minutes) | Action per tick | Observation per tick |
| 2 | **`async-3d`** | REST async | Stateless job (seconds–minutes) | Image + prompt + cameras | Video (.mp4) + 3D (.ply/.spz) |
| 3 | **`async-video`** | REST async | Stateless job (seconds–minutes) | Image + prompt + camera path | Video (.mp4) |

Pattern 2 and Pattern 3 share the same REST async protocol. Pattern 3 is a subset of Pattern 2 (no 3D reconstruction output). A `async-3d` model can serve Pattern 3 requests by skipping the reconstruction step.

---

## 2. Pattern 1: `real-time`

### Interaction Model

Long-lived WebSocket session. Client sends actions, server responds with observations. The model maintains internal state (latent history, VAE cache) across steps.

```
Client                        Server
  │                              │
  │──POST /v1/sessions──────────►│  Create session
  │◄─── {session_id, ws_url} ───│
  │                              │
  │══WS /v1/sessions/{id}/ws═══►│  Open WebSocket
  │                              │
  │──── {type: "reset", ...} ──►│  Initialize with seed image
  │◄─── {type: "observation"}───│  First observation
  │                              │
  │──── {type: "step", ...} ───►│  Send action (every frame)
  │◄─── {type: "observation"}───│  Receive observation
  │      ... repeats at FPS ...  │
  │                              │
  │──── {type: "step",          │  Step with extra fields (e.g. new prompt)
  │       prompt: "..."} ──────►│  Model picks up changes at next chunk
  │◄─── {type: "observation"}───│  Observation reflects new prompt
  │      ... repeats ...         │
  │                              │
  │──── {type: "close"} ───────►│  End session
  │◄─── {type: "closed"} ──────│
  │══════════════════════════════│  WebSocket closed
```

### Action Spaces

The `action_space` declared in the model manifest determines who is driving the session. Two action spaces are defined:

#### `human` — Human in the loop

A human controls the world through physical input devices — keyboard, mouse, gamepad, touchscreen. The action shape is **known and fixed**: discrete keys + optional continuous pointer. The frontend renders game-style controls.

Used by: Matrix Game 2, Matrix Game 3

```jsonc
// Step message (client → server)
{
  "type": "step",
  "timestamp": 1713625200000,
  "action": {
    "keyboard": [0, 1, 0, 0],       // Sparse one-hot: [forward, back, left, right]
                                      // Matrix Game 3 templerun: [nomove, jump, slide, turnleft, turnright, leftside, rightside]
    "mouse": [0.1, 0.0]              // [dy, dx] normalized camera delta. Omit for modes without mouse.
  }
}
```

**Use cases:** Interactive gaming, driving simulators, walkthroughs, demos — anything where a person is playing.

#### `agent` — Machine in the loop

A program, robot, sensor pipeline, or AI agent sends data to the model. The action payload is **model-defined** — the accelerator passes it through without inspection. This covers two fundamentally different interaction modes under one protocol:

**Active agents** send actions that *cause* the next world state (robot moves → world changes):

```jsonc
// Robot arm (V-JEPA 2-AC) — 7-DOF joint deltas
{
  "type": "step",
  "timestamp": 1713625200000,
  "action": {"values": [0.01, -0.02, 0.0, 0.0, 0.0, 0.0, 0.5]}
  //                    dx     dy     dz   droll dpitch dyaw  dgripper
}

// Humanoid (22-DOF) — all joint positions
{"type": "step", "action": {"values": [0.1, -0.05, ...]}}   // 22 floats

// Drone (4-DOF) — thrust and orientation
{"type": "step", "action": {"values": [0.8, 0.01, -0.02, 0.0]}}  // thrust, roll, pitch, yaw

// Mobile base (2-DOF) — velocity commands
{"type": "step", "action": {"values": [1.5, 0.3]}}  // linear_vel, angular_vel
```

**Passive agents** send observations — the model *analyzes* the world rather than changing it:

```jsonc
// Dashcam feed (BADAS-style risk prediction)
{
  "type": "step",
  "timestamp": 1713625200000,
  "action": {"frame": "<base64 JPEG>"}
}

// Multi-camera rig
{"type": "step", "action": {"frames": {"front": "<JPEG>", "left": "<JPEG>", "right": "<JPEG>"}}}

// Sensor fusion (camera + IMU + speed)
{"type": "step", "action": {"frame": "<JPEG>", "imu": [0.1, 0.0, 9.8, 0.01, 0.0, 0.0], "speed_mps": 12.5}}

// LIDAR point cloud
{"type": "step", "action": {"points": [[1.2, 0.5, 0.3, 0.9], ...]}}

// Audio stream (anomaly detection)
{"type": "step", "action": {"samples": "<base64 PCM>", "sample_rate": 16000}}
```

The model defines what goes in `action` — the accelerator never looks inside.

**Active agent use cases:**
- 🤖 Robotics: sim robot in a visual world model (V-JEPA 2-AC)
- 🤖 RL training: environment step loop for policy learning
- 🚁 Drone sim: test flight controllers in predicted visual environments
- 🏭 Digital twin control: simulate interventions in a factory model

**Passive agent use cases:**
- 🚗 Driving safety: dashcam → risk prediction at 99.4% AP (BADAS/Nexar)
- 🏗️ Construction: jobsite cameras → PPE/crane collision alerts
- 🏭 Manufacturing: production line → defect prediction
- 📹 Fleet management: 350K cameras → per-vehicle risk scoring
- 🔍 Anomaly detection: any video/sensor feed → "something is wrong"

### Mid-Session Changes

Any `step` message MAY include extra fields beyond `action`. The model reads what it needs, ignores the rest. No separate event protocol — just add fields to the step.

```jsonc
// Normal step:
{"type": "step", "action": {"keyboard": [1,0,0,0], "mouse": [0.0, 0.1]}}

// Step with prompt change — "make a dragon appear" mid-game (Matrix Game 3):
{"type": "step", "action": {"keyboard": [1,0,0,0], "mouse": [0.0, 0.1]}, "prompt": "A dragon appears in the sky"}

// Step with goal image — robot planner targets this visual state (V-JEPA 2-AC):
{"type": "step", "action": {"values": [0.01, -0.02, 0, 0, 0, 0, 0.5]}, "goal_image": "<base64 JPEG>"}
```

The model picks up changes at the next autoregressive chunk boundary. The next `observation` implicitly confirms.

### Observation Schemas

The observation schema depends on the model. Declared in the manifest under `observation_type`.

#### `pixels` — Rendered frame

Used by: Matrix Game 2, Matrix Game 3

```jsonc
// Observation message (server → client)
{
  "type": "observation",
  "timestamp": 1713625200034,
  "latency_ms": 34,
  "observation": {
    "frame": "<base64-encoded JPEG>",  // Or binary frame over WebSocket binary message
    "resolution": [352, 640],           // [height, width]
    "frame_index": 42
  }
}
```

For low-latency streaming, the server MAY send binary WebSocket messages where the entire payload is the JPEG-encoded frame (no JSON envelope). The client detects binary vs text messages to distinguish.

#### `features` — Latent features + energy

Used by: V-JEPA 2-AC

```jsonc
// Observation message (server → client)
{
  "type": "observation",
  "timestamp": 1713625200050,
  "latency_ms": 50,
  "observation": {
    "features": [[0.12, -0.34, ...], ...],  // [N_tokens, D] float32
    "energy": 0.42,                           // Scalar energy for planning
    "pose": [0.5, 0.1, 0.8, 0.0, 0.1, -0.05, 0.7]  // Current [x,y,z,r,p,y,gripper]
  }
}
```

#### `prediction` — Risk score + explanation

Used by: V-JEPA 2 + prediction head (BADAS-style)

```jsonc
// Observation message (server → client)
{
  "type": "observation",
  "timestamp": 1713625200005,
  "latency_ms": 5,
  "observation": {
    "risk_score": 0.87,                  // [0.0, 1.0] probability of incident
    "heatmap": "<base64-encoded PNG>",   // Attention heatmap (optional, HxW)
    "reasoning": "Brake immediately — dark vehicle crossing intersection from left",
    "features": [[0.12, ...], ...]       // Optional: raw V-JEPA 2 features [N, D]
  }
}
```

### Session Lifecycle Messages

```jsonc
// Reset (client → server) — initialize or restart session
{
  "type": "reset",
  "image": "<base64-encoded JPEG>",     // Seed image
  "config": {
    "mode": "universal",                 // Model-specific: "universal" | "gta_drive" | "templerun"
    "prompt": "A colorful animated city",// Optional text prompt (Matrix Game 3)
    "camera_view": "left",               // Optional camera selection (V-JEPA 2-AC)
    "variant": "flash"                   // Optional model variant (BADAS)
  }
}

// Reset acknowledgement (server → client)
{
  "type": "ready",
  "session_id": "sess_abc123",
  "model_id": "matrix-game-3",
  "action_space": "human",
  "observation_type": "pixels",
  "resolution": [704, 1280],
  "target_fps": 40
}

// Close (client → server)
{
  "type": "close"
}

// Closed (server → client)
{
  "type": "closed",
  "session_id": "sess_abc123",
  "total_frames": 1200,
  "duration_s": 30.0
}

// Error (server → client)
{
  "type": "error",
  "code": "SESSION_TIMEOUT",
  "message": "Session expired after 600s of inactivity"
}
```

---

## 3. Pattern 2: `async-3d`

### Interaction Model

Submit a generation request. Poll for status. Download artifacts.

```
Client                        Server
  │                              │
  │──POST /v1/generate──────────►│  Submit job
  │◄─── {job_id, status} ───────│
  │                              │
  │──GET /v1/jobs/{id}──────────►│  Poll status
  │◄─── {status: "running"} ────│
  │      ... poll ...            │
  │──GET /v1/jobs/{id}──────────►│
  │◄─── {status: "completed",   │  Artifacts ready
  │       artifacts: [...]} ─────│
  │                              │
  │──GET /v1/jobs/{id}/artifacts/│  Download files
  │    video.mp4 ───────────────►│
  │◄─── <binary MP4> ───────────│
  │                              │
  │──GET /v1/jobs/{id}/artifacts/│
  │    scene.ply ───────────────►│
  │◄─── <binary PLY> ───────────│
```

### Request Schema

```jsonc
// POST /v1/generate
{
  "model_id": "flashworld",
  "pattern": "async-3d",            // Explicit pattern selection

  // --- Input media ---
  "image": "<base64-encoded JPEG>",     // Optional for FlashWorld (text-only supported)
                                         // Required for Lyra 2.0
  "image_index": 0,                     // Which camera in the trajectory corresponds to the image

  // --- Text prompt ---
  "prompt": "A medieval castle on a cliff with dramatic lighting",

  // --- Camera trajectory ---
  // Format A: quaternion + position (FlashWorld native)
  "cameras": [
    {
      "quaternion": [1.0, 0.0, 0.0, 0.0],   // [w, x, y, z]
      "position": [0.0, 0.0, 5.0],           // [x, y, z] world coords
      "fx": 500.0, "fy": 500.0,              // Focal length in pixels
      "cx": 352.0, "cy": 240.0               // Principal point in pixels
    }
    // ... N cameras for N frames
  ],

  // --- OR Format B: w2c matrices (Lyra 2.0 native) ---
  "cameras_w2c": {
    "w2c": [[[...4x4...], ...],              // [N, 4, 4] world-to-camera matrices
    "intrinsics": [[[...3x3...], ...]],      // [N, 3, 3] camera intrinsics
    "image_height": 480,
    "image_width": 704
  },

  // --- Per-chunk captions (Lyra 2.0) ---
  "captions": {                              // Optional, keyed by frame index
    "0": "A grand medieval castle entrance",
    "81": "The castle courtyard with stone walls",
    "161": "A tower overlooking a dramatic cliff"
  },

  // --- Resolution ---
  "resolution": {
    "n_frames": 16,                          // Number of viewpoints / frames
    "height": 480,
    "width": 704
  },

  // --- Output options ---
  "outputs": ["video", "ply", "spz"],        // Which artifacts to produce
  "video_fps": 15,                            // FPS for rendered video

  // --- Model-specific options ---
  "options": {
    "use_dmd": false,                         // Lyra 2.0: DMD distillation (15x faster, lower quality)
    "offload_t5": false,                      // FlashWorld: offload text encoder to CPU
    "zoom_in_strength": 0.5,                  // Lyra 2.0 zoom mode
    "zoom_out_strength": 1.5
  }
}
```

### Response Schema

```jsonc
// POST /v1/generate → 202 Accepted
{
  "job_id": "job_xyz789",
  "status": "queued",
  "model_id": "flashworld",
  "created_at": "2026-04-20T15:30:00Z",
  "estimated_duration_s": 7
}
```

### Job Status Schema

```jsonc
// GET /v1/jobs/{job_id}
{
  "job_id": "job_xyz789",
  "status": "completed",               // "queued" | "running" | "completed" | "failed"
  "model_id": "flashworld",
  "created_at": "2026-04-20T15:30:00Z",
  "started_at": "2026-04-20T15:30:01Z",
  "completed_at": "2026-04-20T15:30:08Z",
  "duration_s": 7.2,
  "artifacts": [
    {
      "name": "video.mp4",
      "type": "video/mp4",
      "size_bytes": 2450000,
      "url": "/v1/jobs/job_xyz789/artifacts/video.mp4"
    },
    {
      "name": "scene.ply",
      "type": "application/x-ply",
      "size_bytes": 15000000,
      "url": "/v1/jobs/job_xyz789/artifacts/scene.ply"
    },
    {
      "name": "scene.spz",
      "type": "application/x-spz",
      "size_bytes": 3200000,
      "url": "/v1/jobs/job_xyz789/artifacts/scene.spz"
    }
  ],
  "error": null                         // Non-null if status == "failed"
}
```

### Artifact Download

```
GET /v1/jobs/{job_id}/artifacts/{filename}
→ 200 OK, Content-Type: video/mp4 | application/x-ply | application/x-spz
→ Binary body
```

---

## 4. Pattern 3: `async-video`

Same protocol as Pattern 2. The only difference is the output — video only, no 3D reconstruction.

### Request Schema

```jsonc
// POST /v1/generate
{
  "model_id": "lingbot-fast",
  "pattern": "async-video",

  "image": "<base64-encoded JPEG>",       // Seed image
  "prompt": "Camera slowly orbits around the subject, cinematic lighting",

  // --- Camera trajectory (optional, model-specific) ---
  "camera_path": {
    "type": "orbit",                       // Preset trajectory type
    "radius": 2.0,
    "elevation_deg": 15.0,
    "duration_s": 5.0
  },
  // OR explicit camera poses:
  "cameras": [...],                        // Same format as Pattern 2

  // --- Resolution ---
  "resolution": {
    "n_frames": 81,
    "height": 480,
    "width": 704
  },

  // --- Output options ---
  "outputs": ["video"],
  "video_fps": 24,

  "options": {}
}
```

### Response & Job Status

Identical to Pattern 2, but `artifacts` only contains video:

```jsonc
{
  "job_id": "job_abc456",
  "status": "completed",
  "artifacts": [
    {
      "name": "video.mp4",
      "type": "video/mp4",
      "size_bytes": 8500000,
      "url": "/v1/jobs/job_abc456/artifacts/video.mp4"
    }
  ]
}
```

### Note: Pattern 2 Models as Pattern 3

Any `async-3d` model can serve `async-video` requests. When `outputs: ["video"]` is specified (no `ply`/`spz`), the 3D reconstruction step is skipped.

---

## 5. Model Registry Manifest

Each model is declared in `inference/models/<model-id>/endpoint.yaml`. The accelerator reads these at startup to route and schedule. **Only 4 fields are required** — everything else is discovered at runtime from the model itself.

### Schema

```yaml
id: string              # Unique model identifier (URL-safe)
pattern: string         # "real-time" | "async-3d" | "async-video"
image: string           # ECR image URI (same image runs on both backends)
backend: [string]       # ["ec2", "sagemaker"] — which deployment backends are available
```

That's it. The accelerator uses:
- `id` → route requests to the right model
- `pattern` → decide protocol (WebSocket session vs job queue)
- `image` → pull and run the container
- `backend` → which infrastructure to deploy on

Instance type, scaling policy, warm pool size = infrastructure config (CDK/Terraform), not model manifest. The model is the source of truth for its own capabilities — announced at runtime in the `ready` response.

### Examples

```yaml
# inference/models/matrix-game-3/endpoint.yaml
id: matrix-game-3
pattern: real-time
image: ecr/world-foundry/matrix-game-3:3.0.0
backend: [ec2, sagemaker]
```

```yaml
# inference/models/flashworld/endpoint.yaml
id: flashworld
pattern: async-3d
image: ecr/world-foundry/flashworld:1.0.0
backend: [ec2, sagemaker]
```

```yaml
# inference/models/vjepa2-ac/endpoint.yaml
id: vjepa2-ac
pattern: real-time
image: ecr/world-foundry/vjepa2-ac:2.0.0
backend: [ec2, sagemaker]
```

```yaml
# inference/models/lyra-2/endpoint.yaml
id: lyra-2
pattern: async-3d
image: ecr/world-foundry/lyra-2:2.0.0
backend: [ec2, sagemaker]
```

```yaml
# inference/models/vjepa2-predict/endpoint.yaml
id: vjepa2-predict
pattern: real-time
image: ecr/world-foundry/vjepa2-predict:2.0.0
backend: [sagemaker]          # lightweight, no need for EC2 tinkering
```

```yaml
# inference/models/interactive-world-sim/endpoint.yaml
id: interactive-world-sim
pattern: real-time
image: ecr/world-foundry/interactive-world-sim:1.0.0
backend: [ec2]                # tinkering only, no production endpoint yet
```

```yaml
# inference/models/multiworld-game/endpoint.yaml
id: multiworld-game
pattern: async-video
image: ecr/world-foundry/multiworld-game:1.0.0
backend: [ec2, sagemaker]     # 8×GPU, torchrun inside container
```

```yaml
# inference/models/multiworld-robots/endpoint.yaml
id: multiworld-robots
pattern: async-video
image: ecr/world-foundry/multiworld-robots:1.0.0
backend: [ec2, sagemaker]
```

### Dual Backend: Same Image, Two Deployment Paths

The same ECR image runs on both EC2 and SageMaker. The container contract is identical:

```dockerfile
# Dockerfile — one image, both backends
LABEL com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true
EXPOSE 8080
ENTRYPOINT ["python", "-m", "serve"]
```

| | EC2 | SageMaker |
|---|---|---|
| **How it runs** | `docker run -p 8080:8080 <image>` on a GPU instance | SageMaker pulls image, sidecar connects to the container-local WebSocket |
| **WebSocket path** | `wss://<alb-dns>/ws` (TLS via ALB + ACM cert; unencrypted transport only for loopback local dev) | SageMaker bidi streaming (HTTP/2 ↔ WebSocket bridge) |
| **Multi-session** | 1 session per instance (or N for small models) | SageMaker load-balances across instances |
| **Auto-scaling** | ASG target-tracking on active sessions | SageMaker auto-scaling on invocations |
| **Scale to zero** | No (min=1 for warm pool) | Yes (SageMaker async can scale to 0) |
| **Debug access** | SSM Session Manager, docker logs | CloudWatch logs only |
| **Use case** | Tinkering, research, rapid iteration | Production, demos, multi-tenant |

### Convention Over Configuration

- **Health check:** always `GET /health` — not configurable
- **Port:** always `8080` — not configurable
- **WebSocket path:** always `/ws` — the SageMaker sidecar maps `/invocations-bidirectional-stream` to this
- **Start command:** baked into the container's `ENTRYPOINT`/`CMD` — not in the manifest
- **Instance type:** infrastructure config (CDK/Terraform per model), not model manifest
- **Model capabilities:** returned in the `ready` message at session start (action_space, observation_type, fps, modes)
- **Metadata** (name, description, paper, license, use_cases): lives in the model's README or a separate catalogue file for the UI, not in infrastructure config

---

## 6. REST API Specification

### Base URL

```
https://{host}/v1
```

### Endpoints

| Method | Path | Pattern | Description |
|--------|------|---------|-------------|
| `GET` | `/v1/models` | All | List available models |
| `GET` | `/v1/models/{model_id}` | All | Get model manifest |
| `POST` | `/v1/sessions` | `real-time` | Create interactive session |
| `GET` | `/v1/sessions/{session_id}` | `real-time` | Get session info |
| `DELETE` | `/v1/sessions/{session_id}` | `real-time` | Close session |
| `WS` | `/v1/sessions/{session_id}/ws` | `real-time` | WebSocket step loop |
| `POST` | `/v1/generate` | `async-*` | Submit generation job |
| `GET` | `/v1/jobs/{job_id}` | `async-*` | Get job status |
| `DELETE` | `/v1/jobs/{job_id}` | `async-*` | Cancel job |
| `GET` | `/v1/jobs/{job_id}/artifacts/{name}` | `async-*` | Download artifact |

### `GET /v1/models`

```jsonc
// Response 200
{
  "models": [
    {
      "id": "matrix-game-3",
      "name": "Matrix Game 3.0",
      "pattern": "real-time",
      "action_space": "human",
      "observation_type": "pixels",
      "target_fps": 40,
      "gpu": {"count": 1, "min_vram_gb": 24, "recommended_type": "H100"},
      "status": "available",            // "available" | "warming" | "unavailable"
      "warm_instances": 1
    },
    {
      "id": "flashworld",
      "name": "FlashWorld",
      "pattern": "async-3d",
      "output_formats": ["video", "ply", "spz"],
      "gpu": {"count": 1, "min_vram_gb": 40, "recommended_type": "A100"},
      "status": "available",
      "queue_depth": 0
    }
  ]
}
```

### `POST /v1/sessions`

```jsonc
// Request
{
  "model_id": "matrix-game-3",
  "config": {
    "mode": "universal",
    "prompt": "A vibrant cyberpunk city at night"
  },
  "seed_image": "<base64-encoded JPEG>"   // Optional: use model default if omitted
}

// Response 201
{
  "session_id": "sess_abc123",
  "model_id": "matrix-game-3",
  "ws_url": "wss://{host}/v1/sessions/sess_abc123/ws",
  "status": "initializing",              // "initializing" | "ready" | "active"
  "created_at": "2026-04-20T15:30:00Z",
  "expires_at": "2026-04-20T15:40:00Z"
}
```

### `POST /v1/generate`

See full request/response schemas in [Pattern 2](#3-pattern-2-async-3d) and [Pattern 3](#4-pattern-3-async-video) sections above.

### Error Responses

All errors follow this shape:

```jsonc
// 4xx / 5xx
{
  "error": {
    "code": "MODEL_NOT_FOUND",           // Machine-readable code
    "message": "Model 'foo' not found",  // Human-readable message
    "details": {}                        // Optional additional context
  }
}
```

| Code | HTTP | Description |
|------|------|-------------|
| `MODEL_NOT_FOUND` | 404 | Model ID does not exist in registry |
| `MODEL_UNAVAILABLE` | 503 | Model has no warm instances and cold start is not allowed |
| `SESSION_NOT_FOUND` | 404 | Session ID does not exist or has expired |
| `SESSION_TIMEOUT` | 408 | Session exceeded `session_timeout_s` |
| `JOB_NOT_FOUND` | 404 | Job ID does not exist |
| `INVALID_ACTION_SPACE` | 400 | Action payload does not match model's declared `action_space` |
| `INVALID_CAMERA_FORMAT` | 400 | Camera data does not match model's declared `camera_format` |
| `GPU_CAPACITY` | 503 | No GPU capacity available, try again later |
| `GENERATION_FAILED` | 500 | Model inference failed (see `details` for stack trace) |

---

## 7. Runner Protocol (Python)

Every model implements one of two Python protocols. The accelerator imports and wraps these.

### `RealtimeRunner` — Pattern 1

```python
from typing import Protocol, Any

class RealtimeRunner(Protocol):
    """Pattern 1: stateful step loop.
    
    Lifecycle:
        runner = ModelRunner()          # Load model, allocate GPU
        obs = runner.reset(image, config)  # Initialize session state
        for each frame:
            obs = runner.step(action)   # Action shape depends on action_space
        runner.close()                  # Free session state
    """

    def reset(self, image: bytes, config: dict) -> dict:
        """Initialize a new session.
        
        Args:
            image: JPEG-encoded seed image bytes
            config: Model-specific config (mode, prompt, variant, camera_view, ...)
        
        Returns:
            First observation dict (shape depends on observation_type)
        """
        ...

    def step(self, action: dict) -> dict:
        """Execute one step.
        
        Args:
            action: Action dict. Shape depends on action_space:
                human:  {"keyboard": [int], "mouse": [float]}
                agent:  model-defined dict (see Action Spaces section)
        
        Returns:
            Observation dict. Shape depends on observation_type:
                pixels:     {"frame": bytes, "resolution": [int,int], "frame_index": int}
                features:   {"features": list[list[float]], "energy": float, "pose": list[float]}
                prediction: {"risk_score": float, "heatmap": bytes, "reasoning": str}
        """
        ...

    def close(self) -> None:
        """Free session state. Called when WebSocket closes or session times out."""
        ...
```

### `GenerateRunner` — Patterns 2 & 3

```python
class GenerateRunner(Protocol):
    """Patterns 2 & 3: async generation.
    
    Lifecycle:
        runner = ModelRunner()                  # Load model, allocate GPU
        result = runner.generate(request)       # One-shot generation
        # runner stays warm for the next job
    """

    def generate(self, request: dict) -> dict:
        """Run generation.
        
        Args:
            request: Full generation request dict. Contains:
                - image: Optional[bytes] — seed image
                - prompt: str — text description
                - cameras: list[dict] — camera trajectory (quaternion or w2c format)
                - resolution: dict — {n_frames, height, width}
                - outputs: list[str] — ["video", "ply", "spz"]
                - options: dict — model-specific options
        
        Returns:
            {
                "video_path": str | None,    # Path to generated MP4
                "ply_path": str | None,      # Path to generated PLY
                "spz_path": str | None,      # Path to generated SPZ
                "generation_time_s": float,  # Wall-clock generation time
                "metadata": dict             # Model-specific metadata
            }
        """
        ...
```

---

## 8. Model Catalogue

| Model | Pattern | Action Space / Input | Observation / Output | GPU | Latency / Time |
|-------|---------|---------------------|---------------------|-----|----------------|
| **Matrix Game 2** | `real-time` | `human` (4-key + mouse) | 352×640 JPEG frames | 1× L40S (24GB) | 25 FPS / 40ms |
| **Matrix Game 3** | `real-time` | `human` (4-key + mouse) | 704×1280 JPEG frames | 1× H100 (80GB) | 40 FPS / 25ms |
| **V-JEPA 2-AC** | `real-time` | `agent` (7-dim deltas) | Latent features + energy | 1× L40S (48GB) | 20 FPS / 50ms |
| **V-JEPA 2 Predict** | `real-time` | `agent` (camera feed) | Risk score + heatmap + reasoning | 1× L4 (24GB) | 30 FPS / 5–34ms |
| **FlashWorld** | `async-3d` | Image/text + cameras (quat) | .spz + .ply + .mp4 | 1× A100 (40GB) | 7s |
| **Lyra 2.0** | `async-3d` | Image + w2c cameras + captions | .ply + .mp4 | 1× H100 (80GB) | 35s–9min |
| **MultiWorld (game)** | `async-video` | Image + multi-agent actions + env observation | .mp4 (multi-view) | 8× GPU (torchrun) | batch |
| **MultiWorld (robots)** | `async-video` | Image + robot actions + env observation | .mp4 (multi-view) | 8× GPU (torchrun) | batch |
| **Interactive World Sim** | `real-time` | `human` (WASD/IJKL keyboard) | 128×128 JPEG frames | 1× GPU (2GB+) | 10 FPS / 100ms |
| **LingBot Fast** | `async-video` | Image + prompt + camera path | .mp4 | 1× L40S (48GB) | ~30s |

---

## 9. Use Case Matrix

How the patterns compose for real-world applications:

| Use Case | Pattern(s) Used | Models | Flow |
|----------|----------------|--------|------|
| **🤖 Robotics: environment simulation** | `real-time` (agent) | V-JEPA 2-AC | Robot sends joint actions → predicted visual observations for planning |
| **🤖 Robotics: training env generation** | `async-3d` | FlashWorld, Lyra 2.0 | Text prompt → diverse 3D environments (.ply) → import into sim |
| **🤖 Robotics: risk prediction** | `real-time` (agent) | V-JEPA 2 Predict | Robot camera feed → collision/risk scores in real-time |
| **🎮 Gaming: playable worlds** | `real-time` (human) | Matrix Game 3 | WASD + mouse → 720p 40fps interactive world |
| **🚗 Autonomous driving sim** | `real-time` (human) | Matrix Game 2 (gta_drive) | Steering/gas → first-person driving video |
| **🚗 Autonomous driving safety** | `real-time` (agent) | V-JEPA 2 Predict | Dashcam feed → risk anticipation (BADAS-style) |
| **📊 Synthetic data: 3D scenes** | `async-3d` | FlashWorld | Batch text prompts → 3D Gaussian Splat scenes in 7s each |
| **📊 Synthetic data: environments** | `async-3d` | Lyra 2.0 | Photos → explorable 3D worlds with consistent geometry |
| **🏗️ Digital twins** | `async-3d` | Lyra 2.0 | Building photo → walkable 3D reconstruction |
| **🎮 Multi-agent gaming** | `async-video` | MultiWorld (game) | Two-player actions → multi-view video of "It Takes Two"-style co-op |
| **🤖 Multi-robot sim** | `async-video` | MultiWorld (robots) | Multi-robot actions → multi-view video with 3D-consistent scene |
| **🤖 Robot policy training (latent)** | `real-time` (human) | Interactive World Sim | Keyboard-controlled robot → visual world model for data collection |
| **🎬 Video creation** | `async-video` | LingBot Fast | Prompt + camera path → cinematic video |
| **🔍 Video understanding** | `real-time` (agent) | V-JEPA 2 Predict | Any video → scene features + predictions |

### Chaining Patterns

```
async-3d (Lyra 2.0)           →  Generate 3D training environments
    ↓ .ply files
real-time / agent (V-JEPA 2-AC)  →  Simulate robot in environment
    ↓ predicted observations
real-time / agent (V-JEPA 2 Predict) → Score safety of robot actions
    ↓ risk scores
Robot Policy ← reward signal
```

---

## Appendix A: Camera Format Conversion

Models accept different camera formats. The router converts between them:

```python
def quaternion_position_to_w2c(quat, pos, fx, fy, cx, cy, h, w):
    """Convert FlashWorld format to Lyra 2.0 format."""
    R = quaternion_to_rotation_matrix(quat)  # [3, 3]
    t = -R @ pos                              # [3]
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return w2c, K

def w2c_to_quaternion_position(w2c, K, h, w):
    """Convert Lyra 2.0 format to FlashWorld format."""
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pos = -R.T @ t
    quat = rotation_matrix_to_quaternion(R)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    return quat, pos, fx, fy, cx, cy
```

## Appendix B: Binary WebSocket Frame Format

For maximum performance in Pattern 1 (`real-time`), the server MAY send binary WebSocket frames instead of JSON:

```
Binary frame layout (server → client, observation_type=pixels):
┌──────────────────────────────────────────────────┐
│ Byte 0-3:   frame_index (uint32, little-endian)  │
│ Byte 4-7:   latency_ms (float32, little-endian)  │
│ Byte 8-N:   JPEG image data                      │
└──────────────────────────────────────────────────┘

Binary frame layout (client → server, action_space=human):
┌──────────────────────────────────────────────────┐
│ Byte 0:     action type (0x01 = step)            │
│ Byte 1-4:   keyboard bitmask (uint32)            │
│ Byte 5-8:   mouse_dy (float32, little-endian)    │
│ Byte 9-12:  mouse_dx (float32, little-endian)    │
└──────────────────────────────────────────────────┘
```

The client and server negotiate binary mode during the `reset` handshake via `"binary_protocol": true`.
