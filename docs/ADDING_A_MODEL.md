# Adding a model

Copy the template and fill in three files. **No Dockerfile, no server, no CDK
changes.**

```bash
cp -r inference/models/_template inference/models/my-model
```

## 1. `endpoint.yaml`

Declares where the weights live and what to run them on.

```yaml
model_data: s3://world-model-artifacts-${AWS_ACCOUNT_ID}/my-model/
version: 0.1.0

hf:
  - { repo: my-org/my-model, subdir: "" }

ec2:
  instance: g6e.2xlarge      # smallest GPU that fits the model
  volume_size: 200           # GB; must exceed the weights

sagemaker:
  instance: ml.g6e.2xlarge
  scale: { min: 0, max: 1 }
  output: s3://world-model-outputs-${AWS_ACCOUNT_ID}/my-model/
```

Two things worth knowing:

- **The bucket name in `model_data` is ignored.** Only the path prefix after the
  bucket is used; the real bucket comes from the foundation stack via SSM.
- **The serving mode is derived from `sagemaker.output`, not from a `mode:` key.**
  Present → `async`. Absent → `real-time`. A `mode:` line is documentation only.

## 2. `runner.py`

Subclass `Runner` and provide a `create_runner()` factory.

```python
from lib.runner import Runner

class MyRunner(Runner):
    def setup(self, ckpt_dir, device, rank, world_size):
        """Load weights once. Called on every rank at startup."""

    def generate(self, **params) -> str:
        """Run one job, return the output file path.
        Called on all ranks in lockstep."""

def create_runner() -> Runner:
    return MyRunner()
```

For **real-time** models override `stream()` instead and set
`supports_streaming = True`:

```python
    @property
    def supports_streaming(self) -> bool:
        return True

    def stream(self, actions: ActionBuffer) -> Iterator[bytes]:
        while (action := actions.get()) is not None:
            yield self.step(action)      # yield one frame per iteration
```

Optionally override `reset()` to clear per-session state.

**You never write a server.** `inference/lib/serve` starts `torchrun` with one
process per GPU, calls `setup()` on every rank, serves the secured FastAPI app from
rank 0, and broadcasts each job so all ranks run `generate()` together. See
[Architecture](ARCHITECTURE.md#run).

## 3. `requirements.txt`

Your model's dependencies. The shared image already provides the serving stack
(`fastapi`, `uvicorn`, `python-multipart`, `websockets`, `imageio`).

## 4. Stage weights and deploy

```bash
python scripts/stage-weights.py my-model --source hf:my-org/my-model
./deploy.sh my-model
```

Gated Hugging Face repos need `HF_TOKEN` set. Staging is one-off — later launches
sync from S3.

## 5. Document it and surface it in the UI

Three places, and `tests/test_cartridge_parity.py` enforces most of it, so a stale
page cannot merge. What it actually checks: all three exist for every cartridge; the
card's and the table's `instance` both match `endpoint.yaml`; the card's `deploy`
targets are a subset of what the manifest declares; `fps > 0` exactly when the
manifest describes a real-time endpoint; the `restricted` flag matches
`license_restricted`; and the table's **Wired?** marker agrees with the card's
`build`. What it does *not* check is prose — no test can tell whether your measured
numbers are real, so the rest of this section is on you.

1. **`inference/models/my-model/README.md`** — the per-model page, and the one thing a
   reader looks for first. What it must answer: upstream link and licence; which
   weights to stage and how (the `hf:` block is documentation — nothing reads it, so
   spell out the `stage-weights.py` invocations); the exact `./deploy.sh` commands;
   **what hardware, with the measured number on each shape you actually ran** and an
   explicit note on which shapes are untested; what to invoke (routes, and the session
   protocol if real-time); tunable environment variables; and caveats a first-time user
   will otherwise hit. `waypoint-1-5` and `matrix-game-3` are the worked examples.
2. **A row in [Endpoints](ENDPOINTS.md)** — the models table plus a short per-model
   section linking to the README above.
3. **A card in `frontend/src/data/cartridges.ts`** — `instance` and `deploy` must match
   the manifest; `fps > 0` only for real-time; `build: 'wired'` only once you have
   invoked it end to end. Quote *measured* fps, and say so in `notes` when a number is
   a design target rather than a measurement.

Write some tests, too. A cartridge can be covered offline — stub the model library and
torch — which is worth doing because the alternative is discovering a wire-format bug
on a $8/hr instance. `tests/test_waypoint_1_5_session.py` drives a whole WebSocket
session against the real route and the real `stream()` with a fake engine, and
`tests/test_waypoint_1_5.py` pins the manifest and every action wire format.

---

## How the container image is built

Every model gets an image in ECR (`world-model-<id>:<version>`), built by CodeBuild —
no local Docker required. Two paths:

### Shared default (preferred)

With no per-model `Dockerfile`, `inference/Dockerfile.default` builds the cartridge.
Tune it declaratively from `endpoint.yaml`:

| Key | Purpose |
|---|---|
| `base_image` | Base to build on. Default `nvcr.io/nvidia/pytorch:24.01-py3` |
| `pip_no_deps` | `0` installs `requirements.txt` **with** transitive deps. The default `--no-deps` stops pip replacing the CUDA-matched torch the NGC base ships |
| `flash_attn_wheel` | URL of a prebuilt flash-attn wheel. Put `{ABI}` where the C++11-ABI flag goes and it is substituted at build time from the installed torch |

The build also asserts that `runner.py` exposes `create_runner()`, so a broken
cartridge fails at build time rather than on an expensive GPU instance.

### Custom `Dockerfile`

Drop one in the model directory and it takes precedence — for genuinely custom
builds: native CUDA extensions, upstream git clones, pinned toolchains. See `lyra-2`
for a worked example.

`./deploy.sh list` shows which path each model uses. Set `REBUILD_IMAGE=1` to force a
rebuild after changing inference code, since an unchanged image tag is reused by
default.

## Restricted licences

If a model's weights are under a licence that limits redistribution or production
use, declare it:

```yaml
license: nvidia-internal-research
license_url: https://huggingface.co/nvidia/Lyra-2.0
license_restricted: true
```

`deploy.sh` then requires interactive acknowledgement before deploying.
`ACCEPT_LICENSE=1` skips the prompt in automation. `lyra-2` is the current example.
