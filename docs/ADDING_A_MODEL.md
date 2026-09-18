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

## 5. Optional: surface it in the UI

Add a row to the table in [Endpoints](ENDPOINTS.md) and a card in
`frontend/src/data/cartridges.ts`.

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
