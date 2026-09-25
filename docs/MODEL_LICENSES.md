# Model licences

This repository's platform code is licensed under Apache-2.0 (see `LICENSE`).
The models it deploys are licensed separately by their authors, and some of
those licences restrict how the models may be used. Deploying a model means
accepting its licence; review it before use.

Each cartridge involves up to three licences:

| Layer | Where it lives |
|---|---|
| Platform code | This repository — Apache-2.0 |
| Upstream model code | Either vendored into this repository (noted below) or installed into the container image at build time |
| Model weights | Never in this repository; downloaded from Hugging Face at deploy time |

## Licences per cartridge

Verified 2026-09-22 against the Hugging Face model API and the upstream GitHub
repositories.

| Cartridge | Weights | Weights licence | Upstream code |
|---|---|---|---|
| `cosmos3-nano` | `nvidia/Cosmos3-Nano` | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) | none in repository |
| `lingbot-fast` | `robbyant/lingbot-world-base-cam`, `robbyant/lingbot-world-fast` | Apache-2.0 | 53 vendored files under `wan/`, Apache-2.0 (`UPSTREAM_LICENSE.txt`) |
| `matrix-game-3` | `Skywork/Matrix-Game-3.0` | Apache-2.0 | none in repository; `SkyworkAI/Matrix-Game` (MIT) installed into the image |
| `vjepa2` | `facebook/vjepa2-vitg-fpc64-384` | Apache-2.0 | none in repository; upstream is MIT |
| `lyra-2` | `nvidia/Lyra-2.0` | NVIDIA Internal Scientific Research and Development Model License (`nvidia-internal-research`) — **Yes, restricted: internal research only** | none in repository; `nv-tlabs/lyra` (Apache-2.0) installed into the image |
| `echo-async`, `echo-realtime` | none (test fixtures) | Apache-2.0 | none |
| `waypoint-1-5` | `Overworld/Waypoint-1.5-1B` | Apache-2.0 | none in repository; `world_engine` (GPL-3.0) is installed into the container image only |
| `lingbot-v2` (not offered) | `robbyant/lingbot-world-v2-14b-causal-fast-diffusers` | CC BY-NC-SA 4.0 — **Yes, restricted: non-commercial** | not integrated: the weights licence is incompatible with this project's intended uses |

`lingbot-v2` weights are licensed for non-commercial use only, which is
incompatible with this project's intended uses; the model is therefore listed
for transparency but not offered as a cartridge.

`lyra-2` may not be used for production, public deployment, or redistribution.
`deploy.sh` requires explicit acknowledgement before deploying it. At the time
of verification the Hugging Face model card for `nvidia/Lyra-2.0` carried no
licence metadata; confirm the current terms with NVIDIA before any external use.

## How this is enforced

1. Every `inference/models/<id>/endpoint.yaml` declares `license`,
   `license_url`, and `license_restricted`. `deploy.sh` reads these fields and
   refuses to deploy a restricted model until the operator acknowledges its
   terms (`ACCEPT_LICENSE=1` accepts non-interactively).
2. `tests/test_model_licenses.py` fails if a cartridge omits these fields, is
   missing from this document, or declares a licence this document does not
   state.

## Adding a cartridge

1. Determine the weights licence from the Hugging Face model card and the model
   API (`https://huggingface.co/api/models/<repo>`). A `license: other` tag is
   not a licence — read the linked `license_name`/`license_link`.
2. Check the upstream code licence separately; it is often different from the
   weights licence.
3. If upstream code is vendored, copy its licence to
   `inference/models/<id>/UPSTREAM_LICENSE.txt` and note modifications in any
   changed file.
4. Declare `license`, `license_url`, and `license_restricted` in
   `endpoint.yaml`. Use `license_restricted: true` for anything
   non-commercial, research-only, or unlicensed.
5. Add a row to the table above.
