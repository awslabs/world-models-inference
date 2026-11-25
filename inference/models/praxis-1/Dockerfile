# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Praxis-1 (Runway) on the shared serving layer. Mirrors the install that ran the
# upstream DROID sim end to end on a g7e.2xlarge (RTX PRO 6000 Blackwell, CUDA 13
# driver): Python 3.12, torch 2.13 / cu130, causal-conv1d built from source,
# flash-attn-4, and the cuDNN pin — in that order, which is what makes them agree.
#
# The CUDA 13 devel base provides nvcc for the causal-conv1d build. The host's
# NVIDIA driver must support CUDA 13 (R580 or newer); see README.md.
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

ENV DEBIAN_FRONTEND=noninteractive
ENV SM_MODEL_DIR=/opt/ml/model
ENV SM_MODULE_DIR=/opt/ml/code
ENV SAGEMAKER_BIND_TO_PORT=8080
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH
# MuJoCo renders the sim cameras through EGL on the GPU. The NVIDIA container
# toolkit only mounts the driver's EGL/GL libraries for the `graphics` capability,
# which the CUDA images leave out by default (compute,utility).
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Python 3.12 (Ubuntu 24.04's own), the build toolchain, and the EGL/GL libraries
# MuJoCo's headless renderer needs (MUJOCO_GL=egl).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv git build-essential ninja-build ffmpeg \
        libegl1 libgl1 libgles2 libglvnd0 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip

# torch 2.13 built for CUDA 13.0. Retried: the PyTorch index has refused CodeBuild's
# TLS handshake before (see the lyra-2 Dockerfile); fail loudly if it never answers.
RUN for i in 1 2 3 4 5; do \
        pip install --no-cache-dir --retries 5 --timeout 120 \
            --index-url https://download.pytorch.org/whl/cu130 \
            torch==2.13.0 torchvision==0.28.0 && exit 0; \
        echo "torch install attempt $i failed; retrying..."; sleep 15; \
    done; echo "torch install failed after 5 attempts" >&2; exit 1

# Hold torch where it is for every later install, so nothing can swap it for a
# wheel built against another CUDA.
RUN python -c "import torch, torchvision; open('/etc/torch-constraints.txt','w').write(f'torch=={torch.__version__.split(\"+\")[0]}\ntorchvision=={torchvision.__version__.split(\"+\")[0]}\n')" && \
    cat /etc/torch-constraints.txt
ENV PIP_CONSTRAINT=/etc/torch-constraints.txt

# causal-conv1d has no wheel for this torch/CUDA, so it is compiled against the
# torch above (no build isolation). Slow (~15 min); MAX_JOBS bounds build memory.
RUN pip install --no-cache-dir packaging ninja wheel && \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 \
        pip install --no-cache-dir --no-build-isolation causal-conv1d==1.6.1

# Upstream at a fixed commit. A branch moves; a rebuild must not pick up new code.
ARG PRAXIS_SHA=2042edf6f5512294f9242636ee1ce235153b96f0
RUN git clone https://github.com/runwayml/praxis-1-oss.git /opt/praxis-1 && \
    git -C /opt/praxis-1 checkout --quiet ${PRAXIS_SHA}
ENV PRAXIS_REPO=/opt/praxis-1

# Upstream's MoE GEMM uses three pipeline stages, which need more shared memory per
# block than an RTX PRO 6000 / L40S-class GPU has (101,376 B) and fail with Triton
# OutOfResources. Cap them at two there; pipelining only, the maths is unchanged.
# The script fails the build if upstream's text has moved.
COPY inference/models/praxis-1/patch_moe_smem.py /tmp/patch_moe_smem.py
RUN python /tmp/patch_moe_smem.py /opt/praxis-1/praxis/kernels/moe_gemm_triton.py

# praxis + the sim package from the checkout, with the text-encoder kernels the
# validation used ([kernels]), then flash-attn-4. 4.0.0b36 is the build that was
# validated; upstream's optional [flash] extra names b37 at PRAXIS_SHA.
RUN cd /opt/praxis-1 && \
    pip install --no-cache-dir -e packages/praxis-sim -e ".[kernels]" "imageio>=2.30" "imageio-ffmpeg>=0.4" "scipy>=1.10" && \
    pip install --no-cache-dir "flash-attn-4[cu13]==4.0.0b36"

# Bake the MuJoCo Menagerie robot models (pinned commit, sha256-checked by
# praxis-sim) into the image, so a rollout never downloads at run time.
ENV PRAXIS_SIM_CACHE=/opt/praxis-sim-cache
RUN python -c "from praxis_sim import assets; print(assets.ensure_assets())"

# Shared serving deps, including the Cognito token verifier (lib/cognito.py).
# Installed from one file so a cartridge cannot ship an image that is missing the
# auth library — that is how an endpoint once kept serving with auth unenforced
# (PCSR D484282450, round 3). A missing auth dep must break the build instead.
COPY inference/lib/requirements.txt /tmp/lib-requirements.txt
RUN pip install --no-cache-dir -r /tmp/lib-requirements.txt

COPY inference/lib/ /opt/ml/code/lib/
COPY inference/models/praxis-1/requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt
COPY inference/models/praxis-1/ /opt/ml/code/

# The cuDNN the validated run used, last so no later install can move it. --no-deps:
# the wheel alone, so it cannot drag in a second CUDA runtime. pip reports that it
# differs from torch's own pin; that difference is the validated configuration.
RUN pip install --no-cache-dir --no-deps nvidia-cudnn-cu13==9.24.0.43

# Build-time sanity guard: fail here, not on the GPU, if the core imports disagree.
# praxis.sim.yam_player covers the two-arm rig's own imports (scipy: the [sim] extra,
# which is not installed whole because it would add a second praxis-sim from PyPI).
RUN python -c "import torch, praxis, praxis_sim, mujoco, imageio_ffmpeg, scipy, praxis.sim.yam_player; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'mujoco', mujoco.__version__, 'ok')"

# One child process owns the one GPU per job (runner.py); force a single rank.
ENV NPROC_PER_NODE=1

# Run as a non-root user (M3). /tmp is world-writable for job/output scratch; the
# kernel cache (PRAXIS_CACHE_DIR) defaults under it.
# praxis-sim's asset cache is verified (and refetched on a mismatch) at run time by the
# non-root user, so it must own it; some fetched files are mode 0600.
RUN useradd --create-home appuser && chown -R appuser /opt/ml/code /opt/praxis-1 /opt/praxis-sim-cache
USER appuser

WORKDIR /opt/ml/code
EXPOSE 8080
# Container health: /ping is unauthenticated and cheap. setup() only checks paths,
# but the instance first syncs ~148 GB of weights, hence the long start period; the
# status is advisory (ALB/SageMaker run their own /ping probes), nothing restarts on it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/ping', timeout=4).status == 200 else 1)"
ENTRYPOINT ["./lib/serve"]
