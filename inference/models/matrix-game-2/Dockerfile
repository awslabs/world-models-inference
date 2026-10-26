# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Matrix Game 2.0 — Skywork's real-time interactive world model.
#
# The exact toolchain that ran this model on an H100 at the AWS Summit Dubai
# booth. Each pin is there because the obvious alternative failed:
#
#   torch 2.8.0 (cu128)  2.6 lacks torch._dynamo.config.recompile_limit, which
#                        the compiled VAE decoder needs; 2.9 has no matching
#                        Python 3.10 flash-attn wheel (only cp312 / CUDA 13).
#   Python 3.10          the flash-attn wheel below is cp310. Ubuntu 22.04's
#                        system Python, so no deadsnakes PPA.
#   flash-attn 2.8.3     prebuilt for exactly torch 2.8 + cu12 + cp310; building
#                        from source takes over an hour and needs nvcc.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# git: upstream clone. gcc/python3-dev: triton JIT-compiles its launcher stubs
# with the host C compiler when the VAE decoder is torch.compile'd ("Failed to
# find C compiler" without them). libgl1/libglib2.0-0: in case upstream's
# imports reach cv2.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip python3-dev gcc libc6-dev \
        git ca-certificates libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip wheel

# torch first, from the cu128 index, so nothing later can swap in a different
# build. The cu128 wheels bundle their own CUDA libraries; only the host driver
# matters, which the DLAMI provides.
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Inline rather than an ARG so scripts/license-check.py can see the source
# (flash-attention is BSD-3-Clause, already in its reviewed list).
RUN pip install --no-cache-dir --no-deps \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Pin the upstream commit — the same snapshot matrix-game-3 pins, and the one
# the Dubai deployment cloned. Matrix-Game-2/ is plain Python imported by path;
# nothing in it is pip-installed.
ARG MATRIX_GAME_REF=71c3cd7f741311f8100f6cf9cde942b6c1378d11
RUN git clone --filter=blob:none https://github.com/SkyworkAI/Matrix-Game.git \
        /opt/ml/code/Matrix-Game && \
    cd /opt/ml/code/Matrix-Game && \
    git checkout ${MATRIX_GAME_REF} && \
    rm -rf .git Matrix-Game-3

# Shared serving deps, including the Cognito token verifier (lib/cognito.py).
# Installed from one file so no cartridge can ship an image missing the auth
# library — a missing auth dependency must break the build, not the security
# posture.
COPY inference/lib/requirements.txt /tmp/lib-requirements.txt
RUN pip install --no-cache-dir -r /tmp/lib-requirements.txt

COPY inference/models/matrix-game-2/requirements.txt /tmp/mg2-requirements.txt
RUN pip install --no-cache-dir -r /tmp/mg2-requirements.txt && \
    python -c "import torch, flash_attn; print('torch', torch.__version__, 'flash_attn', flash_attn.__version__); \
assert torch.__version__.startswith('2.8.0'), 'a dependency replaced torch'"

# Weights are staged to /opt/ml/model — nothing is fetched from the Hub at runtime.
ENV HF_HUB_OFFLINE=1
ENV MG2_REPO=/opt/ml/code/Matrix-Game/Matrix-Game-2
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Single-GPU model: keep torchrun to one rank even on a multi-GPU host.
ENV NPROC_PER_NODE=1

# Compile caches must live somewhere the non-root user can write.
ENV TRITON_CACHE_DIR=/tmp/triton-cache
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache
ENV HOME=/home/appuser

COPY inference/lib/ /opt/ml/code/lib/
COPY inference/models/matrix-game-2/ /opt/ml/code/

# Run as a non-root user (M3). /tmp is world-writable for job/output scratch.
RUN useradd --create-home appuser && chown -R appuser /opt/ml/code
USER appuser

# SageMaker bidirectional streaming contract
LABEL com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
EXPOSE 8080
# /ping is unauthenticated and cheap. Warmup compiles the VAE decoder before
# /ping goes green, which takes minutes on first boot — hence the long start
# period. Advisory only; the ALB/SageMaker run their own probes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/ping', timeout=4).status == 200 else 1)"
WORKDIR /opt/ml/code
ENTRYPOINT ["./lib/serve"]
