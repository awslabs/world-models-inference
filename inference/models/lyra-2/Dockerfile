# Plain CUDA base + PyPI torch, matching Lyra-2's official INSTALL.md conda
# recipe (torch 2.7.1 / cu128 / Python 3.10). The NGC prebuilt images were the
# problem: they ship a second numpy 1.x (conda-layer) that pip can't cleanly
# remove, which shadows the numpy 2.x Lyra needs and breaks cv2's import. A clean
# CUDA base with torch installed from PyPI (numpy-2.x native) avoids that at root.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# System deps: Python 3.10, build toolchain, and the native libs vipe/opencv need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip git cmake ninja-build \
        build-essential libgl1 libglib2.0-0 ffmpeg libeigen3-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --no-cache-dir --upgrade pip

# PyTorch 2.7.1 from plain PyPI (numpy-2.x native). We deliberately avoid
# download.pytorch.org/whl/cu128 — its Cloudflare-R2 backend
# (download-r2.pytorch.org) repeatedly refuses the TLS handshake from CodeBuild
# (SSLV3_ALERT_HANDSHAKE_FAILURE), which kills the build. PyPI's torch 2.7.1 is a
# CUDA-12.6 build: it runs fine on this 12.8 base (CUDA minor-version compat) and
# satisfies the flash-attn cu12/torch2.7 wheel below. Retry a few times and fail
# loudly if the index is unreachable (don't let a later step mask a missing torch).
RUN for i in 1 2 3 4 5; do \
        pip install --no-cache-dir --retries 5 --timeout 120 \
            torch==2.7.1 torchvision==0.22.1 && exit 0; \
        echo "torch install attempt $i failed; retrying..."; sleep 15; \
    done; echo "torch install failed after 5 attempts" >&2; exit 1

# python-multipart is required by lib/app.py's /generate form route (Form/File).
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Pin upstream to a specific commit for reproducible builds. The runner imports
# the `lyra_2` package in-process (load-once); the two native submodules
# (vipe, depth_anything_3) are C++/CUDA builds that must be `pip install -e`.
ARG LYRA_SHA=87f79a52b81b366d1d4aa3a526aa12e54207c998
RUN git clone --recursive https://github.com/nv-tlabs/lyra.git /opt/ml/code/lyra && \
    cd /opt/ml/code/lyra && git checkout ${LYRA_SHA} && git submodule update --init --recursive

# Upstream pinned deps (numpy>=2, opencv==4.11.0.86, transformers, ...) resolve
# cleanly here since torch is numpy-2.x native. --no-deps per INSTALL.md; MoGe
# also --no-deps (its runtime deps are covered and free resolution conflicts).
RUN cd /opt/ml/code/lyra/Lyra-2 && \
    pip install --no-cache-dir --no-deps -r requirements.txt && \
    pip install --no-cache-dir --no-deps "git+https://github.com/microsoft/MoGe.git"

# Build the CUDA extensions only for the GPUs we deploy to (H100=9.0, L40S=8.9).
# Upstream defaults to ~11 arches, OOM-killing cc1plus. MAX_JOBS bounds memory.
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0"
ENV MAX_JOBS=2

# Pin torch so editable CUDA-ext installs (--no-build-isolation) can't swap it.
RUN python -c "import torch, torchvision; open('/tmp/torch-constraints.txt','w').write(f'torch=={torch.__version__.split(\"+\")[0]}\ntorchvision=={torchvision.__version__.split(\"+\")[0]}\n')" && \
    cat /tmp/torch-constraints.txt
ENV PIP_CONSTRAINT=/tmp/torch-constraints.txt

# Build backends for the editable installs, plus flash-attn.
RUN pip install --no-cache-dir "hatchling" "pathspec" "editables" "hatch-vcs"
RUN pip install --no-cache-dir --no-build-isolation "transformer_engine[pytorch]" || \
    echo "WARN: transformer_engine install failed (may be optional at runtime)"

# flash-attn: install the official prebuilt wheel instead of compiling from
# source. Upstream pins 2.6.3, but that release predates torch 2.7 and has no
# matching prebuilt wheel — a source compile against torch 2.7's headers fails
# ("Error compiling objects for extension"). 2.8.3.post1 ships a cu12/torch2.7/
# cp310 wheel and keeps the same public API Lyra uses (flash_attn_func /
# flash_attn_varlen_func), so we pick the wheel matching torch's C++11-ABI flag.
RUN ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") && \
    echo "torch C++11 ABI: $ABI" && \
    pip install --no-cache-dir \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.7cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl"

# depth_anything_3 first (fast), then vipe (slow CUDA compile).
# No [gs] extra: gaussian-splat reconstruction (Step 2) is not wired here.
RUN cd /opt/ml/code/lyra/Lyra-2 && \
    pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3' && \
    USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'
ENV LYRA_REPO=/opt/ml/code/lyra/Lyra-2

COPY inference/lib/ /opt/ml/code/lib/
COPY inference/models/lyra-2/requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

COPY inference/models/lyra-2/ /opt/ml/code/

# Build-time sanity guard: fail loudly here (not on the GPU) if the core imports
# don't line up. On this clean base numpy/torch/cv2 should all import.
RUN python -c "import numpy, torch, cv2; import torch.nn.functional; print('numpy', numpy.__version__, 'torch', torch.__version__, 'cv2', cv2.__version__, 'ok')"

# Lyra loads in-process on a single GPU (context-parallel is not wired). Force a
# single rank regardless of how many GPUs the instance exposes.
ENV NPROC_PER_NODE=1

# Run as a non-root user (M3). /tmp is world-writable for job/output scratch.
RUN useradd --create-home appuser && chown -R appuser /opt/ml/code
USER appuser

WORKDIR /opt/ml/code
EXPOSE 8080
ENTRYPOINT ["./lib/serve"]
