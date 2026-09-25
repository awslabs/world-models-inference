# Matrix Game 3.0 — real-time interactive world model.
#
# Its own base image on purpose: upstream needs torch 2.10 and Python 3.12,
# while the shared inference/Dockerfile.default is pinned to torch 2.4 with
# FlashAttention 2 for the LingBot lineage. Two model families, two toolchains.
FROM nvcr.io/nvidia/pytorch:25.09-py3

# Pin the upstream commit. An unpinned clone means the image silently changes
# under us and a working deployment stops being reproducible. This is main as of
# 2026-09-09, the snapshot the ulysses patch below was written against and the
# commit upstream issue #80 reproduces on.
ARG MATRIX_GAME_REF=71c3cd7f741311f8100f6cf9cde942b6c1378d11
RUN git clone --filter=blob:none https://github.com/SkyworkAI/Matrix-Game.git \
        /opt/ml/code/Matrix-Game && \
    cd /opt/ml/code/Matrix-Game && \
    git checkout ${MATRIX_GAME_REF} && \
    rm -rf .git

# opencv-python needs libGL, which no slim CUDA base ships. Upstream imports cv2
# on the pipeline import path, so without these the server dies on
# "ImportError: libGL.so.1: cannot open shared object file".
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install upstream's dependencies WITHOUT torch.
#
# Their requirements.txt pins torch==2.10.0, and letting pip honour that replaces
# this image's torch 2.9.0a0/CUDA 13.0 with the PyPI cu128 build. That in turn
# orphans the flash-attn 2.7.4 that NGC ships pre-built against the image's own
# torch, and Matrix Game 3 cannot run without flash-attn: upstream's
# wan/modules/model.py calls flash_attention() directly, and that function opens
# with `assert FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE`. There is no
# SDPA path through the DiT, so a missing or broken flash-attn is fatal, not slow.
#
# Keeping the image's torch keeps its matching flash-attn, which is both the
# fastest and the only working combination available as a prebuilt wheel: there
# is no Linux flash-attn wheel for torch 2.10 + cu128 + cp312 (only Windows).
RUN cd /opt/ml/code/Matrix-Game/Matrix-Game-3 && \
    grep -vE '^(torch|torchvision|torchaudio)([=<>!~[:space:]]|$)' requirements.txt \
        > /tmp/mg3-requirements.txt && \
    pip install --no-cache-dir -r /tmp/mg3-requirements.txt && \
    python -c "import torch, flash_attn; print('torch', torch.__version__, 'flash_attn', flash_attn.__version__)"



# Patch upstream's Ulysses attention dispatch before anything imports it.
#
# wan/distributed/ulysses.py calls attention(..., fa_version=...) but that
# function declares the keyword as `version`, so EVERY run with
# --ulysses_size > 1 raises TypeError inside the first attention block. That is
# upstream's own documented multi-GPU path, including their test.sh, and it has
# never run as shipped. Reported as issue #80; fixed by PRs #61 (2026-03-27) and
# #85 (2026-09-02), both still unmerged, so we carry the one-word fix ourselves.
#
# https://github.com/SkyworkAI/Matrix-Game/issues/80
# https://github.com/SkyworkAI/Matrix-Game/pull/61
RUN cd /opt/ml/code/Matrix-Game/Matrix-Game-3 && \
    grep -q "fa_version=fa_version," wan/distributed/ulysses.py && \
    sed -i "s/^\( *\)fa_version=fa_version,$/\1version=fa_version,/" wan/distributed/ulysses.py && \
    grep -n "version=fa_version," wan/distributed/ulysses.py && \
    echo "patched ulysses attention dispatch (upstream issue #80)"

ENV MATRIX_GAME_REPO=/opt/ml/code/Matrix-Game/Matrix-Game-3

# Upstream reads some assets by relative path — utils/visualize.py opens
# "assets/images/mouse.png" to draw the mouse overlay on each iteration's video.
# Our WORKDIR is /opt/ml/code, not the upstream repo, so cv2.imread returns None
# and the run dies with "AttributeError: 'NoneType' object has no attribute
# 'shape'" *after* generating frames. Symlinking is less invasive than chdir,
# which would be process-global inside a threaded server.
RUN ln -sfn /opt/ml/code/Matrix-Game/Matrix-Game-3/assets /opt/ml/code/assets

# Shared serving deps, including the Cognito token verifier (lib/cognito.py).
# Installed from one file so a cartridge cannot ship an image that is missing the
# auth library — that is how an endpoint once kept serving with auth unenforced
# (PCSR D484282450, round 3). A missing auth dep must break the build instead.
#
# This also repairs drift: each cartridge used to name these deps itself, and they
# had diverged (lyra-2 installed bare `uvicorn`, cosmos3-nano omitted `websockets`,
# matrix-game-3 listed none). Cartridge-specific deps stay in its requirements.txt.
COPY inference/lib/requirements.txt /tmp/lib-requirements.txt
RUN pip install --no-cache-dir -r /tmp/lib-requirements.txt

COPY inference/lib/ /opt/ml/code/lib/
COPY inference/models/matrix-game-3/requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

COPY inference/models/matrix-game-3/ /opt/ml/code/

# Run as a non-root user (M3). /tmp is world-writable for job/output scratch.
RUN useradd --create-home appuser && chown -R appuser /opt/ml/code
USER appuser

# SageMaker bidirectional streaming contract
LABEL com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true
EXPOSE 8080
WORKDIR /opt/ml/code
# Use the shared launcher, not `python -m lib.serve` directly. lib/serve is a
# shell script that counts the GPUs and execs
# `torchrun --nproc_per_node=<count> -m lib.serve`, which is what gives the
# cartridge all 8 H100s and lets the runner set ulysses_size=world_size.
# Calling python directly leaves WORLD_SIZE=1, so the model ran on one GPU and a
# chunk took ~70s instead of streaming.
#
# SageMaker launches as `docker run <image> serve` which overrides CMD but not
# ENTRYPOINT — the trailing "serve" is ignored by our handler's argv.
ENTRYPOINT ["./lib/serve"]
