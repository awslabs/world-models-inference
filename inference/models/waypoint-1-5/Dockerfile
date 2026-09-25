# Waypoint-1.5-1B — Overworld's real-time interactive world model.
#
# Its own base image on purpose: upstream `world_engine` pins torch 2.11 +
# triton 3.6 + transformers 5.x, all newer than any NGC image the shared
# inference/Dockerfile.default supports. Rather than fight an NGC base's
# preinstalled stack, start from a slim CUDA runtime and install the exact
# wheels upstream develops against (the same combination Overworld's Biome
# server pins: torch 2.11.0 from the cu128 index).
#
# The cu128 torch wheels bundle their own CUDA libraries and include sm_120
# kernels, so they run on the g7e's RTX PRO 6000 Blackwell as well as Ada/
# Hopper parts; only the host driver matters, which the DLAMI provides.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# torch first, from the cu128 index, so the world_engine install below sees
# its `torch==2.11.0` pin already satisfied and cannot swap in PyPI's default
# build (which targets a newer CUDA than some host drivers ship).
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# Pin the upstream release. Same tag Biome 1.1.x pins; an unpinned install
# means the image silently changes under us between builds.
ARG WORLD_ENGINE_REF=1.5.6
RUN pip install --no-cache-dir \
    "world_engine @ https://github.com/Overworldai/world_engine/archive/refs/tags/${WORLD_ENGINE_REF}.zip"

# Serving deps the shared FastAPI app needs, plus fast JPEG for the frame path.
RUN pip install --no-cache-dir \
        fastapi "uvicorn[standard]" python-multipart websockets \
        imageio imageio-ffmpeg pillow numpy simplejpeg

# Triton JIT-compiles its CUDA launcher stubs with the host C compiler at
# runtime ("RuntimeError: Failed to find C compiler" without this). Kept as
# its own layer, after the heavy pip installs, so image iterations stay cached.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends gcc libc6-dev python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Weights, config and the TAEHV autoencoder are all staged to /opt/ml/model —
# nothing may be fetched from the Hub at runtime.
ENV HF_HUB_OFFLINE=1

# Single-GPU model: keep torchrun to one rank even on a multi-GPU host.
ENV NPROC_PER_NODE=1

# Runtime knobs (see runner.py). Defaults favour throughput on Ada/Blackwell.
ENV WAYPOINT_QUANT=fp8w8a8
ENV WAYPOINT_MOUSE_SCALE=15
ENV WAYPOINT_WARMUP=1

# triton's JIT cache must live somewhere the non-root user can write.
ENV TRITON_CACHE_DIR=/tmp/triton-cache
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache
ENV HOME=/home/appuser

COPY inference/lib/ /opt/ml/code/lib/
COPY inference/models/waypoint-1-5/ /opt/ml/code/

# Run as a non-root user (M3). /tmp is world-writable for job/output scratch.
RUN useradd --create-home appuser && chown -R appuser /opt/ml/code
USER appuser

# SageMaker bidirectional streaming contract
LABEL com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
EXPOSE 8080
WORKDIR /opt/ml/code
ENTRYPOINT ["./lib/serve"]
