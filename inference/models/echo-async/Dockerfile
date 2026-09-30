# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/pytorch:23.10-py3

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

ENV SM_MODEL_DIR=/opt/ml/model
ENV SM_MODULE_DIR=/opt/ml/code
ENV SAGEMAKER_BIND_TO_PORT=8080

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart imageio imageio-ffmpeg

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
COPY inference/models/echo-async/ /opt/ml/code/

# Run as a non-root user (M3). /tmp is world-writable for job/output scratch.
RUN useradd --create-home appuser && chown -R appuser /opt/ml/code
USER appuser

WORKDIR /opt/ml/code
EXPOSE 8080
# Container health: /ping is unauthenticated and cheap. Model load can take many
# minutes on first boot, hence the long start period; the status is advisory
# (ALB/SageMaker run their own /ping probes), nothing restarts on it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/ping', timeout=4).status == 200 else 1)"
ENTRYPOINT ["./lib/serve"]
