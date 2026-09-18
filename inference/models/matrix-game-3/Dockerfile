FROM nvcr.io/nvidia/pytorch:26.04-py3

# Clone the official Matrix-Game repo (Matrix-Game-3.0 lives in Matrix-Game-3/)
RUN git clone https://github.com/SkyworkAI/Matrix-Game.git /opt/ml/code/Matrix-Game

RUN cd /opt/ml/code/Matrix-Game/Matrix-Game-3 && \
    pip install --no-cache-dir -r requirements.txt

ENV MATRIX_GAME_REPO=/opt/ml/code/Matrix-Game/Matrix-Game-3

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
# SageMaker launches as `docker run <image> serve` which overrides CMD but not
# ENTRYPOINT — the trailing "serve" is ignored by our handler's argv.
ENTRYPOINT ["python", "-m", "lib.serve"]
