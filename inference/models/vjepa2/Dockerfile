FROM nvcr.io/nvidia/pytorch:26.04-py3

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
LABEL com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true

ENV SM_MODEL_DIR=/opt/ml/model
ENV SM_MODULE_DIR=/opt/ml/code
ENV SAGEMAKER_BIND_TO_PORT=8080

RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" python-multipart websockets \
    transformers huggingface_hub safetensors accelerate Pillow

COPY inference/lib/ /opt/ml/code/lib/
COPY inference/models/vjepa2/ /opt/ml/code/

# Run as a non-root user (M3). /tmp is world-writable for job/output scratch.
RUN useradd --create-home appuser && chown -R appuser /opt/ml/code
USER appuser

WORKDIR /opt/ml/code
EXPOSE 8080
ENTRYPOINT ["./lib/serve"]
