FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

ENV SM_MODEL_DIR=/opt/ml/model
ENV SM_MODULE_DIR=/opt/ml/code
ENV SAGEMAKER_BIND_TO_PORT=8080

RUN pip install --no-cache-dir --upgrade numpy && \
    pip install --no-cache-dir \
    "diffusers @ git+https://github.com/huggingface/diffusers.git@6afdfd9f4e2e2381c5d5cced0dc4262e9e4267bd" \
    accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg \
    transformers fastapi "uvicorn[standard]" python-multipart pillow

COPY inference/lib/ /opt/ml/code/lib/
COPY inference/models/cosmos3-nano/ /opt/ml/code/

# Run as a non-root user (M3). /tmp is world-writable for job/output scratch.
RUN useradd --create-home appuser && chown -R appuser /opt/ml/code
USER appuser

WORKDIR /opt/ml/code
EXPOSE 8080
ENTRYPOINT ["./lib/serve"]
