FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder-base
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0
WORKDIR /app

FROM builder-base AS packages-builder
ARG TARGETARCH

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    if [ "$TARGETARCH" = "arm64" ]; then \
        uv sync --locked --no-install-project --no-dev -v ; \
    else \
        uv sync --locked --no-install-project --no-dev --extra gpu -v ; \
    fi

RUN if [ "$TARGETARCH" = "arm64" ]; then \
        uv pip install --python /app/.venv/bin/python \
            https://github.com/jxlarrea/wyoming-onnx-asr/releases/download/v0.5.0/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl ; \
        uv pip install --python /app/.venv/bin/python 'numpy<2' ; \
    fi

FROM builder-base AS app-builder
COPY wyoming_onnx_asr/ ./wyoming_onnx_asr/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv venv && \
    uv pip install --no-deps .

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=packages-builder /app/.venv /app/.venv
COPY --from=app-builder /app/.venv/lib/python3.12/site-packages /app/.venv/lib/python3.12/site-packages/
COPY wyoming_onnx_asr/ /app/wyoming_onnx_asr/
RUN ln -sf /usr/bin/python3.12 /app/.venv/bin/python
ENV PATH="/app/.venv/bin:$PATH"
VOLUME /data
ENV ONNX_ASR_MODEL_DIR="/data"
ENTRYPOINT ["python3.12", "-m", "wyoming_onnx_asr", "--device", "gpu"]
CMD [ "--uri", "tcp://*:10300", "--model-en", "nemo-parakeet-tdt-0.6b-v2" ]