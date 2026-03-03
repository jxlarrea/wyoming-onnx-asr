FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder-base
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0
WORKDIR /app

FROM builder-base AS packages-builder
ARG TARGETARCH

COPY wheels/ /tmp/wheels/

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    if [ "$TARGETARCH" = "arm64" ]; then \
        uv sync --locked --no-install-project --no-dev -v ; \
    else \
        uv sync --locked --no-install-project --no-dev --extra gpu -v ; \
    fi

RUN if [ "$TARGETARCH" = "arm64" ]; then \
        /app/.venv/bin/pip install /tmp/wheels/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl ; \
    fi && \
    rm -rf /tmp/wheels

FROM builder-base AS app-builder
COPY wyoming_onnx_asr/ ./wyoming_onnx_asr/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv venv && \
    uv pip install --no-deps .

FROM python:3.12-slim-bookworm
WORKDIR /app
COPY --from=packages-builder --chown=app:app /app/.venv /app/.venv
COPY --from=app-builder --chown=app:app /app/.venv/lib/python3.12/site-packages /app/.venv/lib/python3.12/site-packages/
COPY wyoming_onnx_asr/ /app/wyoming_onnx_asr/
ENV PATH="/app/.venv/bin:$PATH"
VOLUME /data
ENV ONNX_ASR_MODEL_DIR="/data"
ENTRYPOINT ["python", "-m", "wyoming_onnx_asr", "--device", "gpu"]
CMD [ "--uri", "tcp://*:10300", "--model-en", "nemo-parakeet-tdt-0.6b-v2" ]