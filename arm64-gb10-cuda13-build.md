# wyoming-onnx-asr-gpu on DGX Spark (ARM64 / GB10 / CUDA 13)

Build guide for running [wyoming-onnx-asr](https://github.com/jxlarrea/wyoming-onnx-asr) with GPU-accelerated ASR (Parakeet) on the NVIDIA DGX Spark.

Published image: `ghcr.io/jxlarrea/wyoming-onnx-asr-gpu:latest`

## The Problem

There is no prebuilt `onnxruntime-gpu` wheel for aarch64 on PyPI. On the DGX Spark, it must be built from source with the correct CUDA architecture target (sm_121 / compute capability 12.1 for the GB10).

Key issues encountered during the build:

- **No aarch64 onnxruntime-gpu wheel on PyPI** — only CPU wheels exist
- **CUDA architecture** — GB10 is sm_121; requires `CMAKE_CUDA_ARCHITECTURES=121`
- **cuDNN path** — on DGX Spark (Ubuntu 24.04), cuDNN is installed system-wide under `/usr`, not under `/usr/local/cuda`; the build requires `--cudnn_home /usr`
- **numpy 2.x incompatibility** — onnxruntime build fails with numpy ≥2; must pin `numpy<2`

## Hardware & Software

| Component | Detail |
|---|---|
| Hardware | NVIDIA DGX Spark (GB10 Blackwell, 128GB unified LPDDR5x) |
| Architecture | ARM64 / aarch64 |
| OS | Ubuntu 24.04 |
| CUDA Toolkit | 13.0 |
| GPU Compute | sm_121 (Blackwell consumer) |
| ASR Model | nemo-parakeet-tdt-0.6b-v2 |

## Step 1: Build onnxruntime-gpu from Source

This is the critical prerequisite. The build takes approximately 1–2 hours on the Grace cores.

```bash
sudo apt install -y cmake python3-dev python3-pip git

# Clone onnxruntime
git clone --recursive https://github.com/microsoft/onnxruntime.git ~/onnxruntime-src
cd ~/onnxruntime-src

# Pin numpy before building
pip install "numpy<2"

# Build with CUDA + cuDNN for GB10 (sm_121)
./build.sh --config Release \
    --build_wheel \
    --use_cuda \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr \
    --CMAKE_CUDA_ARCHITECTURES=121 \
    --parallel \
    --skip_tests

# The wheel will be in build/Linux/Release/dist/
ls build/Linux/Release/dist/onnxruntime_gpu*.whl
# → onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl
```

### Build Notes

- `--cudnn_home /usr` — on DGX Spark Ubuntu 24.04, cuDNN headers are at `/usr/include/cudnn*.h` and libraries at `/usr/lib/aarch64-linux-gnu/libcudnn*`. Without this flag, cmake cannot find cuDNN.
- `CMAKE_CUDA_ARCHITECTURES=121` — targets the GB10's exact compute capability. Using defaults or lower values results in missing CUDA kernels at runtime.
- `numpy<2` — the onnxruntime build has an incompatibility with numpy 2.x.

### Publish the Wheel

Upload the built wheel as a GitHub release asset so the Docker build can fetch it without needing a local copy:

```bash
# Example: uploaded to
# https://github.com/jxlarrea/wyoming-onnx-asr/releases/download/v0.5.0/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl
```

## Step 2: The Dockerfile

The Dockerfile uses a multi-stage build. On ARM64, it skips the PyPI `onnxruntime-gpu` (which doesn't exist for aarch64) and instead installs from the pre-built wheel hosted on GitHub Releases.

### gpu.Dockerfile

```dockerfile
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

# On ARM64, install onnxruntime-gpu from pre-built wheel (no PyPI wheel available)
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        uv pip install --python /app/.venv/bin/python \
            https://github.com/jxlarrea/wyoming-onnx-asr/releases/download/v0.5.0/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl && \
        uv pip install --python /app/.venv/bin/python 'numpy<2' ; \
    fi

FROM builder-base AS app-builder
COPY wyoming_onnx_asr/ ./wyoming_onnx_asr/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv venv && \
    uv pip install --no-deps .

FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

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
```

### Key Design Decisions

- **Multi-stage build** — uses `uv` (astral-sh) for fast, locked dependency resolution in the builder stages, then copies only the venv into the slim CUDA runtime image.
- **`TARGETARCH` conditional** — on x86_64, `uv sync --extra gpu` pulls the normal PyPI `onnxruntime-gpu`. On ARM64, it skips that extra and instead installs the custom-built wheel from GitHub Releases + pins `numpy<2`.
- **Final stage base image** — `nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04` provides the CUDA 13.0 + cuDNN runtime libraries matching the DGX Spark's toolkit.
- **Device flag baked in** — `ENTRYPOINT` includes `--device gpu` so the CUDA execution provider is always used.

### Build

```bash
# From the wyoming-onnx-asr repo root
docker build -f gpu.Dockerfile -t ghcr.io/jxlarrea/wyoming-onnx-asr-gpu:latest .

# Push
docker push ghcr.io/jxlarrea/wyoming-onnx-asr-gpu:latest
```

## Step 3: docker-compose.yml

```yaml
services:
  wyoming-onnx-asr:
    image: ghcr.io/jxlarrea/wyoming-onnx-asr-gpu:latest
    container_name: wyoming-onnx-asr
    restart: unless-stopped
    ports:
      - "10300:10300"
    volumes:
      - asr-models:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

volumes:
  asr-models:
```

```bash
docker compose up -d
```

## Step 4: Verify

```bash
# Check logs — should show model loading on CUDA
docker logs wyoming-onnx-asr

# Verify CUDA provider is active
docker exec wyoming-onnx-asr python3.12 -c \
    "import onnxruntime; print(onnxruntime.get_available_providers())"
# → ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

## Performance

With Parakeet TDT 0.6B on the GB10 via CUDA execution provider:

- "What time is it?" → ~33ms inference
- ~1GB GPU memory usage
- CUDA execution provider (not TensorRT) is recommended — TensorRT adds significant complexity and VRAM overhead during engine building with negligible performance improvement for ASR workloads

## Home Assistant Integration

1. Go to **Settings → Devices & Services → Add Integration**
2. Search for **Wyoming Protocol**
3. Enter the DGX Spark's IP and port **10300**
4. Configure your Assist voice pipeline to use the new STT provider

## Troubleshooting

### cuDNN not found during onnxruntime build

On DGX Spark Ubuntu 24.04, cuDNN is under `/usr`, not `/usr/local/cuda`:

```bash
ls /usr/include/cudnn*.h                    # Headers
ls /usr/lib/aarch64-linux-gnu/libcudnn*     # Libraries
```

Pass `--cudnn_home /usr` to the onnxruntime `build.sh`.

### numpy ≥2 breaks the build

```bash
pip install "numpy<2"
```

Must be pinned before running `build.sh`. Also pinned in the Dockerfile's ARM64 path.

### TensorRT fills all memory

The TensorRT execution provider (`gpu-trt`) requires massive workspace during engine building. Stick with `--device gpu` (plain CUDA) for Home Assistant voice — the performance difference is negligible.
