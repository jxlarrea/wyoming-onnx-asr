# wyoming-onnx-asr-gpu on DGX Spark (ARM64 / GB10 / CUDA 13)

Build guide for running [wyoming-onnx-asr-gpu](https://github.com/tboby/wyoming-onnx-asr-gpu) with GPU-accelerated ASR (Parakeet) on the NVIDIA DGX Spark.

## The Problem

The upstream `ghcr.io/tboby/wyoming-onnx-asr-gpu` Docker image is x86_64 only. On the DGX Spark (ARM64/aarch64), there is no prebuilt `onnxruntime-gpu` wheel for aarch64 with CUDA support — it must be built from source.

Key issues encountered:

- **No aarch64 onnxruntime-gpu wheel** — PyPI only has CPU wheels for aarch64
- **CUDA architecture mismatch** — GB10 is compute capability 12.1 (sm_121), requires `CMAKE_CUDA_ARCHITECTURES=121`
- **cuDNN path** — on DGX Spark Ubuntu 24.04, cuDNN headers are in `/usr` not the default CUDA path; requires `--cudnn_home /usr`
- **numpy version conflict** — onnxruntime build requires `numpy<2`

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

This is the critical step. The build takes approximately 1–2 hours on the Grace cores.

### Prerequisites

```bash
sudo apt install -y cmake python3-dev python3-pip git
```

### Build

```bash
# Clone onnxruntime
git clone --recursive https://github.com/microsoft/onnxruntime.git ~/onnxruntime-src
cd ~/onnxruntime-src

# Build with CUDA + cuDNN for GB10 (sm_121)
# Key flags:
#   --CMAKE_CUDA_ARCHITECTURES=121  (GB10 compute capability)
#   --cudnn_home /usr               (cuDNN location on DGX Spark Ubuntu 24.04)
#   numpy<2                         (build constraint)
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
```

### Important Build Notes

- `--cudnn_home /usr` is required because on DGX Spark (Ubuntu 24.04), cuDNN is installed system-wide under `/usr/include` and `/usr/lib/aarch64-linux-gnu`, not under `/usr/local/cuda`. Without this flag, cmake fails to find cuDNN.
- `CMAKE_CUDA_ARCHITECTURES=121` targets the GB10's exact compute capability. Using the default or a lower value (e.g., 80, 90) will result in missing kernels at runtime.
- `numpy<2` — the onnxruntime build system has an incompatibility with numpy 2.x. Pin numpy before building: `pip install "numpy<2"`.

## Step 2: Build the Docker Image

The custom Docker image packages the locally-built onnxruntime-gpu wheel into the wyoming-onnx-asr-gpu container.

### Dockerfile

```dockerfile
FROM nvidia/cuda:13.0.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev curl git \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the locally-built onnxruntime-gpu wheel
COPY onnxruntime_gpu*.whl /tmp/

# Install onnxruntime-gpu from local wheel
RUN pip3 install --no-cache-dir "numpy<2" && \
    pip3 install --no-cache-dir /tmp/onnxruntime_gpu*.whl && \
    rm /tmp/onnxruntime_gpu*.whl

# Install wyoming-onnx-asr
RUN pip3 install --no-cache-dir wyoming-onnx-asr

# Create data directory for models
RUN mkdir -p /data

EXPOSE 10300

ENTRYPOINT ["python3", "-m", "wyoming_onnx_asr"]
CMD ["--device", "gpu", "--uri", "tcp://0.0.0.0:10300", "--model-dir", "/data"]
```

### Build and Tag

```bash
# Copy the wheel into the build context
cp ~/onnxruntime-src/build/Linux/Release/dist/onnxruntime_gpu*.whl .

# Build
docker build -t ghcr.io/jxlarrea/wyoming-onnx-asr-gpu:latest .

# Push (optional)
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
    environment:
      - TZ=America/Guayaquil
    command: >
      --device gpu
      --model-en nemo-parakeet-tdt-0.6b-v2
      --uri tcp://0.0.0.0:10300
      --model-dir /data
    volumes:
      - /opt/models/asr:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
```

```bash
docker compose up -d
```

## Step 4: Verify

```bash
# Check logs
docker logs wyoming-onnx-asr

# Verify CUDA provider is active
docker exec wyoming-onnx-asr python3 -c \
    "import onnxruntime; print(onnxruntime.get_available_providers())"
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

## Performance

With Parakeet TDT 0.6B on the GB10 via CUDA execution provider:

- "What time is it?" → ~33ms inference
- Uses approximately 1GB GPU memory
- The CUDA execution provider (not TensorRT) is recommended — TensorRT provides minimal performance improvement for ASR while adding significant complexity and VRAM overhead during engine building

## Troubleshooting

### `onnxruntime` not finding CUDA

Ensure the container has GPU access and the NVIDIA Container Toolkit is installed:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu22.04 nvidia-smi
```

### cuDNN not found during build

On DGX Spark Ubuntu 24.04, cuDNN is installed under `/usr`, not `/usr/local/cuda`:

```bash
ls /usr/include/cudnn*.h        # Headers
ls /usr/lib/aarch64-linux-gnu/libcudnn*  # Libraries
```

Use `--cudnn_home /usr` in the onnxruntime build command.

### TensorRT variant issues

The TensorRT execution provider (`gpu-trt`) can fill all available memory during engine building. The default workspace size is insufficient for models like Parakeet, and increasing it to 12GB+ is needed. For Home Assistant voice use, stick with the regular CUDA provider (`gpu`) — performance difference is negligible (~33ms vs ~30ms) and it avoids all TensorRT complexity.
