# Docker Setup Guide for DGX Spark

Complete guide to configuring Docker containers for LLM training on DGX Spark.

## Prerequisites

### Install Docker

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y docker.io

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (logout/login required)
sudo usermod -aG docker $USER
```

### Install NVIDIA Container Toolkit

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

### Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## NVIDIA PyTorch Container

### Official Container (Recommended)

NVIDIA's PyTorch container is optimized for Blackwell architecture:

```bash
docker pull nvcr.io/nvidia/pytorch:25.09-py3
```

**What's included:**
- PyTorch optimized for NVIDIA GPUs
- CUDA 12.x with Blackwell support
- cuDNN, NCCL, TensorRT
- Triton for kernel compilation
- Development tools

### Container Versions

| Tag | CUDA | PyTorch | Notes |
|-----|------|---------|-------|
| `25.09-py3` | 12.9 | 2.5 | Recommended for DGX Spark |
| `25.10-py3` | 13.0 | 2.5 | Latest |
| `24.12-py3` | 12.6 | 2.4 | Stable fallback |

## Running Containers

### Basic Run

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.09-py3 bash
```

### Full Configuration (Recommended)

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 -it \
    --ulimit stack=67108864 \
    -v "$PWD":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -w /workspace \
    --rm nvcr.io/nvidia/pytorch:25.09-py3 bash
```

### Flag Explanations

| Flag | Purpose |
|------|---------|
| `--gpus all` | Enable all GPUs |
| `--ipc=host` | Share IPC namespace (required for PyTorch DataLoader) |
| `--ulimit memlock=-1` | Unlimited locked memory (required for large models) |
| `--ulimit stack=67108864` | Increased stack size (64MB) |
| `-v "$PWD":/workspace` | Mount current directory |
| `-v "$HOME/.cache/huggingface":/root/.cache/huggingface` | Persist model cache |
| `-e HF_TOKEN` | Pass Hugging Face token |
| `-w /workspace` | Set working directory |
| `--rm` | Remove container on exit |

## Volume Mounts

### Essential Mounts

```bash
# Working directory
-v "$PWD":/workspace

# Hugging Face cache (persist downloaded models)
-v "$HOME/.cache/huggingface":/root/.cache/huggingface

# Output directory (if different from PWD)
-v "$HOME/models":/models
```

### Additional Mounts

```bash
# Custom datasets
-v "$HOME/datasets":/datasets

# Checkpoints
-v "$HOME/checkpoints":/checkpoints

# Logs
-v "$HOME/logs":/logs
```

## Environment Variables

### Hugging Face

```bash
-e HF_TOKEN="${HF_TOKEN}"           # Authentication token
-e HF_HOME="/root/.cache/huggingface"  # Cache directory
-e HF_HUB_OFFLINE=1                  # Offline mode (use cached only)
-e HF_HUB_ENABLE_HF_TRANSFER=1       # Fast downloads
```

### CUDA/PyTorch

```bash
-e CUDA_VISIBLE_DEVICES=0           # Select GPU
-e TORCH_CUDA_ARCH_LIST="12.0"      # CUDA architecture
-e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"  # Memory allocator
```

### Weights & Biases (Optional)

```bash
-e WANDB_API_KEY="${WANDB_API_KEY}"
-e WANDB_PROJECT="my-project"
```

## Container Scripts

### Start Script

Create `start_training.sh`:

```bash
#!/bin/bash
# start_training.sh - Start DGX Spark training container

CONTAINER="nvcr.io/nvidia/pytorch:25.09-py3"
WORKSPACE="${WORKSPACE:-$(pwd)}"
HF_CACHE="${HOME}/.cache/huggingface"

docker run --gpus all --ipc=host --ulimit memlock=-1 -it \
    --ulimit stack=67108864 \
    -v "$WORKSPACE":/workspace \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -w /workspace \
    --rm "$CONTAINER" bash
```

### Training Script Wrapper

Create `run_training.sh`:

```bash
#!/bin/bash
# run_training.sh - Run training inside container

CONTAINER="nvcr.io/nvidia/pytorch:25.09-py3"
SCRIPT="${1:-train.py}"

docker run --gpus all --ipc=host --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$(pwd)":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -w /workspace \
    --rm "$CONTAINER" \
    bash -c "pip install -q transformers peft datasets trl && \
             pip install -q --no-deps unsloth unsloth_zoo bitsandbytes && \
             python $SCRIPT"
```

## Custom Docker Images

### Unsloth Image

Create `Dockerfile.unsloth`:

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.09-py3

# Install dependencies
RUN pip install --no-cache-dir \
    transformers \
    peft \
    "datasets>=2.14.0" \
    "trl>=0.7.0" \
    hf_transfer

# Install Unsloth
RUN pip install --no-cache-dir --no-deps unsloth unsloth_zoo bitsandbytes

# Set environment
ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace
```

Build and run:

```bash
docker build -f Dockerfile.unsloth -t dgx-spark-unsloth .
docker run --gpus all --ipc=host -it --rm \
    -v "$PWD":/workspace dgx-spark-unsloth bash
```

### LLaMA Factory Image

Create `Dockerfile.llamafactory`:

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.09-py3

# Clone LLaMA Factory
RUN git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /opt/LLaMA-Factory

# Install
WORKDIR /opt/LLaMA-Factory
RUN pip install --no-cache-dir -e ".[metrics]"

# Set environment
ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace
```

## Networking

### Expose Ports

```bash
# For WebUI (LLaMA Factory, Gradio)
-p 7860:7860

# For Jupyter
-p 8888:8888

# For TensorBoard
-p 6006:6006
```

### Example with Jupyter

```bash
docker run --gpus all --ipc=host -it \
    -p 8888:8888 \
    -v "$PWD":/workspace \
    nvcr.io/nvidia/pytorch:25.09-py3 \
    bash -c "pip install jupyter && jupyter notebook --ip=0.0.0.0 --allow-root"
```

## Persistent Containers

### Named Container

```bash
# Create named container
docker run --gpus all --ipc=host --name dgx-training \
    -v "$PWD":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -d nvcr.io/nvidia/pytorch:25.09-py3 sleep infinity

# Attach to container
docker exec -it dgx-training bash

# Stop container
docker stop dgx-training

# Start container
docker start dgx-training
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  training:
    image: nvcr.io/nvidia/pytorch:25.09-py3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    volumes:
      - .:/workspace
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HF_TOKEN=${HF_TOKEN}
    working_dir: /workspace
    stdin_open: true
    tty: true
```

Run:

```bash
docker-compose run training bash
```

## Troubleshooting

### GPU Not Found

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# If fails, reinstall nvidia-container-toolkit
sudo apt-get install --reinstall nvidia-container-toolkit
sudo systemctl restart docker
```

### Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or:
newgrp docker
```

### Out of Disk Space

```bash
# Clean unused images
docker system prune -a

# Remove specific images
docker rmi $(docker images -q)
```

### Container Exits Immediately

```bash
# Check logs
docker logs <container_id>

# Run interactively to debug
docker run --gpus all -it nvcr.io/nvidia/pytorch:25.09-py3 bash
```

### Slow Model Downloads

```bash
# Enable fast transfers
-e HF_HUB_ENABLE_HF_TRANSFER=1

# Or use offline mode with pre-downloaded models
-e HF_HUB_OFFLINE=1
```
