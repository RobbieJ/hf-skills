# NVIDIA DGX Spark Hardware Specifications

Complete hardware specifications for NVIDIA DGX Spark, the desktop AI supercomputer with Blackwell architecture.

## Overview

The DGX Spark is a compact desktop workstation featuring NVIDIA's GB10 Grace Blackwell Superchip. It provides unified memory architecture (UMA) where CPU and GPU share the same 128GB memory pool.

## Core Specifications

| Component | Specification |
|-----------|--------------|
| **Superchip** | NVIDIA GB10 Grace Blackwell |
| **GPU Architecture** | Blackwell |
| **CUDA Cores** | 6,144 |
| **Tensor Cores** | 5th Generation |
| **RT Cores** | 4th Generation |
| **AI Performance** | 1 PFLOP FP4 sparse, 1000 TOPS |
| **CUDA Compute Capability** | 12.1 |

## CPU Specifications

| Component | Specification |
|-----------|--------------|
| **Architecture** | ARM (Grace) |
| **Total Cores** | 20 |
| **Performance Cores** | 10x Cortex-X925 |
| **Efficiency Cores** | 10x Cortex-A725 |

## Memory Specifications

| Component | Specification |
|-----------|--------------|
| **Type** | LPDDR5x |
| **Capacity** | 128GB |
| **Interface** | 256-bit |
| **Speed** | 4266 MHz |
| **Bandwidth** | 273 GB/s |
| **Architecture** | Unified Memory (shared CPU/GPU) |

### Unified Memory Architecture (UMA)

The most unique feature of DGX Spark is its Unified Memory Architecture:

- **Full 128GB accessible by GPU** - Unlike discrete GPUs where VRAM is separate, the entire 128GB is available to the GPU
- **No PCIe transfers** - CPU and GPU share memory via NVLink-C2C, eliminating slow PCIe memory copies
- **Dynamic allocation** - Memory can be used by either CPU or GPU as needed
- **Optimal for large models** - Enables loading models larger than typical GPU VRAM

## Storage

| Component | Specification |
|-----------|--------------|
| **Type** | NVMe M.2 SSD |
| **Capacity Options** | 1TB or 4TB |
| **Features** | Self-encryption |

## Connectivity

| Interface | Specification |
|-----------|--------------|
| **USB** | USB4, USB-C, USB-A |
| **Display** | DisplayPort 1.4a |
| **Network** | 10GbE (optional NVIDIA ConnectX) |
| **Expansion** | NVLink-C2C for dual-Spark |

## Power & Physical

| Specification | Value |
|---------------|-------|
| **Dimensions** | 150mm x 150mm x 50.5mm (6" x 6" x 2") |
| **Weight** | 1.2 kg (2.6 lbs) |
| **Power Supply** | 240W external |
| **SOC TDP** | 140W |
| **Cooling** | Passive/active hybrid |

## Model Capacity

### Fine-tuning Capacity

| Method | Maximum Model Size |
|--------|-------------------|
| **QLoRA 4-bit** | Up to 120B parameters |
| **LoRA 16-bit** | Up to 20B parameters |
| **Full Fine-tune** | Up to 3B parameters |

### Inference Capacity

| Configuration | Maximum Model Size |
|--------------|-------------------|
| **Single DGX Spark** | Up to 200B parameters |
| **Dual DGX Spark** | Up to 405B parameters |

## Memory Usage Estimates

### Training Memory (QLoRA 4-bit)

| Model Size | Estimated Memory |
|------------|-----------------|
| 1B | ~4GB |
| 3B | ~8GB |
| 7B | ~15GB |
| 8B | ~18GB |
| 13B | ~25GB |
| 20B | ~35GB |
| 34B | ~45GB |
| 70B | ~68GB |
| 120B | ~68GB* |

*With Unsloth optimizations

### Training Memory (LoRA 16-bit)

| Model Size | Estimated Memory |
|------------|-----------------|
| 1B | ~8GB |
| 3B | ~20GB |
| 7B | ~40GB |
| 8B | ~45GB |
| 13B | ~75GB |
| 20B | ~110GB |

### Inference Memory (FP16)

| Model Size | Estimated Memory |
|------------|-----------------|
| 7B | ~14GB |
| 13B | ~26GB |
| 34B | ~68GB |
| 70B | ~140GB* |

*Requires quantization on single DGX Spark

## Performance Benchmarks

### Training Throughput

Based on community benchmarks:

| Task | Model | Time |
|------|-------|------|
| QLoRA Fine-tune | Llama 3 8B | ~36 minutes |
| Dreambooth LoRA | FLUX.1 Dev | ~4 hours |
| SFT (1K samples) | Qwen 7B | ~30 minutes |

### Inference Performance

| Model | Quantization | Tokens/sec |
|-------|--------------|------------|
| Llama 3 8B | Q4_K_M | ~40 tok/s |
| Qwen 2.5 32B | Q4_K_M | ~15 tok/s |
| Deepseek R1 70B | Q4_K_M | ~8 tok/s |

## Comparison with Cloud GPUs

| Specification | DGX Spark | A100-40GB | A100-80GB | H100-80GB |
|--------------|-----------|-----------|-----------|-----------|
| Memory | 128GB UMA | 40GB HBM2e | 80GB HBM2e | 80GB HBM3 |
| Memory BW | 273 GB/s | 1.6 TB/s | 2.0 TB/s | 3.35 TB/s |
| CUDA Cores | 6,144 | 6,912 | 6,912 | 16,896 |
| FP16 TFLOPS | ~125 | 312 | 312 | 989 |
| Cost | $4,000 (one-time) | ~$4-8/hr | ~$8-12/hr | ~$3-4/hr |

**Key Trade-offs:**
- DGX Spark has more accessible memory but lower bandwidth
- Cloud GPUs are faster but cost money per hour
- DGX Spark excels at loading large models; cloud GPUs excel at throughput
- DGX Spark is ideal for iterative development; cloud for production training

## Software Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CUDA** | 12.9 | 13.0+ |
| **Driver** | 560+ | Latest |
| **Docker** | 24.0+ | Latest |
| **Container** | nvcr.io/nvidia/pytorch:25.09-py3 | Latest |

## Additional Resources

- [NVIDIA DGX Spark Product Page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/)
- [DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)
- [NVIDIA Build - DGX Spark](https://build.nvidia.com/spark)
