---
name: dgx-spark-trainer
description: This skill should be used when users want to train or fine-tune language models locally on NVIDIA DGX Spark hardware with Blackwell architecture. Covers Unsloth (2x faster) and LLaMA Factory (flexible YAML configs) training frameworks, supporting SFT, LoRA, and QLoRA methods. Includes guidance on Docker container setup, unified memory optimization, model size limits (up to 70B fine-tuning), and Hugging Face Hub integration. Should be invoked for tasks involving local GPU training on DGX Spark, desktop AI workstation training, or when users mention training without cloud infrastructure.
version: 1.0.0
license: Apache-2.0
---

# Local LLM Training on NVIDIA DGX Spark

## Overview

Train and fine-tune language models locally on NVIDIA DGX Spark—a desktop AI supercomputer with 128GB unified memory and Blackwell architecture. No cloud infrastructure required.

**Two recommended frameworks:**
- **Unsloth** - 2x faster training, 70% less memory, optimized for Blackwell
- **LLaMA Factory** - Flexible YAML configs, WebUI option, wide model support

**Key capabilities:**
- Fine-tune models up to 70B parameters locally
- Run inference on models up to 200B parameters
- 128GB unified memory shared between CPU and GPU
- Persistent local storage (no ephemeral environment concerns)

**See also:** `references/hardware_specs.md` for complete DGX Spark specifications

## When to Use This Skill

Use this skill when users:
- Own or have access to NVIDIA DGX Spark hardware
- Want to train models locally without cloud costs
- Need data privacy (training data stays on-device)
- Want iterative development with instant job starts
- Prefer persistent local storage over cloud ephemeral environments
- Are running Blackwell-optimized workloads

**Do NOT use this skill when:**
- User wants cloud-based training → Use `model-trainer` skill (HF Jobs)
- User doesn't have DGX Spark hardware
- User needs multi-node distributed training

## Key Directives

When assisting with DGX Spark training:

1. **Use Bash commands, NOT `hf_jobs()`** - DGX Spark runs locally via Docker containers and CLI commands. The `hf_jobs()` MCP tool is for cloud training only.

2. **Docker-first approach** - Always use NVIDIA's official PyTorch container (`nvcr.io/nvidia/pytorch:25.09-py3` or newer) for optimal performance on Blackwell architecture.

3. **Verify hardware first** - Before training, run `scripts/verify_dgx_spark.py` to confirm DGX Spark setup.

4. **Recommend Unsloth for speed** - Default to Unsloth for 2x faster training. Use LLaMA Factory when users need YAML configs or WebUI.

5. **Always use QLoRA for large models** - For models >13B, always recommend 4-bit QLoRA to fit in memory.

## Prerequisites Checklist

Before starting any training, verify:

### Hardware & System
- [ ] NVIDIA DGX Spark device (GB10 Grace Blackwell Superchip)
- [ ] CUDA 12.9+ installed (`nvcc --version`)
- [ ] GPU accessible (`nvidia-smi` shows GB10)
- [ ] Docker with GPU support (`docker run --gpus all nvidia/cuda:12.0-base nvidia-smi`)
- [ ] 50GB+ free storage for models and checkpoints

### Authentication
- [ ] Hugging Face account (for gated models and Hub uploads)
- [ ] `huggingface-cli login` completed (or HF_TOKEN environment variable set)

### Verification Command
```bash
python scripts/verify_dgx_spark.py
```

This script checks all prerequisites and reports any issues.

## Hardware Specifications

| Specification | Value |
|--------------|-------|
| **GPU** | NVIDIA GB10 Grace Blackwell Superchip |
| **CUDA Cores** | 6,144 |
| **Memory** | 128GB LPDDR5x Unified (shared CPU/GPU) |
| **Memory Bandwidth** | 273 GB/s |
| **AI Performance** | 1 PFLOP FP4 sparse |
| **CPU** | 20-core ARM (10x Cortex-X925 + 10x Cortex-A725) |
| **Storage** | 1TB or 4TB NVMe SSD |
| **CUDA Compute** | 12.1 |

**Unique advantage:** Unified Memory Architecture (UMA) allows the full 128GB to be used by either CPU or GPU without data transfers.

## Memory Guidelines by Model Size

| Model Size | QLoRA 4-bit | LoRA 16-bit | Full Fine-tune | Recommended |
|------------|-------------|-------------|----------------|-------------|
| **<3B** | ~8GB | ~15GB | ~40GB | Any method works |
| **3-7B** | ~12GB | ~28GB | ~80GB | LoRA or QLoRA |
| **7-13B** | ~20GB | ~50GB | ~120GB | QLoRA recommended |
| **13-20B** | ~35GB | ~70GB | Not feasible | QLoRA required |
| **20-70B** | ~50-68GB | ~100GB+ | Not feasible | QLoRA required |
| **70-120B** | ~68GB* | Not feasible | Not feasible | QLoRA + Unsloth only |

*With Unsloth optimizations

**Rule of thumb:** Use QLoRA 4-bit for any model >7B parameters on DGX Spark.

## Quick Start: Unsloth (Recommended)

Unsloth provides 2x faster training with 70% less memory usage.

### 1. Launch Docker Container

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 -it \
  --ulimit stack=67108864 --rm \
  -v "$PWD":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  nvcr.io/nvidia/pytorch:25.09-py3 bash
```

### 2. Install Unsloth

```bash
pip install transformers peft "datasets>=2.14.0" "trl>=0.7.0"
pip install --no-deps unsloth unsloth_zoo
pip install hf_transfer
pip install --no-deps bitsandbytes
```

### 3. Run Training

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)

# Load dataset
dataset = load_dataset("trl-lib/Capybara", split="train[:1000]")

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        output_dir="./qwen-finetuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
        learning_rate=2e-4,
        bf16=True,
    ),
)
trainer.train()

# Save locally
model.save_pretrained("./qwen-finetuned")
tokenizer.save_pretrained("./qwen-finetuned")

# Push to Hub (optional)
model.push_to_hub("username/qwen-finetuned")
tokenizer.push_to_hub("username/qwen-finetuned")
```

**See:** `scripts/train_sft_unsloth.py` for complete production-ready example

## Quick Start: LLaMA Factory

LLaMA Factory provides flexible YAML-based configuration and optional WebUI.

### 1. Launch Docker Container

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 -it \
  --ulimit stack=67108864 --rm \
  -v "$PWD":/workspace \
  nvcr.io/nvidia/pytorch:25.09-py3 bash
```

### 2. Install LLaMA Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[metrics]"
```

### 3. Login to Hugging Face (for gated models)

```bash
huggingface-cli login
```

### 4. Run Training with YAML Config

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

Or create custom config (see `templates/llamafactory_config.yaml`).

### 5. Test the Model

```bash
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
```

### 6. Export/Merge LoRA

```bash
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

**See:** `scripts/train_sft_llamafactory.py` and `references/llamafactory_guide.md` for details

## Framework Selection Guide

| Criteria | Unsloth | LLaMA Factory |
|----------|---------|---------------|
| **Training Speed** | 2x faster | Standard |
| **Memory Usage** | 70% less | Standard |
| **Configuration** | Python code | YAML files |
| **WebUI** | No | Yes (optional) |
| **Model Support** | Popular models | Very wide support |
| **Best For** | Speed, large models | Flexibility, beginners |

**Default recommendation:** Start with **Unsloth** for faster iteration, switch to LLaMA Factory if you need specific features.

## Docker Setup Details

### NVIDIA PyTorch Container (Recommended)

Always use NVIDIA's official container for Blackwell-optimized PyTorch:

```bash
docker pull nvcr.io/nvidia/pytorch:25.09-py3
```

**Important flags:**
- `--gpus all` - Enable GPU access
- `--ipc=host` - Required for PyTorch multiprocessing
- `--ulimit memlock=-1` - Unlimited locked memory
- `--ulimit stack=67108864` - Increased stack size
- `-v "$HOME/.cache/huggingface":/root/.cache/huggingface` - Persist downloaded models

### Full Docker Run Command

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 -it \
  --ulimit stack=67108864 \
  -v "$PWD":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  --rm nvcr.io/nvidia/pytorch:25.09-py3 bash
```

**See:** `references/docker_setup.md` for advanced configurations

## Saving and Deploying Models

### Local Save

Models are saved to persistent local storage (unlike cloud ephemeral environments):

```python
model.save_pretrained("./my-model")
tokenizer.save_pretrained("./my-model")
```

### Push to Hugging Face Hub

```python
model.push_to_hub("username/my-model")
tokenizer.push_to_hub("username/my-model")
```

Or via CLI:
```bash
huggingface-cli upload username/my-model ./my-model
```

### Convert to GGUF for Ollama/llama.cpp

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Convert to GGUF
python convert_hf_to_gguf.py ../my-model --outfile my-model.gguf

# Quantize (optional)
./llama-quantize my-model.gguf my-model-q4_k_m.gguf q4_k_m
```

## Memory Optimization Tips

### Clear Memory Cache

DGX Spark uses Unified Memory Architecture (UMA). Clear caches before training:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

### Reduce Memory Usage

1. **Use QLoRA 4-bit** - Required for models >7B
2. **Reduce batch size** - Start with `per_device_train_batch_size=1`
3. **Increase gradient accumulation** - `gradient_accumulation_steps=8`
4. **Enable gradient checkpointing** - `gradient_checkpointing=True`
5. **Use Unsloth** - Automatic memory optimizations

### Monitor Memory

```bash
# During training, in another terminal:
watch -n 1 nvidia-smi
```

**See:** `references/memory_optimization.md` for detailed guidance

## Common Training Patterns

### Quick Demo (5-10 minutes)

```python
# Small model, small dataset, few steps
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-1.5B-bnb-4bit",
    max_seq_length=512,
    load_in_4bit=True,
)
# Train on 100 examples, 1 epoch
```

### Development Iteration (30-60 minutes)

```python
# 7B model, 1K examples, validation monitoring
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
# Train on 1000 examples, 1-2 epochs with eval
```

### Production Training (2-8 hours)

```python
# 7-13B model, full dataset, checkpointing
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.1-8B-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)
# Train on full dataset, 3 epochs, save checkpoints
```

### Large Model Training (4-12 hours)

```python
# 70B model with QLoRA, careful memory management
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
# Uses ~68GB, train with batch_size=1, gradient_accumulation=16
```

## Troubleshooting

### CUDA Out of Memory

**Fix (try in order):**
1. Clear memory cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`
2. Reduce batch size to 1
3. Increase gradient accumulation
4. Use smaller sequence length
5. Switch to 4-bit QLoRA if using LoRA
6. Use smaller model

### Triton Compilation Errors

**Fix:** The NVIDIA container includes Blackwell-optimized Triton. If issues persist:
```bash
export TORCH_CUDA_ARCH_LIST="12.0"
pip install --upgrade triton
```

### Docker GPU Not Found

**Fix:**
```bash
# Verify nvidia-container-toolkit is installed
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU access
docker run --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Model Download Fails (Gated Models)

**Fix:**
```bash
# Login to Hugging Face
huggingface-cli login

# Or set token in environment
export HF_TOKEN="hf_your_token_here"
```

### Training Loss Not Decreasing

**Fix:**
1. Check dataset format matches model expectations
2. Reduce learning rate (try 1e-5 or 5e-6)
3. Increase warmup steps
4. Verify tokenizer matches model

**See:** `references/troubleshooting.md` for complete troubleshooting guide

## Example Training Scripts

Production-ready templates in `scripts/`:

- **`scripts/train_sft_unsloth.py`** - Complete Unsloth SFT with QLoRA
- **`scripts/train_sft_llamafactory.py`** - LLaMA Factory YAML-based training
- **`scripts/train_qlora_example.py`** - QLoRA training for large models
- **`scripts/setup_environment.sh`** - Docker environment setup
- **`scripts/verify_dgx_spark.py`** - Hardware verification

## Resources

### References (In This Skill)
- `references/hardware_specs.md` - Complete DGX Spark specifications
- `references/unsloth_guide.md` - Unsloth deep-dive and advanced usage
- `references/llamafactory_guide.md` - LLaMA Factory configuration reference
- `references/memory_optimization.md` - Unified memory best practices
- `references/docker_setup.md` - Docker container configuration
- `references/troubleshooting.md` - Common issues and solutions

### Templates (In This Skill)
- `templates/unsloth_config.yaml` - Unsloth training configuration
- `templates/llamafactory_config.yaml` - LLaMA Factory YAML config

### External Resources
- [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)
- [NVIDIA Build - LLaMA Factory](https://build.nvidia.com/spark/llama-factory)
- [NVIDIA Build - Unsloth](https://build.nvidia.com/spark/unsloth)
- [Unsloth Documentation](https://docs.unsloth.ai)
- [LLaMA Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/)

## Key Takeaways

1. **Use Bash commands** - DGX Spark is local, not cloud. No `hf_jobs()` needed.
2. **Docker-first** - Always use NVIDIA's PyTorch container for Blackwell optimization.
3. **Verify hardware** - Run `scripts/verify_dgx_spark.py` before training.
4. **QLoRA for large models** - Required for models >7B parameters.
5. **Unsloth for speed** - 2x faster training, 70% less memory.
6. **Persistent storage** - Unlike cloud, your models persist locally.
7. **128GB unified memory** - Full memory available to GPU without transfers.
8. **Clear cache** - Use memory flush command before large training runs.
