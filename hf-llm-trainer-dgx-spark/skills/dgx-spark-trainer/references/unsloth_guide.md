# Unsloth Training Guide for DGX Spark

Comprehensive guide to using Unsloth for optimized LLM training on NVIDIA DGX Spark.

## Overview

Unsloth provides 2x faster training with 70% less memory usage through:
- Custom Triton kernels optimized for modern GPUs
- Efficient gradient checkpointing
- Optimized attention implementations
- Native 4-bit quantization support

## Key Benefits on DGX Spark

| Feature | Benefit |
|---------|---------|
| **2x Faster** | Reduced training time |
| **70% Less Memory** | Train larger models |
| **Blackwell Support** | Optimized for GB10 |
| **Unified Memory** | Leverages 128GB UMA |

## Installation

### Inside NVIDIA Container

```bash
# Start container
docker run --gpus all --ipc=host --ulimit memlock=-1 -it \
    --ulimit stack=67108864 --rm \
    -v "$PWD":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    nvcr.io/nvidia/pytorch:25.09-py3 bash

# Install dependencies
pip install transformers peft "datasets>=2.14.0" "trl>=0.7.0"
pip install --no-deps unsloth unsloth_zoo
pip install hf_transfer
pip install --no-deps bitsandbytes
```

### Custom Docker Image (Advanced)

For production use, build a custom image:

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.09-py3

# Build Triton from source for Blackwell
RUN pip install --upgrade triton

# Build xformers for CUDA 12.1
ENV TORCH_CUDA_ARCH_LIST="12.0"
RUN pip install xformers --no-build-isolation

# Install Unsloth
RUN pip install transformers peft datasets trl hf_transfer
RUN pip install --no-deps unsloth unsloth_zoo bitsandbytes
```

## Pre-quantized Models

Unsloth provides pre-quantized 4-bit models that load faster:

```python
# Unsloth pre-quantized models (recommended)
"unsloth/Qwen2.5-0.5B-bnb-4bit"
"unsloth/Qwen2.5-1.5B-bnb-4bit"
"unsloth/Qwen2.5-3B-bnb-4bit"
"unsloth/Qwen2.5-7B-bnb-4bit"
"unsloth/Qwen2.5-14B-bnb-4bit"
"unsloth/Qwen2.5-32B-bnb-4bit"
"unsloth/Qwen2.5-72B-bnb-4bit"

"unsloth/Llama-3.1-8B-bnb-4bit"
"unsloth/Llama-3.1-8B-Instruct-bnb-4bit"
"unsloth/Llama-3.1-70B-bnb-4bit"
"unsloth/Llama-3.1-70B-Instruct-bnb-4bit"
"unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

"unsloth/Mistral-7B-v0.3-bnb-4bit"
"unsloth/Mistral-Nemo-Base-2407-bnb-4bit"
"unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit"

"unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit"
"unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit"
"unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit"
```

## Basic Training Pattern

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1. Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 2. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # LoRA rank
    lora_alpha=32,           # LoRA alpha
    target_modules=[         # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",  # Optimized checkpointing
)

# 3. Load dataset
dataset = load_dataset("trl-lib/Capybara", split="train")

# 4. Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
    ),
)
trainer.train()

# 5. Save
model.save_pretrained("./output")
tokenizer.save_pretrained("./output")
```

## Advanced Configuration

### LoRA Target Modules

Different models have different module names:

```python
# Qwen, Llama, Mistral
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# All linear layers (automatic)
target_modules = "all-linear"
```

### LoRA Rank Selection

| Model Size | Recommended Rank | Memory Impact |
|------------|-----------------|---------------|
| <3B | r=32, alpha=64 | Low |
| 3-7B | r=16, alpha=32 | Medium |
| 7-13B | r=16, alpha=32 | Medium |
| 13-70B | r=8, alpha=16 | High (prefer lower) |
| >70B | r=4-8, alpha=8-16 | Very High |

### Gradient Checkpointing Options

```python
# Unsloth optimized (recommended)
use_gradient_checkpointing="unsloth"

# Standard PyTorch
use_gradient_checkpointing=True

# Disabled (more memory, faster)
use_gradient_checkpointing=False
```

### Optimizer Selection

```python
# 8-bit Adam (memory efficient)
optim="adamw_8bit"

# Standard Adam
optim="adamw_torch"

# Fused Adam (faster if available)
optim="adamw_torch_fused"
```

## Memory Optimization

### For Very Large Models (70B+)

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    max_seq_length=2048,  # Reduce if needed
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,               # Lower rank
    lora_alpha=16,     # Lower alpha
    target_modules=[   # Fewer modules
        "q_proj", "k_proj", "v_proj", "o_proj",
    ],
    use_gradient_checkpointing="unsloth",
)

# Training config
SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_8bit",
)
```

### Clearing Memory Cache

Before loading large models:

```bash
# From terminal
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

```python
# From Python
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

## Inference After Training

### Direct Inference

```python
from unsloth import FastLanguageModel

# Load trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    "./output",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Enable fast inference
FastLanguageModel.for_inference(model)

# Generate
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

### Streaming Inference

```python
from transformers import TextStreamer

streamer = TextStreamer(tokenizer)
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    streamer=streamer,
)
```

## Exporting Models

### Save in Different Formats

```python
# Standard HuggingFace format
model.save_pretrained("./output")

# Merged model (LoRA + base)
model.save_pretrained_merged("./merged", tokenizer)

# GGUF for llama.cpp
model.save_pretrained_gguf("./gguf", tokenizer, quantization_method="q4_k_m")

# Push to Hub
model.push_to_hub("username/model-name")
model.push_to_hub_gguf("username/model-name-gguf", quantization_method="q4_k_m")
```

### GGUF Quantization Methods

| Method | Size | Quality | Use Case |
|--------|------|---------|----------|
| `q4_k_m` | Smallest | Good | CPU inference |
| `q5_k_m` | Medium | Better | Balanced |
| `q8_0` | Larger | Best | High quality |
| `f16` | Largest | Lossless | Reference |

## Reinforcement Learning

Unsloth supports RL training on DGX Spark:

```python
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32)

# DPO Training
trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=preference_dataset,
    args=DPOConfig(
        output_dir="./dpo-output",
        beta=0.1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        bf16=True,
    ),
)
trainer.train()
```

## Troubleshooting

### Triton Errors

```bash
# Set compute capability
export TORCH_CUDA_ARCH_LIST="12.0"

# Reinstall Triton
pip install --upgrade triton
```

### xformers Issues

```bash
# Build from source
pip install xformers --no-build-isolation
```

### Out of Memory

1. Reduce sequence length: `max_seq_length=1024`
2. Reduce batch size: `per_device_train_batch_size=1`
3. Increase gradient accumulation
4. Lower LoRA rank: `r=4` or `r=8`
5. Use fewer target modules

### Slow Training

1. Ensure using NVIDIA container
2. Check `bf16=True` is set
3. Verify `use_gradient_checkpointing="unsloth"`
4. Use pre-quantized Unsloth models

## Resources

- [Unsloth Documentation](https://docs.unsloth.ai)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth DGX Spark Guide](https://docs.unsloth.ai/basics/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth)
- [NVIDIA Unsloth Playbook](https://build.nvidia.com/spark/unsloth)
