# LLaMA Factory Guide for DGX Spark

Comprehensive guide to using LLaMA Factory for flexible LLM training on NVIDIA DGX Spark.

## Overview

LLaMA Factory provides:
- YAML-based configuration for reproducible training
- Optional WebUI for visual configuration
- Support for 100+ model architectures
- Multiple training methods (SFT, DPO, PPO, RLHF)
- Built-in dataset management

## Key Features

| Feature | Description |
|---------|-------------|
| **YAML Configs** | Declarative, version-controlled training |
| **WebUI** | Visual configuration and monitoring |
| **Model Support** | LLaMA, Qwen, Mistral, Yi, Gemma, etc. |
| **Methods** | SFT, DPO, ORPO, PPO, KTO, Reward Modeling |
| **Quantization** | GPTQ, AWQ, bitsandbytes (4-bit, 8-bit) |

## Installation

### Docker Setup (Recommended)

```bash
# Start NVIDIA container
docker run --gpus all --ipc=host --ulimit memlock=-1 -it \
    --ulimit stack=67108864 --rm \
    -v "$PWD":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -p 7860:7860 \
    nvcr.io/nvidia/pytorch:25.09-py3 bash

# Clone and install
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[metrics]"

# Verify installation
llamafactory-cli version
```

### Login for Gated Models

```bash
huggingface-cli login
```

## Quick Start

### Using Built-in Examples

```bash
cd LLaMA-Factory

# List available examples
ls examples/train_lora/

# Train with example config
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

# Test the model
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml

# Export merged model
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

### WebUI (Optional)

```bash
# Start WebUI
llamafactory-cli webui

# Access at http://localhost:7860
```

## YAML Configuration

### Basic SFT Config

```yaml
# train_config.yaml
### Model
model_name_or_path: Qwen/Qwen2.5-7B
template: qwen

### Method
stage: sft
do_train: true
finetuning_type: lora

### Dataset
dataset: alpaca_en_demo
cutoff_len: 2048

### Output
output_dir: ./output
overwrite_output_dir: true

### Training
num_train_epochs: 1.0
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 5e-5
lr_scheduler_type: cosine
warmup_ratio: 0.1

### LoRA
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

### Optimization
bf16: true
gradient_checkpointing: true

### Logging
logging_steps: 10
save_steps: 100
```

### QLoRA Config (Large Models)

```yaml
# qlora_config.yaml
model_name_or_path: meta-llama/Llama-3.1-70B
template: llama3

stage: sft
do_train: true
finetuning_type: lora

# Enable 4-bit quantization
quantization_bit: 4
quantization_method: bitsandbytes

dataset: alpaca_en_demo
cutoff_len: 2048

output_dir: ./llama70b-qlora

# Conservative settings for large model
num_train_epochs: 1.0
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2e-5

# Smaller LoRA for memory
lora_rank: 8
lora_alpha: 16
lora_target: q_proj,v_proj

bf16: true
gradient_checkpointing: true

logging_steps: 5
save_steps: 50
```

### DPO Config

```yaml
# dpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
template: qwen

stage: dpo
do_train: true
finetuning_type: lora

dataset: dpo_demo
cutoff_len: 2048

output_dir: ./dpo-output

# DPO specific
dpo_beta: 0.1
dpo_loss: sigmoid

num_train_epochs: 1.0
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5e-6

lora_rank: 16
lora_alpha: 32

bf16: true
gradient_checkpointing: true
```

## Model Templates

LLaMA Factory uses templates for proper chat formatting:

| Model | Template |
|-------|----------|
| Llama 3/3.1/3.3 | `llama3` |
| Qwen 2/2.5 | `qwen` |
| Mistral | `mistral` |
| Mixtral | `mixtral` |
| Yi | `yi` |
| Gemma | `gemma` |
| DeepSeek | `deepseek` |
| Phi | `phi` |
| ChatGLM | `chatglm` |

## Built-in Datasets

### Demo Datasets

```yaml
# Small demo datasets for testing
dataset: alpaca_en_demo     # ~50 examples, English instruction
dataset: alpaca_zh_demo     # ~50 examples, Chinese instruction
dataset: dpo_demo           # DPO preference data
```

### Custom Datasets

Create `data/dataset_info.json`:

```json
{
  "my_dataset": {
    "file_name": "my_data.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "response": "output"
    }
  }
}
```

Data format (`my_data.json`):

```json
[
  {
    "instruction": "What is machine learning?",
    "output": "Machine learning is a subset of AI..."
  }
]
```

### Hub Datasets

```yaml
# Use HuggingFace datasets
dataset: trl-lib/Capybara
dataset_dir: ""  # Empty for Hub datasets
```

## CLI Commands

### Training

```bash
# Train with config
llamafactory-cli train config.yaml

# Override parameters
llamafactory-cli train config.yaml \
    --num_train_epochs 3 \
    --learning_rate 1e-5
```

### Inference/Chat

```bash
# Interactive chat
llamafactory-cli chat \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --adapter_name_or_path ./output \
    --template qwen

# With config file
llamafactory-cli chat inference_config.yaml
```

### Export (Merge LoRA)

```bash
# Merge LoRA into base model
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --adapter_name_or_path ./output \
    --template qwen \
    --export_dir ./merged
```

### Evaluation

```bash
# Evaluate on benchmark
llamafactory-cli eval \
    --model_name_or_path ./merged \
    --template qwen \
    --task mmlu
```

## Memory Optimization

### For Large Models

```yaml
# Reduce memory usage
quantization_bit: 4            # 4-bit quantization
per_device_train_batch_size: 1 # Minimum batch
gradient_accumulation_steps: 16 # Compensate
gradient_checkpointing: true   # Trade compute for memory
lora_rank: 8                   # Smaller LoRA
lora_target: q_proj,v_proj     # Fewer modules
```

### Clear Memory Before Training

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

## Multi-GPU Training

LLaMA Factory uses Accelerate for distributed training:

```bash
# Auto-detect GPUs
accelerate launch --multi_gpu \
    --num_processes 2 \
    llamafactory-cli train config.yaml
```

For DGX Spark (single GPU), standard `llamafactory-cli train` is sufficient.

## Checkpointing and Resume

### Save Checkpoints

```yaml
save_steps: 100
save_total_limit: 3
```

### Resume Training

```bash
llamafactory-cli train config.yaml \
    --resume_from_checkpoint ./output/checkpoint-100
```

## Export Formats

### Merged HuggingFace Model

```yaml
# export_config.yaml
model_name_or_path: Qwen/Qwen2.5-7B
adapter_name_or_path: ./output
template: qwen
finetuning_type: lora
export_dir: ./merged
export_size: 2  # Shard size in GB
export_device: auto
```

```bash
llamafactory-cli export export_config.yaml
```

### Push to Hub

```bash
# After export
huggingface-cli upload username/my-model ./merged
```

## Troubleshooting

### CUDA Out of Memory

1. Enable quantization: `quantization_bit: 4`
2. Reduce batch size to 1
3. Increase gradient accumulation
4. Reduce sequence length: `cutoff_len: 1024`
5. Use smaller LoRA: `lora_rank: 4`

### Template Mismatch

Ensure template matches model:
```bash
# Check model's chat template
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('model'); print(t.chat_template)"
```

### Dataset Errors

Verify dataset format matches expected columns:
```python
from datasets import load_dataset
ds = load_dataset("your/dataset")
print(ds.column_names)
```

### Slow Training

1. Ensure using NVIDIA container
2. Enable bf16: `bf16: true`
3. Enable gradient checkpointing
4. Use flash attention if available

## Resources

- [LLaMA Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [LLaMA Factory Documentation](https://github.com/hiyouga/LLaMA-Factory/wiki)
- [NVIDIA LLaMA Factory Playbook](https://build.nvidia.com/spark/llama-factory)
- [Example Configs](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples)
