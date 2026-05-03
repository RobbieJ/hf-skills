# Native Hugging Face TRL Training Guide for DGX Spark

Comprehensive guide to using native Hugging Face libraries (Transformers, TRL, PEFT) for LLM training on DGX Spark.

## Overview

Native Hugging Face training uses the foundational libraries that power the ML ecosystem:

- **Transformers** - Model loading, tokenization, architectures
- **TRL** - Training recipes (SFT, DPO, GRPO, PPO, Reward Modeling)
- **PEFT** - Parameter-efficient fine-tuning (LoRA, QLoRA)
- **BitsAndBytes** - Quantization (4-bit, 8-bit)
- **Accelerate** - Distributed training, mixed precision

## When to Use Native HF vs Wrappers

| Use Native HF When | Use Unsloth/LLaMA Factory When |
|-------------------|-------------------------------|
| Need full control over training | Want fastest training (Unsloth) |
| Using custom architectures | Prefer YAML configs (LLaMA Factory) |
| Debugging training issues | Quick experimentation |
| Custom data collators | Standard training patterns |
| Advanced RL methods (PPO, GRPO) | SFT/DPO with sensible defaults |
| Research and experimentation | Production fine-tuning |

## Installation

### Inside NVIDIA Container

```bash
# Start container
docker run --gpus all --ipc=host --ulimit memlock=-1 -it \
    --ulimit stack=67108864 --rm \
    -v "$PWD":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    nvcr.io/nvidia/pytorch:25.09-py3 bash

# Install HF libraries
pip install transformers datasets accelerate
pip install trl peft bitsandbytes
pip install hf_transfer  # Fast downloads
```

### Version Requirements

```
transformers>=4.40.0
trl>=0.8.0
peft>=0.10.0
bitsandbytes>=0.43.0
accelerate>=0.27.0
```

## Basic SFT Training Pattern

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer.pad_token = tokenizer.eos_token

# 2. Configure quantization (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 3. Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

# 4. Prepare for k-bit training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# 5. Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)

# 6. Load dataset
dataset = load_dataset("trl-lib/Capybara", split="train")

# 7. Configure training
training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=100,
    max_seq_length=2048,
)

# 8. Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)
trainer.train()

# 9. Save
trainer.save_model("./output")
```

## Quantization Options

### 4-bit Quantization (QLoRA)

Most memory efficient, recommended for models >7B:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # or "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Nested quantization
)
```

### 8-bit Quantization

Less aggressive, slightly better quality:

```python
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)
```

### No Quantization (Full Precision)

For small models (<3B) or when you have memory:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

## LoRA Configuration

### Standard LoRA

```python
lora_config = LoraConfig(
    r=16,              # Rank (lower = less memory)
    lora_alpha=32,     # Scaling factor (typically 2x rank)
    lora_dropout=0.05, # Dropout for regularization
    bias="none",       # Don't train biases
    task_type="CAUSAL_LM",
    target_modules=[   # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
```

### Target Modules by Model

| Model | Attention Modules | MLP Modules |
|-------|------------------|-------------|
| Llama/Qwen/Mistral | q_proj, k_proj, v_proj, o_proj | gate_proj, up_proj, down_proj |
| GPT-2/GPT-J | q_proj, v_proj | mlp.fc_in, mlp.fc_out |
| Falcon | query_key_value | dense_h_to_4h, dense_4h_to_h |

### Automatic Target Detection

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",  # Auto-detect all linear layers
)
```

## Training Methods

### Supervised Fine-Tuning (SFT)

Standard instruction tuning:

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./sft-output",
        max_seq_length=2048,
        dataset_text_field="messages",  # For chat datasets
    ),
)
```

### Direct Preference Optimization (DPO)

Alignment from preference data:

```python
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model=model,
    args=DPOConfig(
        output_dir="./dpo-output",
        beta=0.1,  # KL penalty coefficient
        loss_type="sigmoid",  # or "hinge", "ipo"
    ),
    train_dataset=preference_dataset,  # Must have chosen/rejected
    processing_class=tokenizer,
)
```

### ORPO (Odds Ratio Preference Optimization)

Combines SFT and preference optimization:

```python
from trl import ORPOTrainer, ORPOConfig

trainer = ORPOTrainer(
    model=model,
    args=ORPOConfig(
        output_dir="./orpo-output",
        beta=0.1,
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
)
```

### Reward Modeling

Train a reward model for RLHF:

```python
from trl import RewardTrainer, RewardConfig

trainer = RewardTrainer(
    model=reward_model,
    args=RewardConfig(
        output_dir="./reward-output",
        max_length=512,
    ),
    train_dataset=comparison_dataset,
    processing_class=tokenizer,
)
```

## Dataset Formats

### Chat/Messages Format (SFT)

```python
# For datasets with "messages" field
{
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
}

# Configuration
SFTConfig(dataset_text_field="messages")
```

### Text Completion Format (SFT)

```python
# For datasets with plain text
{"text": "Question: What is AI?\nAnswer: AI is..."}

# Configuration
SFTConfig(dataset_text_field="text")
```

### Preference Format (DPO)

```python
# Required columns: prompt, chosen, rejected
{
    "prompt": "What is the capital of France?",
    "chosen": "The capital of France is Paris.",
    "rejected": "France is a country in Europe."
}
```

## Memory Optimization

### Gradient Checkpointing

```python
# In model loading
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

# In training config
SFTConfig(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```

### 8-bit Optimizer

```python
SFTConfig(optim="adamw_8bit")
```

### Paged Optimizer (for very large models)

```python
SFTConfig(optim="paged_adamw_8bit")
```

### Reduce Memory Fragmentation

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
```

## Attention Implementations

### Scaled Dot Product Attention (SDPA)

Default, good balance of speed and compatibility:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="sdpa",
)
```

### Flash Attention 2

Fastest, requires flash-attn package:

```python
pip install flash-attn --no-build-isolation

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
)
```

## Callbacks and Monitoring

### TensorBoard

```python
SFTConfig(report_to="tensorboard")

# View logs
tensorboard --logdir ./output/runs
```

### Weights & Biases

```python
import wandb
wandb.login()

SFTConfig(
    report_to="wandb",
    run_name="my-training-run",
)
```

### Custom Callbacks

```python
from transformers import TrainerCallback

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(f"Step {state.global_step}: {logs}")

trainer = SFTTrainer(
    ...,
    callbacks=[LoggingCallback()],
)
```

## Saving and Loading

### Save LoRA Adapter

```python
# Save adapter only
model.save_pretrained("./lora-adapter")

# Load adapter
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("base-model")
model = PeftModel.from_pretrained(base_model, "./lora-adapter")
```

### Merge and Save

```python
# Merge LoRA into base model
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
```

### Auto-Load PEFT Model

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "./lora-adapter",
    device_map="auto",
)
```

### Push to Hub

```python
# Push adapter
model.push_to_hub("username/my-lora-adapter")

# Push merged model
merged_model.push_to_hub("username/my-merged-model")
```

## Advanced Configurations

### Custom Data Collator

```python
from trl import DataCollatorForCompletionOnlyLM

# Only compute loss on assistant responses
collator = DataCollatorForCompletionOnlyLM(
    response_template="<|assistant|>",
    tokenizer=tokenizer,
)

trainer = SFTTrainer(
    ...,
    data_collator=collator,
)
```

### Packing (Multiple Examples per Sequence)

```python
SFTConfig(
    packing=True,
    max_seq_length=4096,
)
```

### NEFTune (Noise Embeddings)

```python
SFTConfig(neftune_noise_alpha=5)
```

## Troubleshooting

### Out of Memory

1. Enable 4-bit quantization
2. Reduce batch size to 1
3. Increase gradient accumulation
4. Enable gradient checkpointing
5. Reduce sequence length
6. Use 8-bit optimizer

### NaN Loss

```python
SFTConfig(
    max_grad_norm=1.0,
    learning_rate=1e-5,  # Lower
)
```

### Slow Training

1. Use SDPA or Flash Attention
2. Enable bf16
3. Use packing for short sequences
4. Increase batch size if memory allows

## Resources

- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [TRL Example Scripts](https://github.com/huggingface/trl/tree/main/examples)
