#!/usr/bin/env python3
"""
QLoRA Training Example for Large Models on DGX Spark.

This script demonstrates 4-bit QLoRA training for large models (13B-70B)
on DGX Spark's 128GB unified memory.

QLoRA enables training large models by:
- 4-bit quantization of base model weights
- LoRA adapters in full precision (fp16/bf16)
- Significant memory reduction (~4x)

Memory requirements (approximate):
- 13B model: ~20GB
- 20B model: ~35GB
- 70B model: ~68GB
- 120B model: ~68GB (with Unsloth optimizations)

Usage:
    # Default 70B model
    python train_qlora_example.py

    # Custom model
    MODEL_NAME="unsloth/Llama-3.1-8B-bnb-4bit" python train_qlora_example.py

Docker command:
    docker run --gpus all --ipc=host --ulimit memlock=-1 -it \
        --ulimit stack=67108864 \
        -v "$PWD":/workspace \
        -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
        -e HF_TOKEN="$HF_TOKEN" \
        -w /workspace \
        nvcr.io/nvidia/pytorch:25.09-py3 \
        bash -c 'pip install -q transformers peft datasets trl && \
                 pip install -q --no-deps unsloth unsloth_zoo bitsandbytes && \
                 python train_qlora_example.py'
"""

import os
import gc
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# =============================================================================
# Configuration
# =============================================================================

# Model - Default to large model to demonstrate QLoRA benefits
MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Llama-3.3-70B-Instruct-bnb-4bit")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))

# Dataset
DATASET_NAME = os.environ.get("DATASET_NAME", "trl-lib/Capybara")
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "500"))  # Fewer samples for large models

# Training - Conservative settings for large models
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./llama70b-qlora")
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))  # Small for large models
GRADIENT_ACCUMULATION = int(os.environ.get("GRADIENT_ACCUMULATION", "16"))  # Compensate
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-5"))  # Lower for large models

# LoRA - Conservative for large models
LORA_R = int(os.environ.get("LORA_R", "8"))  # Lower rank for memory
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))

# =============================================================================
# Memory Management
# =============================================================================

def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_memory_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("QLoRA Training for Large Models on DGX Spark")
    print("=" * 60)
    print()

    # System info
    print("System Information:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Total Memory: {total_mem:.1f} GB")
    print()

    # Configuration
    print("Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Sequence length: {MAX_SEQ_LENGTH}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Max samples: {MAX_SAMPLES}")
    print(f"  Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print()

    # Clear memory before loading
    print("Clearing memory...")
    clear_memory()
    print_memory_usage()
    print()

    # Load model with 4-bit quantization
    print(f"Loading model: {MODEL_NAME}")
    print("This may take several minutes for large models...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect (bf16 on Blackwell)
        load_in_4bit=True,
    )
    print("Model loaded!")
    print_memory_usage()
    print()

    # Add LoRA adapters with conservative settings
    print(f"Adding LoRA adapters (r={LORA_R}, alpha={LORA_ALPHA})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Critical for large models
        random_state=42,
    )
    print("LoRA adapters added!")
    print_memory_usage()
    print()

    # Load dataset
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")

    if MAX_SAMPLES > 0 and len(dataset) > MAX_SAMPLES:
        dataset = dataset.select(range(MAX_SAMPLES))

    print(f"Dataset size: {len(dataset)} examples")
    print()

    # Training configuration - optimized for large models
    print("Preparing trainer...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        # Training - conservative for large models
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,

        # Optimization - critical for large models
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",  # 8-bit optimizer for memory savings

        # Scheduler
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",

        # Logging
        logging_steps=5,  # More frequent for long training
        logging_first_step=True,

        # Save less frequently for large models
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,  # Keep fewer checkpoints

        # No eval during training for memory
        eval_strategy="no",

        # Reproducibility
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    print_memory_usage()
    print()

    # Training
    print("Starting training...")
    print("=" * 60)
    trainer.train()
    print("=" * 60)
    print()

    # Save
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print()
    print("=" * 60)
    print("QLoRA Training Complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print()
    print("To use the model:")
    print(f"  from unsloth import FastLanguageModel")
    print(f"  model, tokenizer = FastLanguageModel.from_pretrained('{OUTPUT_DIR}')")
    print()
    print("To push to Hub:")
    print(f"  model.push_to_hub('username/model-name')")
    print(f"  tokenizer.push_to_hub('username/model-name')")
    print("=" * 60)


if __name__ == "__main__":
    main()
