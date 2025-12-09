#!/usr/bin/env python3
"""
Production-ready SFT training example using Unsloth on DGX Spark.

This script demonstrates:
- Unsloth's 2x faster training with 70% less memory
- QLoRA 4-bit quantization for large models
- Proper configuration for DGX Spark's unified memory
- Hugging Face Hub integration
- Checkpoint management
- Train/eval split for monitoring

Usage (inside Docker container):
    python train_sft_unsloth.py

Or customize via environment variables:
    MODEL_NAME="unsloth/Llama-3.1-8B-bnb-4bit" \
    DATASET_NAME="trl-lib/Capybara" \
    OUTPUT_DIR="./my-model" \
    HUB_MODEL_ID="username/my-model" \
    python train_sft_unsloth.py

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
                 python train_sft_unsloth.py'
"""

import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# =============================================================================
# Configuration (customize via environment variables or edit directly)
# =============================================================================

# Model configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Qwen2.5-7B-bnb-4bit")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))

# Dataset configuration
DATASET_NAME = os.environ.get("DATASET_NAME", "trl-lib/Capybara")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "train")
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))  # 0 = use all

# Training configuration
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./qwen-sft-unsloth")
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "")  # Set to push to Hub
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRADIENT_ACCUMULATION = int(os.environ.get("GRADIENT_ACCUMULATION", "4"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))

# LoRA configuration
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))

# =============================================================================
# Main Training Script
# =============================================================================

def main():
    print("=" * 60)
    print("Unsloth SFT Training on DGX Spark")
    print("=" * 60)
    print()

    # Check GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load model with 4-bit quantization
    print(f"Loading model: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    print("Model loaded successfully!")
    print()

    # Add LoRA adapters
    print(f"Adding LoRA adapters (r={LORA_R}, alpha={LORA_ALPHA})")
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
        use_gradient_checkpointing="unsloth",  # Optimized for Unsloth
        random_state=42,
    )
    print("LoRA adapters added!")
    print()

    # Load dataset
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    if MAX_SAMPLES > 0 and len(dataset) > MAX_SAMPLES:
        dataset = dataset.select(range(MAX_SAMPLES))
        print(f"Limited to {MAX_SAMPLES} samples")

    print(f"Dataset size: {len(dataset)} examples")

    # Create train/eval split
    print("Creating train/eval split (90/10)...")
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Eval: {len(eval_dataset)} examples")
    print()

    # Training configuration
    effective_batch = BATCH_SIZE * GRADIENT_ACCUMULATION
    print(f"Training configuration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Output: {OUTPUT_DIR}")
    if HUB_MODEL_ID:
        print(f"  Hub: {HUB_MODEL_ID}")
    print()

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        # Training parameters
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,

        # Optimization
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",  # Memory efficient optimizer
        weight_decay=0.01,

        # Memory optimization
        bf16=True,  # Use bfloat16 on Blackwell
        gradient_checkpointing=True,

        # Logging
        logging_steps=10,
        logging_first_step=True,

        # Evaluation
        eval_strategy="steps",
        eval_steps=100,

        # Checkpointing
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,

        # Hub settings (if HUB_MODEL_ID is set)
        push_to_hub=bool(HUB_MODEL_ID),
        hub_model_id=HUB_MODEL_ID if HUB_MODEL_ID else None,
        hub_strategy="checkpoint" if HUB_MODEL_ID else None,

        # Reproducibility
        seed=42,
        data_seed=42,
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # Show trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print()

    # Start training
    print("Starting training...")
    print("=" * 60)
    trainer.train()
    print("=" * 60)
    print("Training complete!")
    print()

    # Save model
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Push to Hub if configured
    if HUB_MODEL_ID:
        print(f"Pushing to Hub: {HUB_MODEL_ID}...")
        model.push_to_hub(HUB_MODEL_ID)
        tokenizer.push_to_hub(HUB_MODEL_ID)
        print(f"Model available at: https://huggingface.co/{HUB_MODEL_ID}")

    print()
    print("=" * 60)
    print("Done!")
    print(f"Model saved to: {OUTPUT_DIR}")
    if HUB_MODEL_ID:
        print(f"Hub URL: https://huggingface.co/{HUB_MODEL_ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()
