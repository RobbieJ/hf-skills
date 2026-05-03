#!/usr/bin/env python3
"""
Native Hugging Face DPO Training on DGX Spark.

This script demonstrates Direct Preference Optimization (DPO) training
using native Hugging Face libraries for alignment from preference data.

DPO enables:
- Training models to prefer chosen over rejected responses
- Alignment without a separate reward model
- Better response quality after SFT

Usage:
    python train_dpo_huggingface.py

Docker command:
    docker run --gpus all --ipc=host --ulimit memlock=-1 -it \
        --ulimit stack=67108864 \
        -v "$PWD":/workspace \
        -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
        -e HF_TOKEN="$HF_TOKEN" \
        -w /workspace \
        nvcr.io/nvidia/pytorch:25.09-py3 \
        bash -c 'pip install -q transformers peft datasets trl bitsandbytes accelerate && \
                 python train_dpo_huggingface.py'
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# =============================================================================
# Configuration
# =============================================================================

# Model - should be an instruction-tuned model
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "1024"))
USE_4BIT = os.environ.get("USE_4BIT", "true").lower() == "true"

# Dataset - must have prompt, chosen, rejected columns
DATASET_NAME = os.environ.get("DATASET_NAME", "trl-lib/ultrafeedback_binarized")
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "1000"))

# Training
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./qwen-dpo-hf")
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
GRADIENT_ACCUMULATION = int(os.environ.get("GRADIENT_ACCUMULATION", "8"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-6"))

# DPO specific
DPO_BETA = float(os.environ.get("DPO_BETA", "0.1"))

# LoRA
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))


def main():
    print("=" * 60)
    print("Native Hugging Face DPO Training on DGX Spark")
    print("=" * 60)
    print()

    # System info
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    print("Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  DPO beta: {DPO_BETA}")
    print(f"  Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION}")
    print()

    # =========================================================================
    # Load Tokenizer
    # =========================================================================
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print()

    # =========================================================================
    # Configure Quantization
    # =========================================================================
    if USE_4BIT:
        print("Configuring 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # =========================================================================
    # Load Model
    # =========================================================================
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not USE_4BIT else None,
        attn_implementation="sdpa",
    )

    # Prepare for training
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA configuration
    print(f"Configuring LoRA (r={LORA_R})...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print()

    # =========================================================================
    # Load Dataset
    # =========================================================================
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")

    if MAX_SAMPLES > 0:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

    print(f"Dataset size: {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")
    print()

    # =========================================================================
    # DPO Training Configuration
    # =========================================================================
    print("Configuring DPO training...")

    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,

        # DPO specific
        beta=DPO_BETA,
        loss_type="sigmoid",  # or "hinge", "ipo"

        # Training
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,

        # Optimization
        optim="adamw_torch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",

        # Mixed precision
        bf16=True,
        gradient_checkpointing=True,

        # Sequence length
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_SEQ_LENGTH // 2,

        # Logging
        logging_steps=10,
        report_to="tensorboard",

        # Checkpointing
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,

        # Reproducibility
        seed=42,
    )

    # =========================================================================
    # Initialize DPO Trainer
    # =========================================================================
    print("Initializing DPO trainer...")

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # =========================================================================
    # Train
    # =========================================================================
    print()
    print("Starting DPO training...")
    print("=" * 60)

    trainer.train()

    print("=" * 60)
    print("Training complete!")

    # Save
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print()
    print("=" * 60)
    print(f"DPO model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
