#!/usr/bin/env python3
"""
Native Hugging Face SFT Training on DGX Spark.

This script demonstrates training using native Hugging Face libraries:
- Transformers for model loading
- PEFT for LoRA adapters
- TRL for SFT training
- BitsAndBytes for quantization

This is the foundational approach that Unsloth and LLaMA Factory build upon.
Use this when you want full control or need features not exposed by wrappers.

Usage (inside Docker container):
    python train_sft_huggingface.py

Or customize via environment variables:
    MODEL_NAME="Qwen/Qwen2.5-7B" \
    DATASET_NAME="trl-lib/Capybara" \
    OUTPUT_DIR="./my-model" \
    python train_sft_huggingface.py

Docker command:
    docker run --gpus all --ipc=host --ulimit memlock=-1 -it \
        --ulimit stack=67108864 \
        -v "$PWD":/workspace \
        -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
        -e HF_TOKEN="$HF_TOKEN" \
        -w /workspace \
        nvcr.io/nvidia/pytorch:25.09-py3 \
        bash -c 'pip install -q transformers peft datasets trl bitsandbytes accelerate && \
                 python train_sft_huggingface.py'
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# =============================================================================
# Configuration
# =============================================================================

# Model configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
USE_4BIT = os.environ.get("USE_4BIT", "true").lower() == "true"

# Dataset configuration
DATASET_NAME = os.environ.get("DATASET_NAME", "trl-lib/Capybara")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "train")
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))  # 0 = use all

# Training configuration
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./qwen-sft-hf")
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "")
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRADIENT_ACCUMULATION = int(os.environ.get("GRADIENT_ACCUMULATION", "4"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))

# LoRA configuration
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
# Target modules - comma-separated list, or "default" for standard attention+mlp
LORA_TARGET_MODULES = os.environ.get("LORA_TARGET_MODULES", "default")

# =============================================================================
# Main Training Script
# =============================================================================

def main():
    print("=" * 60)
    print("Native Hugging Face SFT Training on DGX Spark")
    print("=" * 60)
    print()

    # System info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Configuration summary
    print("Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  4-bit quantization: {USE_4BIT}")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print()

    # =========================================================================
    # Load Tokenizer
    # =========================================================================
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print()

    # =========================================================================
    # Configure Quantization (QLoRA)
    # =========================================================================
    if USE_4BIT:
        print("Configuring 4-bit quantization (QLoRA)...")
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
    print("This may take several minutes for large models...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not USE_4BIT else None,
        attn_implementation="sdpa",  # Use scaled dot product attention
    )

    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Model dtype: {model.dtype}")
    print()

    # =========================================================================
    # Configure LoRA
    # =========================================================================
    print(f"Configuring LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")

    # Prepare model for k-bit training if quantized
    if USE_4BIT:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    # Parse target modules
    if LORA_TARGET_MODULES == "default":
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    else:
        target_modules = [m.strip() for m in LORA_TARGET_MODULES.split(",")]

    # LoRA configuration
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    print(f"  Target modules: {target_modules}")

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print()

    # =========================================================================
    # Load Dataset
    # =========================================================================
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    if MAX_SAMPLES > 0 and len(dataset) > MAX_SAMPLES:
        dataset = dataset.select(range(MAX_SAMPLES))

    print(f"Dataset size: {len(dataset)} examples")
    print(f"Dataset columns: {dataset.column_names}")

    # Validate dataset has expected format for chat/instruction training
    has_messages = "messages" in dataset.column_names
    has_text = "text" in dataset.column_names
    has_instruction = "instruction" in dataset.column_names

    if not (has_messages or has_text or has_instruction):
        print()
        print("WARNING: Dataset may not have standard format.")
        print("  Expected one of: 'messages', 'text', or 'instruction' column")
        print(f"  Found columns: {dataset.column_names}")
        print("  Training may fail or produce unexpected results.")

    # Create train/eval split
    print("Creating train/eval split (90/10)...")
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Eval: {len(eval_dataset)} examples")
    print()

    # =========================================================================
    # Training Configuration
    # =========================================================================
    print("Configuring training...")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        # Training parameters
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,

        # Optimization
        optim="adamw_torch",  # or "adamw_8bit" for memory savings
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Mixed precision
        bf16=True,
        fp16=False,

        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Sequence length
        max_seq_length=MAX_SEQ_LENGTH,

        # Logging
        logging_steps=10,
        logging_first_step=True,
        report_to="tensorboard",

        # Evaluation
        eval_strategy="steps",
        eval_steps=100,

        # Checkpointing
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,

        # Hub settings
        push_to_hub=bool(HUB_MODEL_ID),
        hub_model_id=HUB_MODEL_ID if HUB_MODEL_ID else None,

        # Reproducibility
        seed=42,
        data_seed=42,

        # Dataset
        dataset_text_field="messages",  # For chat datasets
        packing=False,  # Disable packing for simplicity
    )

    # =========================================================================
    # Initialize Trainer
    # =========================================================================
    print("Initializing trainer...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # =========================================================================
    # Train
    # =========================================================================
    print()
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    print("=" * 60)
    print("Training complete!")
    print()

    # =========================================================================
    # Save Model
    # =========================================================================
    print(f"Saving model to {OUTPUT_DIR}...")

    # Save the LoRA adapter
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Push to Hub if configured
    if HUB_MODEL_ID:
        print(f"Pushing to Hub: {HUB_MODEL_ID}...")
        trainer.push_to_hub()
        print(f"Model available at: https://huggingface.co/{HUB_MODEL_ID}")

    print()
    print("=" * 60)
    print("Done!")
    print(f"Model saved to: {OUTPUT_DIR}")
    if HUB_MODEL_ID:
        print(f"Hub URL: https://huggingface.co/{HUB_MODEL_ID}")
    print()
    print("To load and use the model:")
    print("  from peft import AutoPeftModelForCausalLM")
    print(f"  model = AutoPeftModelForCausalLM.from_pretrained('{OUTPUT_DIR}')")
    print("=" * 60)


if __name__ == "__main__":
    main()
