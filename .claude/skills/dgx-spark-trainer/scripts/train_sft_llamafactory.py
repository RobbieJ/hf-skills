#!/usr/bin/env python3
"""
LLaMA Factory training wrapper for DGX Spark.

This script provides a Python interface to LLaMA Factory's CLI-based training,
making it easier to integrate with automation and customize configurations.

LLaMA Factory is typically used via YAML configs and CLI commands:
    llamafactory-cli train config.yaml

This script demonstrates:
- Programmatic configuration generation
- CLI execution from Python
- Custom dataset integration
- Export and merge workflows

Usage (inside Docker container):
    # Using YAML config
    llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

    # Using this Python wrapper
    python train_sft_llamafactory.py

Docker setup:
    docker run --gpus all --ipc=host --ulimit memlock=-1 -it \
        --ulimit stack=67108864 \
        -v "$PWD":/workspace \
        -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
        -e HF_TOKEN="$HF_TOKEN" \
        -w /workspace \
        nvcr.io/nvidia/pytorch:25.09-py3 \
        bash -c 'git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git && \
                 cd LLaMA-Factory && pip install -e ".[metrics]" && \
                 cd /workspace && python train_sft_llamafactory.py'
"""

import os
import sys
import yaml
import subprocess
import tempfile
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Model configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B")
MODEL_TEMPLATE = os.environ.get("MODEL_TEMPLATE", "qwen")  # qwen, llama3, mistral, etc.

# Dataset configuration
DATASET_NAME = os.environ.get("DATASET_NAME", "alpaca_en_demo")  # Built-in or custom
DATASET_DIR = os.environ.get("DATASET_DIR", "")  # Custom dataset directory

# Training configuration
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./llamafactory-output")
NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", "1.0"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRADIENT_ACCUMULATION = int(os.environ.get("GRADIENT_ACCUMULATION", "4"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-5"))
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))

# LoRA configuration
LORA_RANK = int(os.environ.get("LORA_RANK", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))

# Quantization
QUANTIZATION_BIT = int(os.environ.get("QUANTIZATION_BIT", "4"))  # 4 or 8, 0 for none


def generate_training_config() -> dict:
    """Generate LLaMA Factory training configuration."""
    config = {
        # Model
        "model_name_or_path": MODEL_NAME,
        "template": MODEL_TEMPLATE,

        # Method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",

        # Dataset
        "dataset": DATASET_NAME,
        "cutoff_len": MAX_SEQ_LENGTH,
        "preprocessing_num_workers": 4,

        # Output
        "output_dir": OUTPUT_DIR,
        "overwrite_output_dir": True,

        # Training
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,

        # LoRA
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target": "all",  # Target all linear layers

        # Optimization
        "bf16": True,
        "gradient_checkpointing": True,
        "optim": "adamw_torch",

        # Logging
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 3,

        # Evaluation
        "eval_strategy": "steps",
        "eval_steps": 100,
        "per_device_eval_batch_size": BATCH_SIZE,

        # Plotting
        "plot_loss": True,
    }

    # Add quantization if specified
    if QUANTIZATION_BIT > 0:
        config["quantization_bit"] = QUANTIZATION_BIT
        config["quantization_method"] = "bitsandbytes"

    # Add custom dataset directory if specified
    if DATASET_DIR:
        config["dataset_dir"] = DATASET_DIR

    return config


def generate_export_config(adapter_path: str, output_path: str) -> dict:
    """Generate configuration for merging LoRA adapters."""
    return {
        "model_name_or_path": MODEL_NAME,
        "adapter_name_or_path": adapter_path,
        "template": MODEL_TEMPLATE,
        "finetuning_type": "lora",
        "export_dir": output_path,
        "export_size": 2,
        "export_device": "auto",
        "export_legacy_format": False,
    }


def run_llamafactory_cli(command: str, config: dict) -> int:
    """Run LLaMA Factory CLI with the given configuration."""
    # Write config to temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        config_path = f.name

    try:
        # Build command
        cmd = f"llamafactory-cli {command} {config_path}"
        print(f"Running: {cmd}")
        print(f"Config file: {config_path}")
        print()

        # Execute
        result = subprocess.run(cmd, shell=True, check=False)
        return result.returncode
    finally:
        # Cleanup
        os.unlink(config_path)


def check_llamafactory_installed() -> bool:
    """Check if LLaMA Factory is installed."""
    try:
        result = subprocess.run(
            "llamafactory-cli version",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"LLaMA Factory version: {result.stdout.strip()}")
            return True
    except Exception:
        pass
    return False


def main():
    print("=" * 60)
    print("LLaMA Factory SFT Training on DGX Spark")
    print("=" * 60)
    print()

    # Check installation
    if not check_llamafactory_installed():
        print("LLaMA Factory not found!")
        print()
        print("Install with:")
        print("  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git")
        print("  cd LLaMA-Factory && pip install -e '.[metrics]'")
        sys.exit(1)
    print()

    # Print configuration
    print("Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Template: {MODEL_TEMPLATE}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  LoRA rank: {LORA_RANK}")
    if QUANTIZATION_BIT > 0:
        print(f"  Quantization: {QUANTIZATION_BIT}-bit")
    print()

    # Generate and run training
    print("Starting training...")
    print("-" * 60)

    config = generate_training_config()

    # Print config for debugging
    print("Generated config:")
    print(yaml.dump(config, default_flow_style=False))
    print("-" * 60)

    result = run_llamafactory_cli("train", config)

    if result != 0:
        print(f"Training failed with exit code {result}")
        sys.exit(result)

    print()
    print("=" * 60)
    print("Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print()
    print("Next steps:")
    print(f"  1. Test: llamafactory-cli chat --model_name_or_path {MODEL_NAME} --adapter_name_or_path {OUTPUT_DIR} --template {MODEL_TEMPLATE}")
    print(f"  2. Export: python train_sft_llamafactory.py --export")
    print("=" * 60)


def export_model():
    """Export (merge) LoRA adapters into base model."""
    print("=" * 60)
    print("Exporting (merging) LoRA adapters")
    print("=" * 60)
    print()

    adapter_path = OUTPUT_DIR
    export_path = f"{OUTPUT_DIR}-merged"

    print(f"Adapter path: {adapter_path}")
    print(f"Export path: {export_path}")
    print()

    config = generate_export_config(adapter_path, export_path)
    result = run_llamafactory_cli("export", config)

    if result != 0:
        print(f"Export failed with exit code {result}")
        sys.exit(result)

    print()
    print(f"Merged model saved to: {export_path}")


if __name__ == "__main__":
    if "--export" in sys.argv:
        export_model()
    else:
        main()
