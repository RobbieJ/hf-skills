#!/bin/bash
# DGX Spark Environment Setup Script
#
# This script sets up the Docker environment for LLM training on DGX Spark.
# Run this on your DGX Spark system to prepare for training.
#
# Usage:
#   chmod +x setup_environment.sh
#   ./setup_environment.sh [--unsloth|--llamafactory|--both]
#
# Options:
#   --unsloth       Set up Unsloth environment (default)
#   --llamafactory  Set up LLaMA Factory environment
#   --both          Set up both environments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/dgx-spark-training}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check nvidia-smi
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
        exit 1
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi

    # Check Docker GPU support
    if ! docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null; then
        log_error "Docker GPU support not working. Please install nvidia-container-toolkit."
        log_info "Run: sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker"
        exit 1
    fi

    log_info "All prerequisites met!"
}

# Create workspace directory
setup_workspace() {
    log_info "Setting up workspace at $WORKSPACE_DIR..."
    mkdir -p "$WORKSPACE_DIR"
    mkdir -p "$HOME/.cache/huggingface"
    log_info "Workspace created."
}

# Pull NVIDIA PyTorch container
pull_container() {
    local CONTAINER="nvcr.io/nvidia/pytorch:25.09-py3"
    log_info "Pulling NVIDIA PyTorch container: $CONTAINER"
    docker pull "$CONTAINER"
    log_info "Container pulled successfully."
}

# Setup Unsloth environment
setup_unsloth() {
    log_info "Setting up Unsloth environment..."

    cat > "$WORKSPACE_DIR/start_unsloth.sh" << 'EOF'
#!/bin/bash
# Start Unsloth training environment

WORKSPACE="${WORKSPACE:-$(pwd)}"

docker run --gpus all --ipc=host --ulimit memlock=-1 -it \
    --ulimit stack=67108864 \
    -v "$WORKSPACE":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -w /workspace \
    --rm nvcr.io/nvidia/pytorch:25.09-py3 \
    bash -c '
        echo "Installing Unsloth dependencies..."
        pip install -q transformers peft "datasets>=2.14.0" "trl>=0.7.0"
        pip install -q --no-deps unsloth unsloth_zoo
        pip install -q hf_transfer
        pip install -q --no-deps bitsandbytes
        echo ""
        echo "Unsloth environment ready!"
        echo "Example: python train_sft_unsloth.py"
        echo ""
        exec bash
    '
EOF
    chmod +x "$WORKSPACE_DIR/start_unsloth.sh"

    # Create example training script
    cat > "$WORKSPACE_DIR/train_sft_unsloth.py" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Unsloth SFT Training Example for DGX Spark"""

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Configuration
MODEL_NAME = "unsloth/Qwen2.5-7B-bnb-4bit"  # 4-bit quantized
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "./qwen-finetuned"

print(f"Loading model: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)

print("Loading dataset...")
dataset = load_dataset("trl-lib/Capybara", split="train[:1000]")
print(f"Dataset size: {len(dataset)} examples")

print("Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
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

print(f"Saving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete!")
PYTHON_EOF

    log_info "Unsloth environment setup complete."
    log_info "Start with: cd $WORKSPACE_DIR && ./start_unsloth.sh"
}

# Setup LLaMA Factory environment
setup_llamafactory() {
    log_info "Setting up LLaMA Factory environment..."

    cat > "$WORKSPACE_DIR/start_llamafactory.sh" << 'EOF'
#!/bin/bash
# Start LLaMA Factory training environment

WORKSPACE="${WORKSPACE:-$(pwd)}"

docker run --gpus all --ipc=host --ulimit memlock=-1 -it \
    --ulimit stack=67108864 \
    -v "$WORKSPACE":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -w /workspace \
    --rm nvcr.io/nvidia/pytorch:25.09-py3 \
    bash -c '
        echo "Setting up LLaMA Factory..."
        if [ ! -d "LLaMA-Factory" ]; then
            git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
        fi
        cd LLaMA-Factory
        pip install -q -e ".[metrics]"
        echo ""
        echo "LLaMA Factory environment ready!"
        echo "Example: llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml"
        echo ""
        exec bash
    '
EOF
    chmod +x "$WORKSPACE_DIR/start_llamafactory.sh"

    log_info "LLaMA Factory environment setup complete."
    log_info "Start with: cd $WORKSPACE_DIR && ./start_llamafactory.sh"
}

# Main
main() {
    local setup_type="${1:---unsloth}"

    echo "=========================================="
    echo "DGX Spark Training Environment Setup"
    echo "=========================================="
    echo ""

    check_prerequisites
    setup_workspace
    pull_container

    case "$setup_type" in
        --unsloth)
            setup_unsloth
            ;;
        --llamafactory)
            setup_llamafactory
            ;;
        --both)
            setup_unsloth
            setup_llamafactory
            ;;
        *)
            log_error "Unknown option: $setup_type"
            echo "Usage: $0 [--unsloth|--llamafactory|--both]"
            exit 1
            ;;
    esac

    echo ""
    echo "=========================================="
    echo "Setup Complete!"
    echo "=========================================="
    echo ""
    log_info "Workspace: $WORKSPACE_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. cd $WORKSPACE_DIR"
    echo "  2. ./start_unsloth.sh  (or ./start_llamafactory.sh)"
    echo "  3. Run your training script"
    echo ""
}

main "$@"
