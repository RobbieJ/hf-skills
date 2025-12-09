#!/bin/bash
# DGX Spark Skill Test Runner
#
# This script sets up the Docker environment and runs all tests.
# Run this from the DGX Spark after cloning the repository.
#
# Usage:
#   ./run_tests.sh           # Run quick tests
#   ./run_tests.sh --full    # Run full tests with 7B model
#   ./run_tests.sh --help    # Show help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_help() {
    echo "DGX Spark Skill Test Runner"
    echo ""
    echo "Usage: ./run_tests.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --quick       Run quick smoke tests (default)"
    echo "  --full        Run full tests with 7B model"
    echo "  --test NAME   Run specific test (hardware, imports, native-hf, unsloth, llamafactory)"
    echo "  --no-docker   Run tests directly without Docker (assumes deps installed)"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_tests.sh                    # Quick smoke test"
    echo "  ./run_tests.sh --full             # Full test with 7B model"
    echo "  ./run_tests.sh --test unsloth     # Test Unsloth only"
}

# Parse arguments
TEST_MODE="--quick"
SPECIFIC_TEST=""
USE_DOCKER=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            TEST_MODE="--quick"
            shift
            ;;
        --full)
            TEST_MODE="--full"
            shift
            ;;
        --test)
            SPECIFIC_TEST="--test $2"
            shift 2
            ;;
        --no-docker)
            USE_DOCKER=false
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DGX Spark Skill Test Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. Are you on the DGX Spark?${NC}"
    exit 1
fi

echo -e "${GREEN}GPU detected:${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

if $USE_DOCKER; then
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker not found. Please install Docker.${NC}"
        exit 1
    fi

    # Check Docker GPU access
    echo -e "${YELLOW}Checking Docker GPU access...${NC}"
    if ! docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: Docker GPU access failed.${NC}"
        echo "Try: sudo apt-get install nvidia-container-toolkit && sudo systemctl restart docker"
        exit 1
    fi
    echo -e "${GREEN}Docker GPU access OK${NC}"
    echo ""

    # Pull container if needed
    CONTAINER="nvcr.io/nvidia/pytorch:25.09-py3"
    echo -e "${YELLOW}Ensuring container is available: ${CONTAINER}${NC}"
    docker pull "$CONTAINER" 2>/dev/null || true
    echo ""

    # Run tests in Docker
    echo -e "${BLUE}Running tests in Docker container...${NC}"
    echo ""

    docker run --gpus all --ipc=host --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "$SKILL_DIR":/workspace/skill \
        -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -w /workspace/skill \
        --rm "$CONTAINER" \
        bash -c "
            echo 'Installing dependencies...'
            pip install -q transformers peft datasets trl bitsandbytes accelerate hf_transfer
            pip install -q --no-deps unsloth unsloth_zoo 2>/dev/null || true
            echo ''
            python scripts/test_skill.py $TEST_MODE $SPECIFIC_TEST
        "
else
    # Run tests directly
    echo -e "${BLUE}Running tests directly (no Docker)...${NC}"
    echo ""
    cd "$SKILL_DIR"
    python scripts/test_skill.py $TEST_MODE $SPECIFIC_TEST
fi

echo ""
echo -e "${GREEN}Test run complete!${NC}"
