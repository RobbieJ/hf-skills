#!/usr/bin/env python3
"""
DGX Spark Hardware Verification Script

Verifies that the system is properly configured for LLM training on NVIDIA DGX Spark.
Run this before starting any training to catch configuration issues early.

Usage:
    python verify_dgx_spark.py

Or within Docker:
    docker run --gpus all -v "$PWD":/workspace nvcr.io/nvidia/pytorch:25.09-py3 \
        python /workspace/verify_dgx_spark.py
"""

import subprocess
import sys
import os
from typing import Tuple, Optional


def run_command(cmd: str) -> Tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def check_nvidia_smi() -> Tuple[bool, str]:
    """Check if nvidia-smi is available and GPU is detected."""
    success, output = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if success and output.strip():
        return True, f"GPU detected: {output.strip()}"
    return False, "nvidia-smi failed or no GPU detected"


def check_cuda_version() -> Tuple[bool, str]:
    """Check CUDA version."""
    success, output = run_command("nvcc --version")
    if success:
        # Extract version number
        for line in output.split('\n'):
            if 'release' in line.lower():
                return True, f"CUDA: {line.strip()}"
        return True, "CUDA installed (version unknown)"

    # Try alternative method
    success, output = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    if success:
        return True, f"NVIDIA Driver: {output.strip()} (nvcc not found, but driver present)"

    return False, "CUDA not found. Install CUDA toolkit or use NVIDIA Docker container."


def check_memory() -> Tuple[bool, str]:
    """Check available GPU memory."""
    success, output = run_command("nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits")
    if success:
        try:
            total, free = output.strip().split(',')
            total_gb = int(total.strip()) / 1024
            free_gb = int(free.strip()) / 1024

            if total_gb >= 100:  # DGX Spark has 128GB unified memory
                return True, f"Memory: {total_gb:.0f}GB total, {free_gb:.0f}GB free (DGX Spark detected)"
            else:
                return True, f"Memory: {total_gb:.0f}GB total, {free_gb:.0f}GB free (Note: DGX Spark has 128GB)"
        except:
            return True, f"Memory info: {output.strip()}"
    return False, "Could not query GPU memory"


def check_compute_capability() -> Tuple[bool, str]:
    """Check GPU compute capability."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(device)
            cap_str = f"{capability[0]}.{capability[1]}"

            if capability[0] >= 12:  # Blackwell is compute capability 12.x
                return True, f"Compute Capability: {cap_str} (Blackwell architecture)"
            elif capability[0] >= 9:
                return True, f"Compute Capability: {cap_str} (Hopper architecture)"
            elif capability[0] >= 8:
                return True, f"Compute Capability: {cap_str} (Ampere architecture)"
            else:
                return True, f"Compute Capability: {cap_str} (older architecture)"
    except ImportError:
        return False, "PyTorch not installed. Cannot check compute capability."
    except Exception as e:
        return False, f"Error checking compute capability: {e}"

    return False, "CUDA not available in PyTorch"


def check_docker() -> Tuple[bool, str]:
    """Check if Docker is available with GPU support."""
    success, output = run_command("docker --version")
    if not success:
        return False, "Docker not installed"

    # Check if nvidia-container-toolkit is configured (without running a container for speed)
    success, output = run_command("docker info 2>/dev/null | grep -i nvidia")
    if success and output.strip():
        return True, "Docker with NVIDIA runtime: OK"

    # Fall back to container test if needed
    success, output = run_command("docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu24.04 nvidia-smi 2>&1 | head -5")
    if success and "NVIDIA-SMI" in output:
        return True, "Docker with GPU support: OK"

    return True, "Docker installed, but GPU support not verified (may need nvidia-container-toolkit)"


def check_disk_space() -> Tuple[bool, str]:
    """Check available disk space."""
    success, output = run_command("df -h . | tail -1")
    if success:
        parts = output.split()
        if len(parts) >= 4:
            available = parts[3]
            return True, f"Available disk space: {available}"
    return False, "Could not determine disk space"


def check_huggingface_auth() -> Tuple[bool, str]:
    """Check Hugging Face authentication."""
    # Check environment variable
    if os.environ.get("HF_TOKEN"):
        return True, "HF_TOKEN environment variable set"

    # Check HF_HOME environment variable for custom cache location
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    # Check cached token
    token_path = os.path.join(hf_home, "token")
    if os.path.exists(token_path):
        return True, f"Hugging Face token cached ({token_path})"

    # Also check default location if HF_HOME is set to something else
    default_token_path = os.path.expanduser("~/.cache/huggingface/token")
    if default_token_path != token_path and os.path.exists(default_token_path):
        return True, f"Hugging Face token cached ({default_token_path})"

    # Check via CLI
    success, output = run_command("huggingface-cli whoami 2>&1")
    if success and "Not logged in" not in output and "error" not in output.lower():
        return True, f"Logged in as: {output.strip().split()[0] if output.strip() else 'unknown'}"

    return False, "Not logged in to Hugging Face. Run: huggingface-cli login"


def check_python_packages() -> Tuple[bool, str]:
    """Check essential Python packages."""
    packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "datasets": "Datasets",
        "peft": "PEFT (LoRA)",
        "trl": "TRL",
    }

    installed = []
    missing = []

    for pkg, name in packages.items():
        try:
            __import__(pkg)
            installed.append(name)
        except ImportError:
            missing.append(name)

    if missing:
        return False, f"Missing: {', '.join(missing)}. Installed: {', '.join(installed)}"
    return True, f"All essential packages installed: {', '.join(installed)}"


def check_bitsandbytes() -> Tuple[bool, str]:
    """Check if bitsandbytes is available (required for QLoRA)."""
    try:
        import bitsandbytes as bnb
        # Try to verify CUDA support
        try:
            bnb.cuda_setup.main()
            return True, f"BitsAndBytes installed: {getattr(bnb, '__version__', 'version unknown')} (CUDA ready)"
        except Exception:
            return True, f"BitsAndBytes installed: {getattr(bnb, '__version__', 'version unknown')} (CUDA status unknown)"
    except ImportError:
        return False, "BitsAndBytes not installed. Install with: pip install bitsandbytes (required for QLoRA)"


def check_unsloth() -> Tuple[bool, str]:
    """Check if Unsloth is available."""
    try:
        import unsloth
        return True, f"Unsloth installed: {getattr(unsloth, '__version__', 'version unknown')}"
    except ImportError:
        return False, "Unsloth not installed. Install with: pip install --no-deps unsloth unsloth_zoo"


def main():
    print("=" * 60)
    print("DGX Spark Hardware Verification")
    print("=" * 60)
    print()

    checks = [
        ("NVIDIA GPU", check_nvidia_smi),
        ("CUDA Version", check_cuda_version),
        ("GPU Memory", check_memory),
        ("Compute Capability", check_compute_capability),
        ("Docker + GPU", check_docker),
        ("Disk Space", check_disk_space),
        ("Hugging Face Auth", check_huggingface_auth),
        ("Python Packages", check_python_packages),
        ("BitsAndBytes", check_bitsandbytes),
        ("Unsloth", check_unsloth),
    ]

    results = []
    for name, check_func in checks:
        try:
            success, message = check_func()
            results.append((name, success, message))
            status = "PASS" if success else "FAIL"
            icon = "[OK]" if success else "[!!]"
            print(f"{icon} {name}: {message}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"[!!] {name}: Error - {e}")

    print()
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    if passed == total:
        print(f"All {total} checks passed! Ready for training.")
        print("=" * 60)
        return 0
    else:
        print(f"Passed: {passed}/{total}")
        print()
        print("Issues to resolve:")
        for name, success, message in results:
            if not success:
                print(f"  - {name}: {message}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
