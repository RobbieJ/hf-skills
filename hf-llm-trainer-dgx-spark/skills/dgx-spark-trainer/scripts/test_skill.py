#!/usr/bin/env python3
"""
DGX Spark Skill Test Suite

Comprehensive test script to validate all training frameworks on DGX Spark.
Run this after cloning the repo to verify everything works correctly.

Usage:
    # Run all tests (inside Docker container)
    python test_skill.py

    # Run specific test
    python test_skill.py --test hardware
    python test_skill.py --test native-hf
    python test_skill.py --test unsloth
    python test_skill.py --test llamafactory

    # Quick smoke test only
    python test_skill.py --quick

    # Full test with 7B model
    python test_skill.py --full
"""

import argparse
import subprocess
import sys
import os
import time
from typing import Tuple, List

# Test configuration
QUICK_MODEL = "Qwen/Qwen2.5-0.5B"
QUICK_MODEL_UNSLOTH = "unsloth/Qwen2.5-0.5B-bnb-4bit"
FULL_MODEL = "Qwen/Qwen2.5-7B"
FULL_MODEL_UNSLOTH = "unsloth/Qwen2.5-7B-bnb-4bit"
DATASET = "trl-lib/Capybara"
QUICK_SAMPLES = 20
QUICK_STEPS = 5
FULL_SAMPLES = 100
FULL_STEPS = 20


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def log_info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.END} {msg}")


def log_success(msg: str):
    print(f"{Colors.GREEN}[PASS]{Colors.END} {msg}")


def log_error(msg: str):
    print(f"{Colors.RED}[FAIL]{Colors.END} {msg}")


def log_warning(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.END} {msg}")


def log_header(msg: str):
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}\n")


def test_hardware() -> Tuple[bool, str]:
    """Test 1: Hardware verification"""
    log_header("Test 1: Hardware Verification")

    try:
        import torch

        # Check CUDA
        if not torch.cuda.is_available():
            return False, "CUDA not available"

        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute_cap = torch.cuda.get_device_capability(0)

        log_info(f"GPU: {gpu_name}")
        log_info(f"Memory: {gpu_memory:.1f} GB")
        log_info(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        log_info(f"PyTorch: {torch.__version__}")
        log_info(f"CUDA Version: {torch.version.cuda}")

        # Check if it looks like DGX Spark (128GB unified memory)
        if gpu_memory >= 100:
            log_success("DGX Spark detected (128GB unified memory)")
        else:
            log_warning(f"GPU memory is {gpu_memory:.1f}GB - may not be DGX Spark")

        return True, f"GPU: {gpu_name}, Memory: {gpu_memory:.1f}GB"

    except Exception as e:
        return False, str(e)


def test_imports() -> Tuple[bool, str]:
    """Test 2: Import verification"""
    log_header("Test 2: Import Verification")

    required = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "datasets": "Datasets",
        "peft": "PEFT",
        "trl": "TRL",
        "accelerate": "Accelerate",
    }

    optional = {
        "bitsandbytes": "BitsAndBytes",
        "unsloth": "Unsloth",
    }

    results = []

    # Required packages
    for pkg, name in required.items():
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            log_success(f"{name}: {version}")
            results.append(True)
        except ImportError as e:
            log_error(f"{name}: Not installed")
            results.append(False)

    # Optional packages
    for pkg, name in optional.items():
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            log_success(f"{name}: {version} (optional)")
        except ImportError:
            log_warning(f"{name}: Not installed (optional)")

    if all(results):
        return True, "All required packages installed"
    else:
        return False, "Missing required packages"


def test_native_hf(quick: bool = True) -> Tuple[bool, str]:
    """Test 3: Native Hugging Face TRL training"""
    log_header("Test 3: Native Hugging Face TRL Training")

    model_name = QUICK_MODEL if quick else FULL_MODEL
    max_steps = QUICK_STEPS if quick else FULL_STEPS
    num_samples = QUICK_SAMPLES if quick else FULL_SAMPLES

    log_info(f"Model: {model_name}")
    log_info(f"Samples: {num_samples}, Steps: {max_steps}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset

        # Load tokenizer
        log_info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure quantization for larger models
        if not quick:
            log_info("Configuring 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None

        # Load model
        log_info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if bnb_config is None else None,
            trust_remote_code=True,
        )

        # Add LoRA
        if bnb_config:
            model = prepare_model_for_kbit_training(model)

        log_info("Adding LoRA adapters...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log_info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        # Load dataset
        log_info("Loading dataset...")
        dataset = load_dataset(DATASET, split=f"train[:{num_samples}]")

        # Train
        log_info("Starting training...")
        output_dir = "/tmp/test-native-hf"
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=output_dir,
                max_steps=max_steps,
                per_device_train_batch_size=2 if quick else 1,
                gradient_accumulation_steps=2 if quick else 4,
                logging_steps=1,
                bf16=True,
                gradient_checkpointing=not quick,
                max_seq_length=512 if quick else 1024,
                report_to="none",
            ),
        )

        start_time = time.time()
        trainer.train()
        elapsed = time.time() - start_time

        log_success(f"Training completed in {elapsed:.1f}s")

        # Cleanup
        del model, trainer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return True, f"Native HF training completed in {elapsed:.1f}s"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_unsloth(quick: bool = True) -> Tuple[bool, str]:
    """Test 4: Unsloth training"""
    log_header("Test 4: Unsloth Training")

    model_name = QUICK_MODEL_UNSLOTH if quick else FULL_MODEL_UNSLOTH
    max_steps = QUICK_STEPS if quick else FULL_STEPS
    num_samples = QUICK_SAMPLES if quick else FULL_SAMPLES

    log_info(f"Model: {model_name}")
    log_info(f"Samples: {num_samples}, Steps: {max_steps}")

    try:
        # Check if Unsloth is available
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            log_warning("Unsloth not installed - installing now...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q",
                "--no-deps", "unsloth", "unsloth_zoo", "bitsandbytes"
            ], check=True)
            from unsloth import FastLanguageModel

        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
        import torch

        # Load model
        log_info("Loading model with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512 if quick else 2048,
            load_in_4bit=True,
        )

        # Add LoRA
        log_info("Adding LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )

        # Load dataset
        log_info("Loading dataset...")
        dataset = load_dataset(DATASET, split=f"train[:{num_samples}]")

        # Train
        log_info("Starting training...")
        output_dir = "/tmp/test-unsloth"
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=output_dir,
                max_steps=max_steps,
                per_device_train_batch_size=2 if quick else 1,
                gradient_accumulation_steps=2 if quick else 4,
                logging_steps=1,
                bf16=True,
                report_to="none",
            ),
        )

        start_time = time.time()
        trainer.train()
        elapsed = time.time() - start_time

        log_success(f"Training completed in {elapsed:.1f}s")

        # Cleanup
        del model, trainer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return True, f"Unsloth training completed in {elapsed:.1f}s"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_llamafactory() -> Tuple[bool, str]:
    """Test 5: LLaMA Factory installation check"""
    log_header("Test 5: LLaMA Factory")

    try:
        # Check if llamafactory is installed
        result = subprocess.run(
            ["llamafactory-cli", "version"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            version = result.stdout.strip()
            log_success(f"LLaMA Factory installed: {version}")
            return True, f"LLaMA Factory {version}"
        else:
            log_warning("LLaMA Factory not installed")
            log_info("To install: git clone https://github.com/hiyouga/LLaMA-Factory.git && cd LLaMA-Factory && pip install -e '.[metrics]'")
            return True, "LLaMA Factory not installed (optional)"

    except FileNotFoundError:
        log_warning("LLaMA Factory CLI not found")
        log_info("To install: git clone https://github.com/hiyouga/LLaMA-Factory.git && cd LLaMA-Factory && pip install -e '.[metrics]'")
        return True, "LLaMA Factory not installed (optional)"

    except Exception as e:
        return False, str(e)


def run_all_tests(quick: bool = True) -> List[Tuple[str, bool, str]]:
    """Run all tests and return results"""
    results = []

    # Test 1: Hardware
    success, msg = test_hardware()
    results.append(("Hardware Verification", success, msg))

    # Test 2: Imports
    success, msg = test_imports()
    results.append(("Import Verification", success, msg))

    # Test 3: Native HF
    success, msg = test_native_hf(quick=quick)
    results.append(("Native HF/TRL Training", success, msg))

    # Test 4: Unsloth
    success, msg = test_unsloth(quick=quick)
    results.append(("Unsloth Training", success, msg))

    # Test 5: LLaMA Factory
    success, msg = test_llamafactory()
    results.append(("LLaMA Factory", success, msg))

    return results


def print_summary(results: List[Tuple[str, bool, str]]):
    """Print test summary"""
    log_header("Test Summary")

    passed = 0
    failed = 0

    for name, success, msg in results:
        if success:
            log_success(f"{name}: {msg}")
            passed += 1
        else:
            log_error(f"{name}: {msg}")
            failed += 1

    print()
    print(f"{Colors.BOLD}Results: {passed} passed, {failed} failed{Colors.END}")

    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed! The skill is ready for use.{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}Some tests failed. Check the errors above.{Colors.END}")


def main():
    parser = argparse.ArgumentParser(description="DGX Spark Skill Test Suite")
    parser.add_argument("--test", choices=["hardware", "imports", "native-hf", "unsloth", "llamafactory", "all"],
                        default="all", help="Specific test to run")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (small model, few steps)")
    parser.add_argument("--full", action="store_true", help="Full test (7B model, more steps)")

    args = parser.parse_args()

    # Default to quick test
    quick = not args.full

    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}DGX Spark Skill Test Suite{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"Mode: {'Quick' if quick else 'Full'}")
    print()

    if args.test == "all":
        results = run_all_tests(quick=quick)
        print_summary(results)
    elif args.test == "hardware":
        success, msg = test_hardware()
        print(f"\nResult: {'PASS' if success else 'FAIL'} - {msg}")
    elif args.test == "imports":
        success, msg = test_imports()
        print(f"\nResult: {'PASS' if success else 'FAIL'} - {msg}")
    elif args.test == "native-hf":
        success, msg = test_native_hf(quick=quick)
        print(f"\nResult: {'PASS' if success else 'FAIL'} - {msg}")
    elif args.test == "unsloth":
        success, msg = test_unsloth(quick=quick)
        print(f"\nResult: {'PASS' if success else 'FAIL'} - {msg}")
    elif args.test == "llamafactory":
        success, msg = test_llamafactory()
        print(f"\nResult: {'PASS' if success else 'FAIL'} - {msg}")


if __name__ == "__main__":
    main()
