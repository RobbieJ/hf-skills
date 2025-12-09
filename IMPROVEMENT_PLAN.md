# LLM Trainer DGX Spark Skill - Improvement Plan

## Status: IMPLEMENTED

Most improvements in this plan have been implemented. See the "Implementation Status" section at the bottom.

## Executive Summary

The DGX Spark LLM trainer skill is a well-structured Claude Code skill for local LLM training on NVIDIA DGX Spark hardware. It provides comprehensive documentation and training scripts for three frameworks (Unsloth, LLaMA Factory, Native HF/TRL). This plan identifies areas for improvement and clarification.

---

## Current State Assessment

### Strengths
- **Comprehensive documentation** - SKILL.md is thorough with clear quick-start guides
- **Three framework support** - Good coverage of Unsloth, LLaMA Factory, and Native HF/TRL
- **Production-ready scripts** - Training scripts are well-documented with environment variable configuration
- **Hardware-aware** - Good memory guidelines and DGX Spark-specific optimizations
- **Solid test suite** - `test_skill.py` covers multiple frameworks

### Areas for Improvement

---

## Improvement Plan

### 1. Plugin Manifest Updates

**File:** `plugin.json`

| Issue | Current State | Recommendation |
|-------|---------------|----------------|
| Missing Native HF/TRL mention | Description only mentions Unsloth and LLaMA Factory | Add "native Hugging Face TRL" to description |
| Keywords incomplete | Missing "huggingface", "trl", "transformers", "peft" | Add missing keywords for discoverability |
| No homepage/docs link | Empty | Add link to SKILL.md or external docs |

### 2. SKILL.md Clarifications

**File:** `skills/dgx-spark-trainer/SKILL.md`

| Line | Issue | Recommendation |
|------|-------|----------------|
| 66 | CUDA 12.0 in Docker test command | Update to match CUDA 12.9+ requirement stated earlier |
| 91 | CUDA Compute listed as 12.1 | Verify against actual Blackwell specs (should be 12.x) |
| 121-122 | Docker container version hardcoded | Consider mentioning "25.09" as example, note to use latest |
| 127-131 | pip install order | Clarify `--no-deps` requirement rationale for Unsloth |

### 3. Training Script Improvements

#### 3.1 `train_sft_unsloth.py`

| Issue | Recommendation |
|-------|----------------|
| No error handling for missing HF token with gated models | Add pre-check for `HF_TOKEN` when using gated models |
| Hard-coded LoRA target modules | Make `TARGET_MODULES` configurable via environment variable |
| No progress callback | Consider adding optional WandB/TensorBoard integration via env var |
| Missing dataset format validation | Add check that dataset has expected columns |

#### 3.2 `train_sft_huggingface.py`

| Issue | Recommendation |
|-------|----------------|
| Imports `TrainingArguments` but doesn't use it | Remove unused import (line 41-42) |
| `dataset_text_field="messages"` hardcoded | Document or make configurable for different dataset formats |
| Inconsistent optimizer | Uses `adamw_torch` while Unsloth script uses `adamw_8bit` - document the difference |

#### 3.3 `train_dpo_huggingface.py`

| Issue | Recommendation |
|-------|----------------|
| No train/eval split | Add eval dataset support like the SFT scripts |
| Missing dataset column validation | Add check for required DPO columns (prompt, chosen, rejected) |
| No example DPO datasets listed | Add comments with recommended DPO datasets |

### 4. Verification Script Improvements

**File:** `scripts/verify_dgx_spark.py`

| Issue | Recommendation |
|-------|----------------|
| Docker GPU check runs container (slow) | Add `--quick` flag to skip container test |
| No check for bitsandbytes | Add bitsandbytes verification (critical for QLoRA) |
| HF token check may miss token locations | Check `HF_HOME` environment variable as well |
| No memory availability check | Add check that sufficient memory is free (not just total) |

### 5. Test Suite Improvements

**File:** `scripts/test_skill.py`

| Issue | Recommendation |
|-------|----------------|
| No DPO training test | Add `test_dpo()` function |
| Tests use `/tmp/` which may fill | Use unique temp directories, cleanup after tests |
| No GPU memory cleanup validation | Add post-test memory check to ensure proper cleanup |
| Missing timeout handling | Add test timeout to prevent hanging |

### 6. New Features to Add

#### 6.1 Multi-GPU Support Documentation
- DGX Spark supports dual-Spark configurations
- Add section on connecting two DGX Sparks for 256GB training
- Document distributed training setup

#### 6.2 Model Export Scripts
- Add `export_to_gguf.py` script (mentioned but not provided)
- Add `merge_lora_adapters.py` utility

#### 6.3 Inference Testing
- Add `test_inference.py` to verify trained models work
- Include chat template support for instruct models

#### 6.4 Dataset Preparation Guide
- Add `references/dataset_preparation.md`
- Document common dataset formats (Alpaca, ShareGPT, etc.)
- Provide conversion utilities

### 7. Documentation Gaps

#### 7.1 Missing References
| Reference Mentioned | Status | Action |
|---------------------|--------|--------|
| `references/hardware_specs.md` | May not exist | Create or verify |
| `references/troubleshooting.md` | May not exist | Create or verify |

#### 7.2 Inconsistencies to Fix
| Location | Issue |
|----------|-------|
| SKILL.md line 574-576 | NVIDIA Build links may need verification |
| Multiple files | Docker container version `25.09-py3` - verify this exists |

### 8. Security & Best Practices

| Issue | Recommendation |
|-------|----------------|
| HF_TOKEN in docker command visible in history | Document using `--env-file` or Docker secrets |
| No .gitignore for outputs | Add `.gitignore` template for training outputs |
| Scripts run with elevated privileges | Document minimum required permissions |

### 9. Configuration Improvements

#### 9.1 Unified Config Loader
Create a shared configuration utility:

```python
# Proposed: scripts/utils/config.py
def load_config():
    """Load configuration from environment, YAML, or defaults."""
    pass
```

#### 9.2 YAML Config for Native HF
The Unsloth template exists but Native HF scripts don't support YAML config - add parity.

### 10. Quality of Life Improvements

| Feature | Description |
|---------|-------------|
| Training progress bar | Add `tqdm` for dataset loading |
| Estimated time remaining | Log ETA during training |
| Memory usage warnings | Warn if approaching 90% memory before OOM |
| Resume from checkpoint | Document and test checkpoint resume |

---

## Implementation Priority

### Phase 1 - Critical (Bug fixes & accuracy)
1. Fix plugin.json to include Native HF/TRL
2. Verify/update CUDA and compute capability references
3. Add missing imports cleanup
4. Add bitsandbytes to verification script

### Phase 2 - Important (Usability)
1. Add environment variable for LoRA target modules
2. Add DPO training test
3. Create missing reference documents
4. Add dataset validation to training scripts

### Phase 3 - Enhancement (Features)
1. Add model export utilities
2. Add inference testing script
3. Add multi-GPU documentation
4. Create dataset preparation guide

### Phase 4 - Polish
1. Add unified config loader
2. Add progress indicators
3. Add memory warnings
4. Security documentation improvements

---

## File-by-File Change Summary

| File | Changes Needed |
|------|----------------|
| `plugin.json` | Update description, add keywords |
| `SKILL.md` | Fix CUDA version, clarify Docker version, minor edits |
| `train_sft_unsloth.py` | Add HF token check, make target_modules configurable |
| `train_sft_huggingface.py` | Remove unused import, document dataset_text_field |
| `train_dpo_huggingface.py` | Add eval split, add dataset validation |
| `verify_dgx_spark.py` | Add bitsandbytes check, add --quick flag |
| `test_skill.py` | Add DPO test, add cleanup, add timeout |
| `references/` | Create missing docs if needed |

---

## Notes

- The skill is already well-designed and functional
- Most improvements are incremental enhancements
- No major architectural changes needed
- Focus should be on consistency and completeness

---

## Implementation Status

### Completed (Phase 1 & 2)

| Task | Status | Files Modified |
|------|--------|----------------|
| Fix plugin.json - add Native HF/TRL | ✅ Done | `plugin.json` |
| Add missing keywords to plugin.json | ✅ Done | `plugin.json` |
| Fix CUDA version inconsistencies | ✅ Done | `SKILL.md`, `hardware_specs.md`, `troubleshooting.md` |
| Add Docker version clarification | ✅ Done | `SKILL.md` |
| Document --no-deps rationale | ✅ Done | `SKILL.md` |
| Remove unused imports | ✅ Done | `train_sft_huggingface.py` |
| Add bitsandbytes check | ✅ Done | `verify_dgx_spark.py` |
| Add HF_HOME env check | ✅ Done | `verify_dgx_spark.py` |
| Faster Docker GPU check | ✅ Done | `verify_dgx_spark.py` |
| Configurable LoRA target modules | ✅ Done | `train_sft_unsloth.py`, `train_sft_huggingface.py`, `train_dpo_huggingface.py` |
| Dataset format validation | ✅ Done | `train_sft_unsloth.py`, `train_sft_huggingface.py`, `train_dpo_huggingface.py` |
| Add train/eval split to DPO | ✅ Done | `train_dpo_huggingface.py` |
| Add DPO test | ✅ Done | `test_skill.py` |
| Verify reference docs exist | ✅ Done | All 7 reference docs confirmed |

### Remaining (Phase 3 & 4 - Optional Future Work)

| Task | Status | Priority |
|------|--------|----------|
| Add model export utilities | Not started | Low |
| Add inference testing script | Not started | Low |
| Multi-GPU documentation | Not started | Low |
| Dataset preparation guide | Not started | Low |
| Unified config loader | Not started | Low |
| Progress indicators | Not started | Low |
| Memory warnings | Not started | Low |
| Security documentation | Not started | Low |
