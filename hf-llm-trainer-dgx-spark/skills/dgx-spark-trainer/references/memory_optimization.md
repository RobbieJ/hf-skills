# Memory Optimization for DGX Spark

Best practices for managing memory during LLM training on DGX Spark's 128GB Unified Memory Architecture.

## Understanding Unified Memory

DGX Spark uses Unified Memory Architecture (UMA) where:
- CPU and GPU share the same 128GB memory pool
- No PCIe transfers between CPU and GPU memory
- Memory can be dynamically allocated to either processor
- Full 128GB is accessible by the GPU

### Benefits

- Load larger models than typical GPU VRAM allows
- No data copy overhead between CPU and GPU
- Efficient for models that exceed traditional GPU memory

### Considerations

- Memory bandwidth (273 GB/s) is lower than HBM in datacenter GPUs
- Large models may train slower due to bandwidth limitations
- System processes also use unified memory

## Memory Usage by Component

### Model Memory

| Model Size | FP16 | 8-bit | 4-bit |
|------------|------|-------|-------|
| 1B | ~2GB | ~1GB | ~0.5GB |
| 7B | ~14GB | ~7GB | ~4GB |
| 13B | ~26GB | ~13GB | ~7GB |
| 34B | ~68GB | ~34GB | ~17GB |
| 70B | ~140GB | ~70GB | ~35GB |

### Optimizer States

| Optimizer | Memory per Parameter |
|-----------|---------------------|
| Adam/AdamW | 8 bytes (2 states) |
| AdamW 8-bit | 2 bytes |
| SGD | 4 bytes (momentum) |

### Gradient Memory

- Full precision gradients: 4 bytes per parameter
- Gradient checkpointing reduces this significantly

## Memory Optimization Techniques

### 1. Use QLoRA (4-bit Quantization)

Most effective technique for large models:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-70B-bnb-4bit",
    load_in_4bit=True,  # Critical for memory
)
```

**Impact:** ~4x memory reduction for model weights

### 2. LoRA (Low-Rank Adaptation)

Train only small adapter weights:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,           # Lower = less memory
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

**Impact:** Train only ~0.1-1% of parameters

### 3. Gradient Checkpointing

Trade compute for memory:

```python
# Unsloth optimized (recommended)
use_gradient_checkpointing="unsloth"

# In SFTConfig
SFTConfig(
    gradient_checkpointing=True,
)
```

**Impact:** ~50-70% gradient memory reduction

### 4. Reduce Batch Size

```python
SFTConfig(
    per_device_train_batch_size=1,  # Minimum
    gradient_accumulation_steps=16,  # Compensate
)
```

**Impact:** Linear reduction in activation memory

### 5. 8-bit Optimizer

```python
SFTConfig(
    optim="adamw_8bit",  # Instead of adamw_torch
)
```

**Impact:** ~4x reduction in optimizer memory

### 6. Reduce Sequence Length

```python
FastLanguageModel.from_pretrained(
    max_seq_length=1024,  # Instead of 2048+
)
```

**Impact:** Linear reduction in attention memory

### 7. Mixed Precision (bf16)

```python
SFTConfig(
    bf16=True,  # Use bfloat16 on Blackwell
)
```

**Impact:** ~2x reduction in activation memory

## Clearing Memory Cache

### Before Training

DGX Spark's UMA can have fragmented memory. Clear before loading:

```bash
# System-level cache clear
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

```python
# Python-level
import gc
import torch

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

### During Training (if OOM)

```python
def clear_memory():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Call between training runs or on OOM
clear_memory()
```

## Memory Monitoring

### nvidia-smi

```bash
# Continuous monitoring
watch -n 1 nvidia-smi

# One-time check
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
```

### Python Monitoring

```python
import torch

def print_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")

# Call during training
print_memory()
```

### Memory Profiling

```python
# Enable memory profiling
import torch
torch.cuda.memory._record_memory_history(enabled=True)

# ... training code ...

# Save snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

## Memory Budget Planning

### Example: 70B Model with QLoRA

```
Component                Memory
─────────────────────────────────
Base model (4-bit)       ~35GB
LoRA adapters (fp16)     ~0.5GB
Optimizer states         ~2GB
Gradients                ~2GB
Activations (bs=1)       ~10GB
System overhead          ~5GB
─────────────────────────────────
Total                    ~55GB
Available                128GB
Headroom                 ~73GB
```

### Example: 7B Model Full Fine-tune (NOT recommended)

```
Component                Memory
─────────────────────────────────
Base model (fp16)        ~14GB
Gradients (fp16)         ~14GB
Optimizer (Adam)         ~56GB
Activations (bs=4)       ~20GB
─────────────────────────────────
Total                    ~104GB
Available                128GB
Headroom                 ~24GB (risky!)
```

**Recommendation:** Always use LoRA/QLoRA for models >3B

## Recommended Configurations by Model Size

### Small Models (<3B)

```python
# Full fine-tune possible but LoRA recommended
SFTConfig(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    bf16=True,
)
```

### Medium Models (3-13B)

```python
# QLoRA recommended
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="...-bnb-4bit",
    load_in_4bit=True,
)

SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_8bit",
    bf16=True,
)
```

### Large Models (13-70B)

```python
# QLoRA required, conservative settings
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="...-bnb-4bit",
    load_in_4bit=True,
    max_seq_length=2048,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # Lower rank
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_gradient_checkpointing="unsloth",
)

SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="adamw_8bit",
    bf16=True,
)
```

### Very Large Models (70B+)

```python
# Maximum optimization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    load_in_4bit=True,
    max_seq_length=1024,  # Reduced
)

model = FastLanguageModel.get_peft_model(
    model,
    r=4,  # Minimum rank
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],  # Fewer modules
    use_gradient_checkpointing="unsloth",
)

SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
    optim="adamw_8bit",
    bf16=True,
    max_grad_norm=0.3,  # Gradient clipping
)
```

## Troubleshooting OOM

### Step-by-Step Resolution

1. **Clear cache first**
   ```bash
   sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
   ```

2. **Reduce batch size to 1**
   ```python
   per_device_train_batch_size=1
   ```

3. **Enable gradient checkpointing**
   ```python
   gradient_checkpointing=True
   # or
   use_gradient_checkpointing="unsloth"
   ```

4. **Use 8-bit optimizer**
   ```python
   optim="adamw_8bit"
   ```

5. **Reduce sequence length**
   ```python
   max_seq_length=1024  # or lower
   ```

6. **Reduce LoRA rank**
   ```python
   r=4  # or r=8
   ```

7. **Reduce target modules**
   ```python
   target_modules=["q_proj", "v_proj"]  # Only attention queries and values
   ```

8. **Use smaller model**
   - Try 7B instead of 13B
   - Or use more aggressively quantized version

### If Still OOM

The model may be too large for DGX Spark. Consider:
- Using a smaller model
- Using cloud training (HF Jobs) with larger GPUs
- Connecting two DGX Sparks for 256GB memory
