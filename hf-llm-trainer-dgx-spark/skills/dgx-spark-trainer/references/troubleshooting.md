# Troubleshooting Guide for DGX Spark Training

Common issues and solutions when training LLMs on NVIDIA DGX Spark.

## CUDA and GPU Issues

### GPU Not Detected

**Symptoms:**
- `nvidia-smi` shows no GPU
- PyTorch `torch.cuda.is_available()` returns False

**Solutions:**

1. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   ```
   If not working, drivers may need reinstallation.

2. **Check inside Docker:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```

3. **Reinstall nvidia-container-toolkit:**
   ```bash
   sudo apt-get install --reinstall nvidia-container-toolkit
   sudo systemctl restart docker
   ```

4. **Verify Docker GPU flags:**
   ```bash
   docker run --gpus all ...  # Not --gpus=all
   ```

### CUDA Version Mismatch

**Symptoms:**
- `RuntimeError: CUDA error: no kernel image is available`
- Triton compilation errors

**Solutions:**

1. **Check CUDA version:**
   ```bash
   nvcc --version
   nvidia-smi  # Shows driver CUDA version
   ```

2. **Set correct architecture:**
   ```bash
   export TORCH_CUDA_ARCH_LIST="12.0"
   ```

3. **Use correct container:**
   ```bash
   # For DGX Spark (Blackwell), use 25.09 or newer
   docker pull nvcr.io/nvidia/pytorch:25.09-py3
   ```

### Triton Compilation Errors

**Symptoms:**
- `triton.compiler.errors.CompilationError`
- `No kernel image available for execution on the device`

**Solutions:**

1. **Set CUDA architecture:**
   ```bash
   export TORCH_CUDA_ARCH_LIST="12.0"
   ```

2. **Upgrade Triton:**
   ```bash
   pip install --upgrade triton
   ```

3. **Clear Triton cache:**
   ```bash
   rm -rf ~/.triton/cache
   ```

4. **Use NVIDIA container** (has pre-built Triton):
   ```bash
   docker run --gpus all nvcr.io/nvidia/pytorch:25.09-py3
   ```

## Memory Issues

### Out of Memory (OOM)

**Symptoms:**
- `OutOfMemoryError: CUDA out of memory`
- `RuntimeError: CUDA error: out of memory`

**Solutions (in order):**

1. **Clear memory cache:**
   ```bash
   sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
   ```

2. **Reduce batch size:**
   ```python
   per_device_train_batch_size=1
   gradient_accumulation_steps=16
   ```

3. **Enable gradient checkpointing:**
   ```python
   gradient_checkpointing=True
   # Or for Unsloth:
   use_gradient_checkpointing="unsloth"
   ```

4. **Use 8-bit optimizer:**
   ```python
   optim="adamw_8bit"
   ```

5. **Reduce sequence length:**
   ```python
   max_seq_length=1024  # or lower
   ```

6. **Use 4-bit quantization:**
   ```python
   load_in_4bit=True
   ```

7. **Reduce LoRA rank:**
   ```python
   r=8  # or r=4
   ```

8. **Reduce target modules:**
   ```python
   target_modules=["q_proj", "v_proj"]  # Fewer modules
   ```

### Memory Fragmentation

**Symptoms:**
- OOM despite sufficient total memory
- Memory usage grows over time

**Solutions:**

1. **Set memory allocator config:**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
   ```

2. **Clear cache between runs:**
   ```python
   import gc
   import torch
   gc.collect()
   torch.cuda.empty_cache()
   ```

3. **Use `del` for large objects:**
   ```python
   del model
   del trainer
   gc.collect()
   torch.cuda.empty_cache()
   ```

## Training Issues

### Training Loss Not Decreasing

**Solutions:**

1. **Check learning rate:**
   ```python
   # Try lower learning rate
   learning_rate=1e-5  # or 5e-6
   ```

2. **Increase warmup:**
   ```python
   warmup_ratio=0.1  # or warmup_steps=100
   ```

3. **Check dataset format:**
   ```python
   # Verify format matches model's chat template
   print(dataset[0])
   ```

4. **Verify tokenizer:**
   ```python
   # Ensure tokenizer matches model
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   print(tokenizer.chat_template)
   ```

### Training is Very Slow

**Solutions:**

1. **Verify GPU is being used:**
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.current_device())
   ```

2. **Enable bf16:**
   ```python
   bf16=True
   ```

3. **Use Unsloth optimizations:**
   ```python
   use_gradient_checkpointing="unsloth"
   ```

4. **Check DataLoader workers:**
   ```python
   dataloader_num_workers=4
   ```

5. **Use pre-quantized models:**
   ```python
   model_name="unsloth/Qwen2.5-7B-bnb-4bit"  # Pre-quantized
   ```

### NaN Loss

**Solutions:**

1. **Enable gradient clipping:**
   ```python
   max_grad_norm=1.0
   ```

2. **Lower learning rate:**
   ```python
   learning_rate=1e-6
   ```

3. **Check for bad data:**
   ```python
   # Look for empty or very long examples
   for i, example in enumerate(dataset):
       if len(example['text']) == 0 or len(example['text']) > 10000:
           print(f"Bad example at {i}")
   ```

4. **Disable bf16 temporarily:**
   ```python
   bf16=False
   fp16=True
   ```

## Model Loading Issues

### Gated Model Access Denied

**Symptoms:**
- `401 Client Error: Unauthorized`
- `Cannot access gated repo`

**Solutions:**

1. **Login to Hugging Face:**
   ```bash
   huggingface-cli login
   ```

2. **Accept model license:**
   - Visit model page on huggingface.co
   - Accept the license agreement

3. **Set token in environment:**
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

4. **Pass token to Docker:**
   ```bash
   docker run -e HF_TOKEN="$HF_TOKEN" ...
   ```

### Model Download Fails

**Solutions:**

1. **Enable fast transfers:**
   ```bash
   pip install hf_transfer
   export HF_HUB_ENABLE_HF_TRANSFER=1
   ```

2. **Resume interrupted download:**
   ```python
   # Downloads resume automatically from ~/.cache/huggingface
   ```

3. **Use offline mode:**
   ```bash
   # After initial download
   export HF_HUB_OFFLINE=1
   ```

4. **Increase timeout:**
   ```bash
   export HF_HUB_DOWNLOAD_TIMEOUT=600
   ```

### Wrong Model Type

**Symptoms:**
- Model doesn't generate expected output
- Chat format incorrect

**Solutions:**

1. **Use Instruct version:**
   ```python
   # Use Instruct model for chat
   model_name = "Qwen/Qwen2.5-7B-Instruct"  # Not base model
   ```

2. **Check template:**
   ```python
   # LLaMA Factory
   template: qwen  # Must match model

   # Manual
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained(model_name)
   print(tok.chat_template)
   ```

## Docker Issues

### Container Exits Immediately

**Solutions:**

1. **Check logs:**
   ```bash
   docker logs <container_id>
   ```

2. **Run interactively:**
   ```bash
   docker run --gpus all -it nvcr.io/nvidia/pytorch:25.09-py3 bash
   ```

3. **Check entrypoint:**
   ```bash
   docker run --gpus all -it --entrypoint bash nvcr.io/nvidia/pytorch:25.09-py3
   ```

### Permission Denied

**Solutions:**

1. **Add user to docker group:**
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

2. **Fix volume permissions:**
   ```bash
   # Make sure host directories are accessible
   chmod 755 /path/to/volume
   ```

### Disk Space Issues

**Solutions:**

1. **Clean Docker:**
   ```bash
   docker system prune -a
   ```

2. **Remove old images:**
   ```bash
   docker rmi $(docker images -q)
   ```

3. **Check disk usage:**
   ```bash
   df -h
   du -sh ~/.cache/huggingface
   ```

## Unsloth-Specific Issues

### xformers Errors

**Solutions:**

1. **Set CUDA architecture:**
   ```bash
   export TORCH_CUDA_ARCH_LIST="12.0"
   ```

2. **Build from source:**
   ```bash
   pip install xformers --no-build-isolation
   ```

### BitsAndBytes Errors

**Symptoms:**
- `CUDA Setup failed`
- `libbitsandbytes_cuda*.so not found`

**Solutions:**

1. **Install without deps:**
   ```bash
   pip install --no-deps bitsandbytes
   ```

2. **Set library path:**
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
   ```

## LLaMA Factory-Specific Issues

### Template Errors

**Solutions:**

1. **Check available templates:**
   ```python
   from llamafactory.data.template import get_template_and_fix_tokenizer
   # Templates: llama3, qwen, mistral, yi, etc.
   ```

2. **Use correct template for model:**
   ```yaml
   # For Qwen models
   template: qwen

   # For Llama 3 models
   template: llama3
   ```

### Dataset Loading Errors

**Solutions:**

1. **Check dataset format:**
   ```python
   from datasets import load_dataset
   ds = load_dataset("your/dataset")
   print(ds.column_names)
   print(ds[0])
   ```

2. **Verify dataset_info.json:**
   ```json
   {
     "my_dataset": {
       "file_name": "data.json",
       "formatting": "alpaca"
     }
   }
   ```

## Getting Help

### Collect Debug Information

```bash
# System info
nvidia-smi
nvcc --version
python --version
pip list | grep -E "torch|transformers|unsloth|peft|trl"

# Memory info
free -h
df -h

# Docker info
docker --version
docker info | grep -i gpu
```

### Resources

- [NVIDIA DGX Spark Forums](https://forums.developer.nvidia.com/c/dgx-spark/)
- [Unsloth GitHub Issues](https://github.com/unslothai/unsloth/issues)
- [LLaMA Factory GitHub Issues](https://github.com/hiyouga/LLaMA-Factory/issues)
- [Hugging Face Forums](https://discuss.huggingface.co/)
