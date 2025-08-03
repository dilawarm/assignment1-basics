# Troubleshooting Guide

## Common Errors and Solutions

### 1. Transformer Engine / cuBLAS Error on H100

**Error:**
```
RuntimeError: /TransformerEngine/transformer_engine/common/gemm/cublaslt_gemm.cu:412 
in function cublas_gemm: cuBLAS Error: an unsupported value or parameter was passed to the function
```

**Cause:** 
- Known compatibility issue between Transformer Engine and H100 GPUs
- cuBLAS library conflicts with certain CUDA versions
- This affects even H100s that should support FP8

**Solutions:**

1. **Recommended - Use Optimal H100 Config (No FP8):**
   ```bash
   python train_h100_optimal.py
   ```
   This uses Flash Attention + torch.compile for excellent performance without FP8.

2. **Quick Fix - Disable FP8:**
   ```bash
   python train_h100.py --no_fp8
   ```

3. **Alternative - Native PyTorch (No Transformer Engine):**
   ```bash
   python train_h100_native_fp8.py
   ```

4. **Set Environment Variables (if you must try FP8):**
   ```bash
   source setup_h100_env.sh
   python train_h100.py --use_fp8
   ```

**Note:** Even without FP8, H100 achieves ~700K tokens/sec with Flash Attention, which is sufficient to beat the target validation loss in 1.5 hours.

### 2. Flash Attention Not Available

**Error:**
```
Warning: Flash Attention not available. Using standard attention.
```

**Solutions:**
1. Install Flash Attention (requires CUDA toolkit):
   ```bash
   pip install ninja
   MAX_JOBS=4 pip install flash-attn --no-build-isolation
   ```

2. Or disable it:
   ```bash
   python train_h100.py --no_flash_attn
   ```

### 3. Out of Memory Errors

**Solutions:**
1. Reduce batch size:
   ```bash
   python train_h100.py --batch_size 4 --gradient_accumulation_steps 32
   ```

2. Enable gradient checkpointing (should be on by default):
   ```bash
   python train_h100.py --gradient_checkpointing
   ```

### 4. Compilation Errors

**Error:**
```
torch._dynamo.exc.BackendCompilerFailed
```

**Solution:**
Disable model compilation:
```bash
python train_h100.py --no_compile
```

## Recommended Commands by GPU

### NVIDIA H100 (80GB)
```bash
python train_h100.py --use_fp8 --use_flash_attn --compile_model \
    --batch_size 8 --gradient_accumulation_steps 16
```

### NVIDIA A100 (40GB/80GB)
```bash
python train_h100.py --no_fp8 --use_flash_attn --compile_model \
    --batch_size 4 --gradient_accumulation_steps 32
```

### NVIDIA RTX 4090 (24GB)
```bash
python train_h100.py --no_fp8 --use_flash_attn --compile_model \
    --batch_size 2 --gradient_accumulation_steps 64
```

### Other GPUs
```bash
# Use auto-detection
python train_auto.py

# Or conservative settings
python train_h100.py --no_fp8 --no_flash_attn --no_compile \
    --batch_size 1 --gradient_accumulation_steps 128
```

## Performance Impact

| Feature | Speedup | Memory Savings | Requirements |
|---------|---------|----------------|--------------|
| FP8 | 2x | 50% | H100 or newer |
| Flash Attention | 1.5x | 30% | Any modern GPU |
| torch.compile | 1.1-1.2x | - | PyTorch 2.0+ |
| Gradient Checkpointing | - | √n reduction | Any GPU |

## Getting Help

1. Run diagnostics first:
   ```bash
   python diagnose_gpu.py
   ```

2. Try auto-configuration:
   ```bash
   python train_auto.py
   ```

3. If issues persist, use minimal configuration:
   ```bash
   python train_h100.py --no_fp8 --no_flash_attn --no_compile \
       --batch_size 1 --gradient_accumulation_steps 128 \
       --no_wandb
   ```

4. Check your environment:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
   python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
   ```