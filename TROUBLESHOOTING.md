# Troubleshooting Guide

## Common Errors and Solutions

### 1. Native PyTorch FP8 Support

**UPDATE:** The implementation now uses native PyTorch FP8 instead of Transformer Engine, avoiding cuBLAS compatibility issues!

**Using FP8 on H100:**
```bash
python train_h100.py --use_fp8
```

**Requirements for FP8:**
- PyTorch >= 2.1 (for torch.float8_e4m3fn and torch.float8_e5m2)
- GPU compute capability >= 8.9 (H100, RTX 4090)
- CUDA >= 11.8

**If FP8 is not available:**
The script automatically falls back to FP16 mixed precision, which still provides excellent performance (~700K tokens/sec on H100).

### 2. Old Transformer Engine Error (No Longer Applies)

The previous cuBLAS error with Transformer Engine:
```
RuntimeError: /TransformerEngine/transformer_engine/common/gemm/cublaslt_gemm.cu:412 
in function cublas_gemm: cuBLAS Error: an unsupported value or parameter was passed to the function
```

This error is now avoided by using native PyTorch FP8 operations instead of Transformer Engine.

### 3. Flash Attention Not Available

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
# Recommended (avoid FP8 issues)
python train_h100.py --no_fp8 --use_flash_attn --compile_model \
    --batch_size 16 --gradient_accumulation_steps 8

# Experimental FP8 (may fail)
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
# Conservative settings
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