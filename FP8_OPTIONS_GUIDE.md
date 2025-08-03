# Complete Guide to FP8 Training Options

## Quick Answer: Use TorchAO

```bash
# Install
pip install torchao

# Run with FP8
python train_h100.py --use_fp8 --fp8_backend torchao
```

✅ Stable, easy, ~1.5x speedup over FP16  
✅ Works with torch.compile and FSDP  
✅ Official PyTorch solution  

---

## Current State of FP8 Training (2024)

FP8 training is still evolving. Here are ALL your options, ranked by reliability:

## Option 1: TorchAO (Recommended for FP8)

**Status**: Official PyTorch architecture optimization library  
**Performance**: 1.5x speedup over FP16  
**Reliability**: Excellent - this is the successor to float8_experimental

```bash
pip install torchao
```

TorchAO is PyTorch's official library that includes the float8 implementation previously in float8_experimental. It's actively maintained and used in production.

### Usage in train_h100.py:
```bash
python train_h100.py --use_fp8 --fp8_backend torchao
```

### Pros:
- Handles scaling automatically
- Works with torch.compile
- Supports FSDP
- Actively maintained by PyTorch team

### Cons:
- External dependency
- Slightly more memory usage than FP16

### Integration with TorchAO:
```python
from torchao.float8 import convert_to_float8_training

# Simple one-line conversion
convert_to_float8_training(model)

# Or with custom config
from torchao.float8 import Float8LinearConfig, CastConfig, ScalingType

config = Float8LinearConfig(
    cast_config_weight=CastConfig(scaling_type=ScalingType.DYNAMIC),
    cast_config_input=CastConfig(scaling_type=ScalingType.DYNAMIC),
    cast_config_grad_output=CastConfig(scaling_type=ScalingType.DYNAMIC),
)
convert_to_float8_training(model, config=config)
```

## Option 1a: float8_experimental (Archived - DO NOT USE)

**⚠️ IMPORTANT**: float8_experimental has been **archived** and moved to TorchAO. 
The old repository at https://github.com/pytorch-labs/float8_experimental is no longer maintained.

**What to do instead**: Use TorchAO (see Option 1 above)
```bash
# OLD (don't use):
# import float8_experimental

# NEW (use this):
import torchao.float8
```

## Option 2: NVIDIA Transformer Engine (If you can fix cuBLAS issues)

**Status**: Production-ready but compatibility issues  
**Performance**: Best theoretical performance  
**Reliability**: Depends on your system

### Fixing cuBLAS errors:
```bash
# Try different CUDA/driver combinations
# Latest drivers sometimes help:
# Driver 545.x or 550.x with CUDA 12.3+

# Environment variables that might help:
export NVTE_ALLOW_INTERNAL_API=1
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0
```

## Option 3: Native PyTorch FP8 (Limited)

**Status**: Very limited operator coverage  
**Performance**: When it works, good  
**Reliability**: Poor - expect failures

Current limitations ([GitHub issue #123761](https://github.com/pytorch/pytorch/issues/123761)):
- Only `torch._scaled_mm` is available
- Requires specific matrix layouts
- No support in torch.matmul or other ops
- Lots of overhead from scaling

## Option 4: Custom FP8 Implementation

Some teams write custom CUDA kernels for FP8. This requires:
- Deep CUDA expertise
- Custom autograd functions
- Extensive testing

## Option 5: Microsoft DeepSpeed FP8

**Status**: Available in DeepSpeed 0.14+  
**Performance**: Competitive with other solutions  
**Reliability**: Good for DeepSpeed users

```bash
pip install deepspeed>=0.14.0
```

### Features:
- Integrated with ZeRO optimization
- Supports various FP8 formats
- Good distributed training support

### Usage:
```python
# In DeepSpeed config
"fp16": {
    "enabled": false
},
"fp8": {
    "enabled": true,
    "format": "e4m3"
}
```

## Option 6: Alternative Approaches

### Use BF16 instead of FP8
```python
# BF16 is stable and well-supported
model = model.to(torch.bfloat16)
```

### Quantization-Aware Training (QAT)
- Use INT8 quantization instead of FP8
- Tools like PyTorch's quantization API

### Mixed Precision with Different Ratios
- Keep critical layers in FP32
- Use FP16/BF16 for most layers
- Manual FP8 only for large matmuls

## Performance Comparison

| Method | Speedup vs FP32 | Stability | Ease of Use | Recommendation |
|--------|----------------|-----------|-------------|----------------|
| FP16 Mixed Precision | 1.5-2x | Excellent | Easy | Best fallback |
| BF16 Mixed Precision | 1.4-1.8x | Excellent | Easy | Good alternative |
| **TorchAO Float8** | **2.2-2.5x** | **Excellent** | **Easy** | **Best for FP8** |
| float8_experimental | 2-2.4x | Good | Moderate | Use TorchAO instead |
| Transformer Engine | 2.5-3x | System-dependent | Hard | If you can fix issues |
| Native PyTorch FP8 | Variable | Very Poor | Hard | Not recommended |

## Recommendation Decision Tree

```
Do you absolutely need FP8?
├─ No → Use FP16/BF16 mixed precision (stable, fast enough)
└─ Yes → 
    ├─ Want easy setup + stability? → TorchAO (RECOMMENDED)
    ├─ Have NVIDIA support? → Try Transformer Engine
    ├─ Using DeepSpeed? → DeepSpeed FP8
    ├─ Research only? → Try native PyTorch FP8
    └─ Have CUDA expertise? → Custom implementation
```

## Future Outlook

Based on PyTorch discussions:
- Better FP8 support is coming (no timeline)
- torch.matmul FP8 support is requested
- Automatic mixed precision for FP8 is planned
- Better torch.compile integration

## Summary

**For production H100 training today**:
1. **Use TorchAO for FP8** - it's stable, easy, and gives ~1.5x speedup
2. Fall back to FP16 mixed precision if FP8 isn't critical
3. Avoid native PyTorch FP8 (torch._scaled_mm) due to cuBLAS issues

With TorchAO, FP8 training is now practical and stable. The 1.5x speedup is significant and the setup is as easy as one line of code.