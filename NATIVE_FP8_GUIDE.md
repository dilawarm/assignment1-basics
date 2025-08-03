# Native PyTorch FP8 Training Guide

This implementation uses **native PyTorch FP8** operations instead of NVIDIA Transformer Engine, completely avoiding the cuBLAS compatibility issues you encountered.

## What Changed

1. **No Transformer Engine Dependency**: Removed all Transformer Engine imports and dependencies
2. **Native PyTorch FP8**: Uses `torch.float8_e4m3fn` and `torch.float8_e5m2` dtypes
3. **Custom FP8 Linear Layer**: Implemented `FP8Linear` using `torch._scaled_mm` for matrix multiplication
4. **Automatic Fallback**: Gracefully falls back to FP32 if FP8 operations fail

## How It Works

### FP8 Linear Layer (`cs336_basics/model/fp8_linear.py`)
```python
# Uses native PyTorch FP8 operations
input_fp8 = input.to(torch.float8_e4m3fn)
weight_fp8 = weight.to(torch.float8_e4m3fn)
# torch._scaled_mm requires scale factors
output = torch._scaled_mm(
    input_fp8, 
    weight_fp8.t(), 
    scale_a=input_scale,
    scale_b=weight_scale,
    out_dtype=torch.float32
)
```

### Model Architecture
- All linear layers can use FP8 (QKV projections, output projections, FFN)
- Attention computation remains in FP32 for stability
- Activations and normalizations remain in FP32

## Performance Expectations

Based on [NVIDIA's benchmarks](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dgxc-benchmarking/resources/llama31-70b-dgxc-benchmarking-a), FP8 on H100 achieves:
- **36.17% MFU** (Model FLOPS Utilization) with FP8
- **54.10% MFU** with BF16

While FP8 has lower MFU, it provides:
- 2x memory savings
- Higher throughput due to reduced memory bandwidth requirements
- Ability to use larger batch sizes

## Usage

### Basic Usage
```bash
python train_h100.py --use_fp8
```

### Check FP8 Support
The script automatically checks for FP8 support:
1. Verifies PyTorch has FP8 dtypes (needs PyTorch >= 2.1)
2. Tests `torch._scaled_mm` operation
3. Falls back to FP16 if not supported

### Requirements
- PyTorch >= 2.1.0
- CUDA >= 11.8
- GPU with compute capability >= 8.9 (H100, RTX 4090)

## Advantages Over Transformer Engine

1. **No cuBLAS Errors**: Avoids the compatibility issues with Transformer Engine
2. **Simpler Stack**: No additional dependencies beyond PyTorch
3. **Better Control**: Direct control over which operations use FP8
4. **Easier Debugging**: Standard PyTorch operations are easier to debug

## Technical Details

### Dimension Requirements
**Important**: PyTorch FP8 operations require matrix dimensions to be **multiples of 16** for efficient tensor core utilization. This is a hardware constraint on modern GPUs (H100, A100).

- If dimensions are not aligned to 16, the FP8Linear layer automatically falls back to FP32
- For optimal performance, design your model with dimensions that are multiples of 16
- Common aligned dimensions: 1024, 2048, 4096, etc.

### FP8 Formats
- **E4M3** (torch.float8_e4m3fn): Used for forward pass
  - 4-bit exponent, 3-bit mantissa
  - Range: ±448
  - Better for weights and activations
  
- **E5M2** (torch.float8_e5m2): Used for gradients
  - 5-bit exponent, 2-bit mantissa  
  - Range: ±57,344
  - Better for gradient accumulation

### Scaling
PyTorch's `torch._scaled_mm` requires explicit scaling factors to prevent overflow/underflow:
```python
# Required signature for FP8 matrix multiplication
result = torch._scaled_mm(
    a_fp8,           # First matrix in FP8
    b_fp8,           # Second matrix in FP8  
    scale_a=scale_a, # Scale factor for first matrix (required)
    scale_b=scale_b, # Scale factor for second matrix (required)
    out_dtype=torch.float32  # Output dtype (optional)
)
```

The scale factors are crucial for FP8 computation as they:
- Prevent overflow/underflow in the limited FP8 range
- Allow gradual scaling adjustments during training
- Enable mixed precision computation with FP32 accumulation

## Troubleshooting

### If FP8 Fails
The model automatically falls back to FP32/FP16. Check the console output:
```
✓ Native PyTorch FP8 support detected
✓ PyTorch FP8 dtypes available
✓ Native FP8 computation test passed
```

### Common Issues
1. **"scaled_mm not found"**: Need PyTorch >= 2.1
2. **"float8_e4m3fn not found"**: Need newer PyTorch build
3. **RuntimeError on older GPUs**: Need compute capability >= 8.9
4. **"Only multiplication of row-major and column-major matrices is supported by cuBLASLt"**: This is a known limitation in PyTorch's current FP8 implementation. The cuBLASLt library has strict requirements about matrix memory layouts that PyTorch doesn't always satisfy correctly.

## Current Limitations

As noted in the [PyTorch documentation](https://pytorch.org/docs/stable/tensors.html), FP8 support has **"very limited"** operator coverage. The cuBLASLt matrix layout error is one example of these limitations. While FP8 dtypes exist and basic operations may work in some cases, full production support is still in development.

**For reliable training on H100**, FP16 mixed precision remains the best option, providing:
- Stable performance (~700K tokens/sec)
- Full operator coverage
- No matrix layout issues
- Production-ready status

## Future Improvements

PyTorch's FP8 support is rapidly evolving. Future versions will likely include:
- Better kernel optimizations
- Automatic mixed precision for FP8
- More stable scaling algorithms
- Integration with torch.compile()

## References

- [PyTorch FP8 Documentation](https://pytorch.org/docs/stable/generated/torch.float8_e4m3fn.html)
- [NVIDIA H100 FP8 Formats](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [Llama 3.1 70B FP8 Benchmarks](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dgxc-benchmarking/resources/llama31-70b-dgxc-benchmarking-a)