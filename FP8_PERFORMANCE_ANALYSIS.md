# FP8 Performance Analysis

## The Issue

Your FP8 model is performing **worse** than BF16 (34,777 vs 151,293 tokens/sec). This is because:

1. **FP8 models MUST be compiled** - The [PyTorch blog on FP8 acceleration](https://pytorch.org/blog/accelerating-llama3/) shows that FP8 performance depends on kernel fusion and optimization that only happens with compilation.

2. **Your test didn't compile the FP8 model** - While you compiled the BF16 model, the FP8 model was tested without compilation.

## The Fix

I've updated `optimize_performance.py` to add a new test: **"FP8 + Flash + Compile"**. This should show the true FP8 performance.

Run the updated test:
```bash
python optimize_performance.py
```

## Expected Results

Based on the PyTorch blog and H100 capabilities:
- **BF16 + Flash + Compile**: ~150k tokens/sec (what you got)
- **FP8 + Flash + Compile**: ~250-300k tokens/sec (1.7-2x speedup)

## Why FP8 Needs Compilation

According to the PyTorch documentation:

1. **Kernel Fusion**: FP8 operations need to be fused to minimize memory bandwidth bottlenecks
2. **Layout Optimization**: The compiler optimizes tensor memory layouts for FP8 operations
3. **CUDA Graph Integration**: Compilation enables CUDA graphs which reduce kernel launch overhead

Without compilation, FP8 actually adds overhead from:
- Dynamic scaling computations
- Format conversions
- Unoptimized memory access patterns

## Model Size Issue

Your model has 454M parameters instead of 350M. To get exactly 350M, adjust the configuration in `train_h100.py`:

```python
# Current (454M params)
--dim 1024 --n_layers 24 --n_heads 16 --intermediate_size 4096

# For ~350M params, try:
--dim 896 --n_layers 24 --n_heads 14 --intermediate_size 3584
```

## TorchAO FP8 Implementation

Your TorchAO FP8 implementation is correct:
- ✓ Model uses standard `nn.Linear` layers
- ✓ `convert_to_float8_training` successfully converts 120 modules
- ✓ Dynamic scaling is configured properly

The only issue was the missing compilation step for FP8 testing.

## Training Script

Your `train_h100.py` is correctly configured:
- ✓ FP8 conversion happens before compilation
- ✓ Gradient checkpointing is now disabled by default
- ✓ Compilation happens in the Trainer after FP8 setup

## Next Steps

1. Run the updated `optimize_performance.py` to see compiled FP8 performance
2. If FP8 + Compile shows good results, your training should work well
3. Monitor the first few training steps to ensure you're getting expected tokens/sec

The key insight: **FP8 without compilation is slower than BF16, but FP8 with compilation should be significantly faster.**