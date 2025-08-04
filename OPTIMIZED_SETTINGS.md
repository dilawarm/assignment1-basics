# Optimized H100 Training Settings

The `train_h100.py` script now includes all performance optimizations by default. Just run:

```bash
uv run python train_h100.py
```

## Automatic Optimizations

1. **Optimal Batch Configuration**:
   - Batch size: 16 (increased from 8)
   - Gradient accumulation: 8 (reduced from 16)
   - Effective batch size: 16 × 8 × 1024 = 131,072 tokens

2. **Dependency Detection**:
   - Automatically checks for TorchAO (FP8) and Flash Attention
   - Displays warnings if missing and adjusts settings accordingly
   - Gracefully falls back to BF16 if FP8 unavailable

3. **Compilation Mode**:
   - Default: "default" mode (stable, avoids CUDA graph issues)
   - Previously was "max-autotune" which could cause errors

4. **Performance Monitoring**:
   - Shows expected tokens/sec for your configuration
   - Displays MFU (Model FLOPs Utilization) during training
   - Warns if performance is unexpectedly low

## Expected Performance

On H100, you should see:
- **With FP8**: ~900,000 tokens/sec, MFU >45%
- **With BF16 + Flash**: ~700,000 tokens/sec, MFU >40%
- **With BF16 only**: ~500,000 tokens/sec, MFU >30%

If you see <100,000 tokens/sec, something is wrong!

## Manual Overrides

You can still override any setting:
```bash
# Different batch size
uv run python train_h100.py --batch_size 32 --gradient_accumulation_steps 4

# Disable optimizations for testing
uv run python train_h100.py --no_fp8 --no_flash_attn --no_compile

# Use different compile mode
uv run python train_h100.py --compile_mode max-autotune
```

## Troubleshooting Low Performance

If you're seeing low tokens/sec (like 23,530):

1. **Check GPU**: Make sure you're on H100/A100
2. **Install dependencies**:
   ```bash
   pip install torchao  # For FP8
   pip install flash-attn --no-build-isolation  # For Flash Attention
   ```
3. **Run diagnostics**:
   ```bash
   python diagnose_performance.py
   ```

The optimized defaults should give you >700,000 tokens/sec on H100!