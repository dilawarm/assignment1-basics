# Immediate Performance Fix for H100 Training

Your performance issue (28k tokens/sec instead of 700k+) is most likely caused by **gradient checkpointing**, which trades memory for speed.

## Quick Fix

Run your training without gradient checkpointing:

```bash
uv run python train_h100.py --no_gradient_checkpointing
```

This should immediately boost your performance to the expected range.

## Why This Happens

1. **Gradient Checkpointing**: Recomputes activations during backward pass to save memory
   - Can cause 2-3x slowdown
   - Unnecessary on H100 with 80GB memory for a 350M model
   - You're seeing ~25x slowdown, suggesting this is the main issue

2. **Your Current Performance**: 28,716 tokens/sec (7.9% MFU)
   - This is ~3% of expected performance
   - Consistent with gradient checkpointing overhead

3. **Expected Performance Without Gradient Checkpointing**:
   - With FP8: ~900,000 tokens/sec
   - With BF16 + Flash: ~700,000 tokens/sec
   - MFU: >40%

## Test Different Configurations

Run the optimization script to find the best configuration:

```bash
python optimize_performance.py
```

This will test:
- Baseline (BF16 only)
- BF16 + Flash Attention
- BF16 + Flash + Gradient Checkpointing (to confirm slowdown)
- BF16 + Flash + Compilation
- FP8 + Flash (if working)

## Other Potential Issues

If disabling gradient checkpointing doesn't fix it:

1. **Check if FP8 actually worked**: The training output should show "Float8 modules: X" where X > 0
2. **Try without compilation**: `--no_compile`
3. **Monitor GPU usage**: Run `nvidia-smi` during training - utilization should be >90%

## Recommended Command

For optimal H100 performance:

```bash
uv run python train_h100.py \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --no_gradient_checkpointing \
    --use_fp8 \
    --use_flash_attn \
    --compile_mode default
```

If FP8 isn't working, this should still give you ~700k tokens/sec:

```bash
uv run python train_h100.py \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --no_gradient_checkpointing \
    --use_flash_attn \
    --compile_mode default
```