#!/usr/bin/env python3
"""
Optimal H100 training script without FP8.
This configuration works reliably on H100 and still achieves excellent performance.
"""

import subprocess
import sys

# Optimal settings for H100 without problematic FP8
cmd = [
    "python",
    "train_h100.py",
    "--no_fp8",  # Avoid Transformer Engine cuBLAS issues
    "--use_flash_attn",  # Flash Attention works great on H100
    "--compile_model",  # torch.compile for extra speed
    "--batch_size",
    "16",  # H100 can handle larger batches
    "--gradient_accumulation_steps",
    "8",  # 16*8*1024 = 131K tokens per update
    "--max_hours",
    "1.5",
]

# Add any additional arguments
cmd.extend(sys.argv[1:])

print("=" * 80)
print("H100 Optimal Training Configuration")
print("=" * 80)
print("This configuration avoids FP8/cuBLAS issues while still achieving excellent")
print("performance on H100 through Flash Attention and torch.compile optimization.")
print()
print("Expected performance:")
print("  - ~700K tokens/sec (without FP8)")
print("  - Target validation loss: 2.90-2.95")
print("  - Training time: 1.5 hours")
print("=" * 80)
print()
print("Command:", " ".join(cmd))
print()

# Run the training
sys.exit(subprocess.call(cmd))
