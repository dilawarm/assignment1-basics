#!/usr/bin/env python3
"""
H100-specific training script with proper FP8 setup.
This script ensures all environment variables are set correctly for H100.
"""

import os
import sys

# Set environment variables BEFORE importing anything else
# These are critical for H100 FP8 training
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["NVTE_FLASH_ATTN"] = "0"  # Use our own Flash Attention
os.environ["NVTE_FUSED_ATTN"] = "0"  # Avoid conflicts
os.environ["NVTE_BIAS_GELU_NVFUSION"] = "0"  # Disable problematic fusion
os.environ["NVTE_MASKED_SOFTMAX_FUSION"] = "0"  # Disable softmax fusion
os.environ["CUDNN_LOGINFO_DBG"] = "0"  # Reduce debug output
os.environ["CUDNN_LOGERR_DBG"] = "0"
os.environ["NCCL_DEBUG"] = "WARN"  # Only show warnings

# Now import everything
import argparse
import subprocess

import torch


def check_h100_compatibility():
    """Check if we're on an H100 and everything is compatible."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    compute_capability = major + minor / 10

    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {compute_capability}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    if "H100" not in gpu_name:
        print("\nWARNING: This script is optimized for H100 GPUs!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            return False

    if compute_capability < 9.0:
        print(f"\nERROR: Compute capability {compute_capability} < 9.0 (H100 requirement)")
        return False

    # Check Transformer Engine
    try:
        import transformer_engine

        print(f"✓ Transformer Engine version: {transformer_engine.__version__}")
    except ImportError:
        print("\nERROR: Transformer Engine not installed!")
        print("Install with: pip install transformer-engine")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="H100 FP8 Training")
    parser.add_argument("--force", action="store_true", help="Force run even with warnings")
    args, unknown = parser.parse_known_args()

    print("=" * 80)
    print("H100 FP8 Training Setup")
    print("=" * 80)

    if not check_h100_compatibility() and not args.force:
        print("\nExiting due to compatibility issues.")
        print("Use --force to override.")
        sys.exit(1)

    # Build command with H100-optimized defaults
    cmd = [
        "python",
        "train_h100.py",
        "--use_fp8",
        "--use_flash_attn",
        "--compile_model",
        "--batch_size",
        "8",
        "--gradient_accumulation_steps",
        "16",
        "--max_hours",
        "1.5",
    ]

    # Add any additional arguments
    cmd.extend(unknown)

    print("\n" + "-" * 80)
    print("Running command:")
    print(" ".join(cmd))
    print("-" * 80 + "\n")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
        print("\nTroubleshooting suggestions:")
        print("1. Try with reduced batch size: --batch_size 4")
        print("2. Try without compilation: --no_compile")
        print("3. Check GPU memory with: nvidia-smi")
        print("4. See TROUBLESHOOTING.md for more help")
        sys.exit(1)


if __name__ == "__main__":
    main()
