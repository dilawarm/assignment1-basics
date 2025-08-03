#!/usr/bin/env python3
"""
Automatic training script that detects GPU capabilities and runs with appropriate settings.
"""

import os
import subprocess
import sys

import torch


def detect_gpu_capabilities():
    """Detect GPU capabilities and return appropriate flags."""
    flags = []

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    compute_capability = major + minor / 10

    print(f"Detected GPU: {gpu_name}")
    print(f"Compute Capability: {compute_capability}")

    # FP8 support (requires compute capability >= 8.9)
    if compute_capability >= 8.9:
        try:
            import transformer_engine

            print("✓ FP8 supported and Transformer Engine available")
            flags.append("--use_fp8")
        except ImportError:
            print("✗ Transformer Engine not installed, disabling FP8")
            flags.append("--no_fp8")
    else:
        print("✗ FP8 not supported on this GPU")
        flags.append("--no_fp8")

    # Flash Attention
    try:
        import flash_attn

        print("✓ Flash Attention available")
        flags.append("--use_flash_attn")
    except ImportError:
        print("✗ Flash Attention not installed")
        flags.append("--no_flash_attn")

    # Model compilation (can be problematic on some systems)
    if compute_capability >= 7.0:
        print("✓ Model compilation supported")
        flags.append("--compile_model")
    else:
        print("✗ Model compilation not recommended")
        flags.append("--no_compile")

    # Adjust batch size based on GPU memory
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_memory_gb >= 80:  # H100 80GB
        batch_size = 8
        grad_accum = 16
    elif total_memory_gb >= 40:  # A100 40GB
        batch_size = 4
        grad_accum = 32
    elif total_memory_gb >= 24:  # RTX 3090/4090
        batch_size = 2
        grad_accum = 64
    else:
        batch_size = 1
        grad_accum = 128

    flags.extend(["--batch_size", str(batch_size), "--gradient_accumulation_steps", str(grad_accum)])

    print(f"✓ Using batch_size={batch_size}, gradient_accumulation={grad_accum}")

    return flags


def main():
    print("=" * 80)
    print("Auto-detecting GPU capabilities...")
    print("=" * 80)

    # Detect capabilities
    flags = detect_gpu_capabilities()

    # Add any additional arguments passed by user
    flags.extend(sys.argv[1:])

    # Construct command
    cmd = ["python", "train_h100.py"] + flags

    print("\n" + "-" * 80)
    print("Running command:")
    print(" ".join(cmd))
    print("-" * 80 + "\n")

    # Run the training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
