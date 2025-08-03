#!/usr/bin/env python3
"""Diagnose GPU capabilities and recommend settings."""

import sys

import torch


def main():
    print("=" * 80)
    print("GPU Diagnostics for H100-Optimized Training")
    print("=" * 80)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("   Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support.")
        sys.exit(1)

    # Get GPU information
    gpu_count = torch.cuda.device_count()
    print(f"\n✓ CUDA is available with {gpu_count} GPU(s)")

    for i in range(gpu_count):
        print(f"\nGPU {i}:")
        gpu_name = torch.cuda.get_device_name(i)
        print(f"  Name: {gpu_name}")

        # Get compute capability
        major, minor = torch.cuda.get_device_capability(i)
        compute_capability = major + minor / 10
        print(f"  Compute Capability: {compute_capability}")

        # Get memory
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  Total Memory: {total_memory:.1f} GB")

        # Check CUDA version
        print(f"\nCUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")

        # Recommendations
        print("\n" + "-" * 40)
        print("Recommendations:")

        # FP8 support
        if compute_capability >= 8.9:
            print("✓ FP8 training is supported (H100 or newer)")
            fp8_flag = "--use_fp8"
        else:
            print("✗ FP8 not supported (requires compute capability >= 8.9)")
            print("  Use --no_fp8 flag when training")
            fp8_flag = "--no_fp8"

        # Check for Transformer Engine
        try:
            import transformer_engine

            print("✓ Transformer Engine is installed")
        except ImportError:
            print("✗ Transformer Engine not installed")
            print("  FP8 will not be available")
            fp8_flag = "--no_fp8"

        # Check for Flash Attention
        try:
            import flash_attn

            print("✓ Flash Attention is installed")
            flash_flag = "--use_flash_attn"
        except ImportError:
            print("✗ Flash Attention not installed")
            print("  Training will be slower")
            flash_flag = "--no_flash_attn"

        # Provide recommended command
        print("\n" + "-" * 40)
        print("Recommended training command:")
        print(f"\npython train_h100.py {fp8_flag} {flash_flag} \\")
        print("    --batch_size 8 \\")
        print("    --gradient_accumulation_steps 16 \\")
        print("    --compile_model")

        # Alternative command for troubleshooting
        print("\nIf you encounter errors, try:")
        print("\npython train_h100.py --no_fp8 --no_flash_attn --no_compile \\")
        print("    --batch_size 4 \\")
        print("    --gradient_accumulation_steps 32")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
