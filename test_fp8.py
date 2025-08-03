#!/usr/bin/env python3
"""Quick test script to verify FP8 functionality."""

import sys

import torch


def test_fp8_support():
    """Test if PyTorch FP8 is available and working."""
    print("Testing PyTorch FP8 support...")

    # Check FP8 dtypes
    try:
        _ = torch.float8_e4m3fn
        _ = torch.float8_e5m2
        print("✓ FP8 dtypes available")
    except AttributeError:
        print("✗ FP8 dtypes not available - need PyTorch >= 2.1")
        return False

    # Check CUDA
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return False

    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")

    # Check compute capability
    major, minor = torch.cuda.get_device_capability()
    compute_capability = major + minor / 10
    print(f"✓ Compute capability: {compute_capability}")

    if compute_capability < 8.9:
        print(f"✗ FP8 requires compute capability >= 8.9, found {compute_capability}")
        return False

    # Test FP8 operations
    try:
        # Create test tensors with dimensions divisible by 16 (required for FP8)
        a = torch.randn(16, 32, device="cuda")
        b = torch.randn(32, 32, device="cuda")

        # Convert to FP8
        a_fp8 = a.to(torch.float8_e4m3fn)
        b_fp8 = b.to(torch.float8_e4m3fn)

        # Test scaled matrix multiplication
        scale_a = torch.tensor(1.0, device="cuda")
        scale_b = torch.tensor(1.0, device="cuda")

        result = torch._scaled_mm(a_fp8, b_fp8.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32)

        print("✓ FP8 scaled matrix multiplication successful")
        print(f"  Input shapes: {a_fp8.shape} x {b_fp8.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Output dtype: {result.dtype}")

        # Compare with FP32 result
        result_fp32 = torch.mm(a, b.t())
        max_diff = torch.max(torch.abs(result - result_fp32)).item()
        print(f"  Max difference from FP32: {max_diff:.6f}")

        return True

    except Exception as e:
        print(f"✗ FP8 operations failed: {e}")
        return False


if __name__ == "__main__":
    success = test_fp8_support()
    sys.exit(0 if success else 1)
