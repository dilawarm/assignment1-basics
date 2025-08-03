#!/usr/bin/env python3
"""
Demonstrates the current limitations of PyTorch FP8 support.
This shows why the training script falls back to FP16.
"""

import sys

import torch

print("PyTorch FP8 Limitations Demo")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    major, minor = torch.cuda.get_device_capability()
    print(f"Compute capability: {major}.{minor}")

print("\n1. Testing FP8 dtype availability...")
try:
    _ = torch.float8_e4m3fn
    _ = torch.float8_e5m2
    print("✓ FP8 dtypes are available")
except AttributeError:
    print("✗ FP8 dtypes not available (need PyTorch >= 2.1)")
    sys.exit(1)

if not torch.cuda.is_available():
    print("\n✗ CUDA not available, cannot test FP8 operations")
    sys.exit(1)

print("\n2. Testing basic FP8 tensor creation...")
try:
    x = torch.randn(16, 16, device="cuda")
    x_fp8 = x.to(torch.float8_e4m3fn)
    print(f"✓ Created FP8 tensor with shape {x_fp8.shape} and dtype {x_fp8.dtype}")
except Exception as e:
    print(f"✗ Failed to create FP8 tensor: {e}")

print("\n3. Testing torch._scaled_mm (the FP8 matrix multiplication)...")

# Test 1: Basic matrix multiplication
print("\n   Test 1: Basic 16x16 @ 16x16 multiplication")
try:
    a = torch.randn(16, 16, device="cuda")
    b = torch.randn(16, 16, device="cuda")
    a_fp8 = a.to(torch.float8_e4m3fn)
    b_fp8 = b.to(torch.float8_e4m3fn)
    scale = torch.tensor(1.0, device="cuda")

    result = torch._scaled_mm(a_fp8, b_fp8, scale_a=scale, scale_b=scale)
    print(f"   ✓ Success! Result shape: {result.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 2: Different shapes (common in neural networks)
print("\n   Test 2: Different shapes 32x16 @ 16x32")
try:
    a = torch.randn(32, 16, device="cuda")
    b = torch.randn(16, 32, device="cuda")
    a_fp8 = a.to(torch.float8_e4m3fn).contiguous()
    b_fp8 = b.to(torch.float8_e4m3fn).contiguous()
    scale = torch.tensor(1.0, device="cuda")

    result = torch._scaled_mm(a_fp8, b_fp8, scale_a=scale, scale_b=scale)
    print(f"   ✓ Success! Result shape: {result.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: Transposed multiplication (common in linear layers)
print("\n   Test 3: Transposed multiplication A @ B.T")
try:
    a = torch.randn(32, 16, device="cuda")
    b = torch.randn(32, 16, device="cuda")
    a_fp8 = a.to(torch.float8_e4m3fn).contiguous()
    b_fp8 = b.to(torch.float8_e4m3fn).contiguous()
    b_fp8_t = b_fp8.t().contiguous()
    scale = torch.tensor(1.0, device="cuda")

    result = torch._scaled_mm(a_fp8, b_fp8_t, scale_a=scale, scale_b=scale)
    print(f"   ✓ Success! Result shape: {result.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    if "cuBLASLt" in str(e):
        print("      This is the error that causes FP8 to fail in practice!")

print("\n" + "=" * 50)
print("Summary:")
print("- FP8 dtypes exist in PyTorch >= 2.1")
print("- Basic FP8 operations may work in some cases")
print("- But matrix layout requirements often cause failures")
print("- This is why production code falls back to FP16")
print("\nFor reliable H100 training, use FP16 mixed precision.")
