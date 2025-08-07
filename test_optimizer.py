#!/usr/bin/env python3
"""Test script to verify optimizer configuration works correctly."""

import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_optimizer_configurations():
    """Test different AdamW configurations to find what works."""
    print("🧪 Testing AdamW optimizer configurations...")

    # Create a simple model for testing
    model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 50257))

    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer_groups = [
        {"params": decay_params, "weight_decay": 0.1, "lr": 4e-4},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": 4e-4},
    ]

    optimizer_kwargs = {
        "lr": 4e-4,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }

    configurations = [
        {"name": "Fused AdamW", "fused": True, "foreach": False},
        {"name": "Foreach AdamW", "fused": False, "foreach": True},
        {"name": "Standard AdamW", "fused": False, "foreach": False},
    ]

    for config in configurations:
        try:
            print(f"\n🔧 Testing {config['name']}...")

            optimizer = AdamW(optimizer_groups, fused=config["fused"], foreach=config["foreach"], **optimizer_kwargs)

            # Test a simple forward/backward pass
            x = torch.randn(4, 1024)
            y = model(x)
            loss = y.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_params = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
            print(f"✅ {config['name']} works! Parameters: {total_params:,}")

        except Exception as e:
            print(f"❌ {config['name']} failed: {e}")

    print("\n🎉 Optimizer testing complete!")


def test_cuda_optimizations():
    """Test CUDA-specific optimizations if available."""
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping CUDA-specific tests")
        return

    print("\n🎯 Testing CUDA-specific optimizations...")

    # Test model on GPU
    model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 50257)).cuda()

    # Test different precision modes
    precision_modes = [
        {"name": "FP32", "dtype": torch.float32, "autocast": False},
        {"name": "BF16 autocast", "dtype": torch.float32, "autocast": True},
    ]

    for mode in precision_modes:
        try:
            print(f"🔧 Testing {mode['name']}...")

            model = model.to(dtype=mode["dtype"])
            x = torch.randn(4, 1024, device="cuda", dtype=mode["dtype"])

            if mode["autocast"]:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    y = model(x)
            else:
                y = model(x)

            print(f"✅ {mode['name']} works! Output shape: {y.shape}")

        except Exception as e:
            print(f"❌ {mode['name']} failed: {e}")


if __name__ == "__main__":
    test_optimizer_configurations()
    test_cuda_optimizations()
    print("\n🚀 All optimizer tests completed!")
