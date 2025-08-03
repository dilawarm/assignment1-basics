#!/usr/bin/env python3
"""
Fix FP8 issues on H100 by patching Transformer Engine initialization.
Run this before training if you encounter cuBLAS errors.
"""

import os
import sys

import torch


def patch_transformer_engine():
    """Apply patches to Transformer Engine for H100 compatibility."""
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe

        print("Patching Transformer Engine for H100...")

        # Monkey patch to fix initialization issues
        original_linear_init = te.Linear.__init__

        def patched_linear_init(self, in_features, out_features, bias=True, **kwargs):
            # Remove problematic kwargs
            safe_kwargs = {}
            for k, v in kwargs.items():
                if k not in ["params_dtype", "device", "init_method"]:
                    safe_kwargs[k] = v

            # Call original with safe kwargs
            original_linear_init(self, in_features, out_features, bias=bias, **safe_kwargs)

            # Ensure weights are properly initialized
            if hasattr(self, "weight") and self.weight is not None:
                with torch.no_grad():
                    if self.weight.dim() == 2:
                        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)

        te.Linear.__init__ = patched_linear_init
        print("✓ Transformer Engine patched successfully")
        return True

    except Exception as e:
        print(f"Failed to patch Transformer Engine: {e}")
        return False


def verify_h100_setup():
    """Verify H100 is properly set up."""
    print("\n" + "=" * 80)
    print("H100 Setup Verification")
    print("=" * 80)

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)

    print(f"✓ GPU: {gpu_name}")
    print(f"✓ Compute Capability: {major}.{minor}")

    if major < 9:
        print(f"❌ H100 requires compute capability 9.0+, found {major}.{minor}")
        return False

    # Check CUDA version
    print(f"✓ CUDA Runtime: {torch.version.cuda}")

    # Check cuDNN
    print(f"✓ cuDNN: {torch.backends.cudnn.version()}")

    # Test Transformer Engine
    try:
        import transformer_engine

        print(f"✓ Transformer Engine: {transformer_engine.__version__}")

        # Test FP8 support
        from transformer_engine.common import recipe

        fp8_recipe = recipe.DelayedScaling(
            fp8_format=recipe.Format.E4M3,
            amax_history_len=1,
            amax_compute_algo="most_recent",
        )
        print("✓ FP8 recipe created successfully")

    except Exception as e:
        print(f"❌ Transformer Engine error: {e}")
        return False

    print("\n✓ All checks passed!")
    return True


def main():
    print("H100 FP8 Fix Utility")
    print("=" * 80)

    # Set environment variables
    env_vars = {
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NVTE_FLASH_ATTN": "0",
        "NVTE_FUSED_ATTN": "0",
        "NVTE_BIAS_GELU_NVFUSION": "0",
        "NVTE_MASKED_SOFTMAX_FUSION": "0",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }

    print("\nSetting environment variables...")
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")

    # Apply patches
    if not patch_transformer_engine():
        print("\n⚠️  Failed to patch Transformer Engine")
        print("You may need to use --no_fp8 flag")

    # Verify setup
    if verify_h100_setup():
        print("\n✅ H100 is ready for FP8 training!")
        print("\nRecommended command:")
        print("python train_h100_fp8.py")
    else:
        print("\n❌ Setup verification failed")
        print("\nFallback command (without FP8):")
        print("python train_h100.py --no_fp8")


if __name__ == "__main__":
    main()
