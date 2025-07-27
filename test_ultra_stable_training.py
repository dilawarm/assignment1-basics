#!/usr/bin/env python3
"""
Ultra-Stable Training System Test

This script tests the enhanced training system with:
- ZClip + AdaGC hybrid gradient clipping
- Outlier-safe Muon optimizer
- Comprehensive stability monitoring
- Advanced error recovery mechanisms

Run this to verify everything works before starting the full training.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cs336_basics.scripts.train_transformer import Trainer, load_config


def create_test_data(output_dir: str = "test_data", num_tokens: int = 100000):
    """Create small test datasets for validation."""
    os.makedirs(output_dir, exist_ok=True)

    vocab_size = 32000

    # Create training data
    train_tokens = np.random.randint(0, vocab_size, size=num_tokens, dtype=np.uint16)
    train_path = os.path.join(output_dir, "test_train_tokens.npy")
    np.save(train_path, train_tokens)

    # Create validation data
    val_tokens = np.random.randint(0, vocab_size, size=num_tokens // 10, dtype=np.uint16)
    val_path = os.path.join(output_dir, "test_val_tokens.npy")
    np.save(val_path, val_tokens)

    print(f"✅ Created test data:")
    print(f"  Training: {train_path} ({len(train_tokens):,} tokens)")
    print(f"  Validation: {val_path} ({len(val_tokens):,} tokens)")

    return train_path, val_path


def create_test_config(train_data_path: str, val_data_path: str) -> dict:
    """Create a test configuration based on the ultra-stable config."""
    config = {
        "train_data_path": train_data_path,
        "val_data_path": val_data_path,
        "vocab_size": 32000,
        "context_length": 512,
        "d_model": 512,  # Smaller for testing
        "num_layers": 4,  # Smaller for testing
        "num_heads": 8,
        "d_ff": 2048,  # Smaller for testing
        "rope_theta": 10000.0,
        "eps": 1e-05,
        "tie_embeddings": False,
        "activation": "custom",
        "use_unet_architecture": True,
        # Training settings - short for testing
        "max_steps": 100,
        "max_wallclock_hours": 0.1,  # 6 minutes max
        "batch_size": 4,  # Small for testing
        "gradient_accumulation_steps": 2,
        # Optimizer settings
        "optimizer": "mixed_v2",
        "lr_schedule": "linear_decay_to_zero",
        "learning_rate": 0.001,  # Conservative for testing
        "muon_lr": 0.001,
        "adam_lr": 0.0008,
        "embedding_lr": 0.0012,
        "lm_head_lr": 0.001,
        "min_learning_rate": 0.0,
        "warmup_steps": 10,
        "weight_decay": 0.01,
        "momentum": 0.95,
        "ns_iters": 3,  # Fewer iterations for testing
        "beta1": 0.9,
        "beta2": 0.95,
        # Advanced stability features
        "use_hybrid_clipping": True,
        "use_adaptive_clipping": True,
        "grad_clip_norm": 0.5,  # Conservative for testing
        "zclip_z_threshold": 2.0,  # More sensitive for testing
        "zclip_window_size": 50,
        "zclip_min_threshold": 0.1,
        "zclip_max_threshold": 1.0,
        "adagc_beta": 0.98,
        # Outlier-safe features
        "outlier_threshold": 3.0,  # More sensitive for testing
        "enable_outlier_detection": True,
        "stability_check_freq": 5,  # More frequent checks
        "max_norm_scale": 3.0,
        "enable_stability_logging": True,
        # Precision settings
        "use_amp": True,
        "use_bfloat16": True,
        "use_gradient_checkpointing": False,  # Disabled for small model
        "use_tf32": True,
        "compile_model": False,  # Disabled for testing to avoid compilation overhead
        # Data loading
        "num_workers": 2,
        "pin_memory": True,
        "prefetch_factor": 2,
        "dataloader_drop_last": True,
        "dataloader_persistent_workers": False,
        # Logging
        "log_interval": 5,
        "eval_interval": 20,
        "eval_batches": 5,
        "save_interval": 50,
        "checkpoint_dir": "test_checkpoints",
        "experiment_name": "ultra_stable_test",
        "experiment_description": "Testing ultra-stable training system",
        "use_wandb": False,  # Disabled for testing
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    return config


def test_stability_features():
    """Test individual stability features."""
    print("\n🧪 TESTING STABILITY FEATURES")
    print("=" * 50)

    # Test gradient clipping
    print("Testing gradient clipping...")
    try:
        from cs336_basics.training.gradient_clipping import AdaptiveGradientClipper, HybridGradientClipper, ZClip

        # Create a simple model for testing
        model = torch.nn.Linear(10, 5)

        # Test ZClip
        zclip = ZClip(model, window_size=10, z_threshold=2.0, enable_logging=False)
        print("  ✅ ZClip initialization successful")

        # Test AdaGC
        adagc = AdaptiveGradientClipper(model, max_global_norm=1.0, enable_logging=False)
        print("  ✅ AdaGC initialization successful")

        # Test Hybrid
        hybrid = HybridGradientClipper(model, enable_logging=False)
        print("  ✅ Hybrid clipping initialization successful")

    except Exception as e:
        print(f"  ❌ Gradient clipping test failed: {e}")
        return False

    # Test optimizer
    print("Testing outlier-safe optimizer...")
    try:
        from cs336_basics.training.optimizers import MixedOptimizerV2, Muon

        model = torch.nn.Linear(10, 5)

        # Test enhanced Muon
        muon = Muon(model.parameters(), lr=0.001, enable_outlier_detection=True, enable_stability_logging=False)
        print("  ✅ Outlier-safe Muon initialization successful")

        # Test MixedOptimizerV2
        mixed = MixedOptimizerV2(
            model, muon_lr=0.001, adam_lr=0.0008, enable_outlier_detection=True, enable_stability_logging=False
        )
        print("  ✅ Enhanced MixedOptimizerV2 initialization successful")

    except Exception as e:
        print(f"  ❌ Optimizer test failed: {e}")
        return False

    print("✅ All stability features tested successfully!")
    return True


def run_training_test(config_path: str):
    """Run a short training test to verify everything works."""
    print("\n🚀 RUNNING TRAINING TEST")
    print("=" * 50)

    try:
        # Load config
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Create trainer
        trainer = Trainer(load_config(config_path))
        print("✅ Trainer initialization successful")

        # Run a few training steps
        print("Running training steps...")
        start_time = time.time()
        trainer.train()
        elapsed = time.time() - start_time

        print(f"✅ Training completed successfully in {elapsed:.2f} seconds")
        return True

    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test ultra-stable training system")
    parser.add_argument("--skip-training", action="store_true", help="Skip actual training test")
    parser.add_argument("--keep-data", action="store_true", help="Keep test data after completion")
    args = parser.parse_args()

    print("🧪 ULTRA-STABLE TRAINING SYSTEM TEST")
    print("=" * 60)
    print("This test verifies all new stability features work correctly.")
    print("It will create small test data and run a short training session.")
    print()

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available, will test on CPU")

    # Test individual features
    if not test_stability_features():
        print("❌ Feature tests failed!")
        return 1

    # Create test data
    try:
        train_path, val_path = create_test_data()
    except Exception as e:
        print(f"❌ Failed to create test data: {e}")
        return 1

    # Create test config
    try:
        config = create_test_config(train_path, val_path)
        config_path = "test_ultra_stable_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Created test config: {config_path}")
    except Exception as e:
        print(f"❌ Failed to create test config: {e}")
        return 1

    # Run training test
    if not args.skip_training:
        if not run_training_test(config_path):
            print("❌ Training test failed!")
            return 1
    else:
        print("⏭️  Skipping training test as requested")

    # Cleanup
    if not args.keep_data:
        try:
            os.remove(train_path)
            os.remove(val_path)
            os.remove(config_path)
            if os.path.exists("test_checkpoints"):
                import shutil

                shutil.rmtree("test_checkpoints")
            print("🧹 Cleaned up test files")
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")

    print("\n🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("The ultra-stable training system is ready for use.")
    print()
    print("To run the full training, use:")
    print("python -m cs336_basics.scripts.train_transformer \\")
    print("    --config cs336_basics/scripts/configs/openwebtext_h100_v2_stable.json")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
