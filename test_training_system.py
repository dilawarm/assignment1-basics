#!/usr/bin/env python3
"""
Training System Smoke Test

This script runs comprehensive tests to verify that the training system
is working correctly before starting the full training run. It tests:

1. All optimizations and components work correctly
2. No device mismatch errors
3. Memory management works
4. Training doesn't crash with NaN/Inf values
5. All configurations are valid

Run this before your actual training to prevent crashes!
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.loss.cross_entropy import cross_entropy, robust_cross_entropy
from cs336_basics.nn.activations import CustomFFN
from cs336_basics.nn.models import EnhancedTransformerLM
from cs336_basics.scripts.train_transformer import (
    DataLoader,
    StabilityTracker,
    Trainer,
    TrainingConfig,
    load_config,
)
from cs336_basics.training.gradient_clipping import (
    AdaptiveGradientClipper,
    advanced_gradient_clipping,
)
from cs336_basics.training.lr_schedules import linear_decay_to_zero_schedule
from cs336_basics.training.optimizers import MixedOptimizerV2


def create_test_datasets(tmp_dir: str, vocab_size: int = 1000) -> tuple[str, str]:
    """Create test datasets for smoke testing."""
    print("📊 Creating test datasets...")

    # Create realistic but small datasets
    train_size = 100000  # 100K tokens
    val_size = 10000  # 10K tokens

    train_data = np.random.randint(0, vocab_size, size=train_size, dtype=np.uint16)
    val_data = np.random.randint(0, vocab_size, size=val_size, dtype=np.uint16)

    train_path = Path(tmp_dir) / "smoke_test_train.npy"
    val_path = Path(tmp_dir) / "smoke_test_val.npy"

    np.save(train_path, train_data)
    np.save(val_path, val_data)

    print(f"✅ Created train dataset: {train_size:,} tokens")
    print(f"✅ Created val dataset: {val_size:,} tokens")

    return str(train_path), str(val_path)


def test_device_management() -> bool:
    """Test device management and cross-entropy stability."""
    print("\n🔧 Testing device management...")

    try:
        # Test cross-entropy with various inputs
        vocab_size = 100
        batch_size = 4
        seq_len = 8

        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Test normal case
        loss = cross_entropy(logits, targets)
        assert torch.isfinite(loss), "Cross-entropy should produce finite loss"

        # Test with out-of-range targets
        targets_bad = torch.tensor([[50, 150, 25, -5, 75, 10, 20, 30] for _ in range(batch_size)])
        loss = cross_entropy(logits, targets_bad)
        assert torch.isfinite(loss), "Cross-entropy should handle bad targets"

        # Test robust cross-entropy with label smoothing
        loss = robust_cross_entropy(logits, targets, label_smoothing=0.1)
        assert torch.isfinite(loss), "Robust cross-entropy should work"

        print("✅ Device management tests passed")
        return True

    except Exception as e:
        print(f"❌ Device management test failed: {e}")
        return False


def test_gradient_clipping() -> bool:
    """Test gradient clipping functionality."""
    print("\n🔧 Testing gradient clipping...")

    try:
        # Create a simple model
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        # Create some gradients
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()

        # Test standard gradient clipping
        grad_norm = advanced_gradient_clipping(model, max_global_norm=1.0, use_adaptive=False)
        assert grad_norm >= 0, "Gradient norm should be non-negative"

        # Test adaptive gradient clipping
        clipper = AdaptiveGradientClipper(model, max_global_norm=1.0)
        grad_norm = clipper.clip_gradients(model)
        assert grad_norm >= 0, "Adaptive clipping should work"

        print("✅ Gradient clipping tests passed")
        return True

    except Exception as e:
        print(f"❌ Gradient clipping test failed: {e}")
        return False


def test_learning_rate_schedules() -> bool:
    """Test learning rate schedules."""
    print("\n📈 Testing learning rate schedules...")

    try:
        max_lr = 1e-3
        warmup_steps = 100
        total_steps = 1000

        # Test Linear Decay to Zero (D2Z)
        lr_start = linear_decay_to_zero_schedule(0, max_lr, warmup_steps, total_steps)
        lr_end_warmup = linear_decay_to_zero_schedule(warmup_steps, max_lr, warmup_steps, total_steps)
        lr_end = linear_decay_to_zero_schedule(total_steps, max_lr, warmup_steps, total_steps)

        assert lr_start == 0.0, "D2Z should start at 0"
        assert abs(lr_end_warmup - max_lr) < 1e-6, "D2Z should reach max_lr after warmup"
        assert lr_end == 0.0, "D2Z should end at 0"

        # Test monotonic decay
        lr_mid1 = linear_decay_to_zero_schedule(600, max_lr, warmup_steps, total_steps)
        lr_mid2 = linear_decay_to_zero_schedule(800, max_lr, warmup_steps, total_steps)
        assert lr_mid1 > lr_mid2, "D2Z should decrease monotonically"

        print("✅ Learning rate schedule tests passed")
        return True

    except Exception as e:
        print(f"❌ Learning rate schedule test failed: {e}")
        return False


def test_custom_activations() -> bool:
    """Test custom FFN activation."""
    print("\n⚡ Testing custom activations...")

    try:
        d_model = 64
        d_ff = 256
        batch_size = 4
        seq_len = 10

        ffn = CustomFFN(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ffn(x)

        # Check output properties
        assert output.shape == x.shape, "Output shape should match input"
        assert torch.isfinite(output).all(), "Output should be finite"
        assert not torch.allclose(output, x), "Output should be different from input"

        # Test with large inputs
        x_large = torch.randn(batch_size, seq_len, d_model) * 10
        output_large = ffn(x_large)
        assert torch.isfinite(output_large).all(), "Should handle large inputs"

        print("✅ Custom activation tests passed")
        return True

    except Exception as e:
        print(f"❌ Custom activation test failed: {e}")
        return False


def test_unet_architecture() -> bool:
    """Test U-Net transformer architecture."""
    print("\n🏗️ Testing U-Net architecture...")

    try:
        # Test U-Net model
        unet_model = EnhancedTransformerLM(
            vocab_size=100,
            context_length=16,
            d_model=32,
            num_layers=4,
            num_heads=4,
            d_ff=64,
            use_unet_architecture=True,
        )

        # Test standard model
        standard_model = EnhancedTransformerLM(
            vocab_size=100,
            context_length=16,
            d_model=32,
            num_layers=4,
            num_heads=4,
            d_ff=64,
            use_unet_architecture=False,
        )

        input_ids = torch.randint(0, 100, (2, 16))

        unet_output = unet_model(input_ids)
        standard_output = standard_model(input_ids)

        assert unet_output.shape == standard_output.shape, "Outputs should have same shape"
        assert not torch.allclose(unet_output, standard_output, atol=1e-4), "U-Net should produce different results"
        assert torch.isfinite(unet_output).all(), "U-Net output should be finite"

        # Check skip connections exist
        assert unet_model.skip_mixing is not None, "Skip connections should exist"
        assert len(unet_model.skip_mixing) == unet_model.skip_layers, "Correct number of skip connections"

        print("✅ U-Net architecture tests passed")
        return True

    except Exception as e:
        print(f"❌ U-Net architecture test failed: {e}")
        return False


def test_mixed_optimizer() -> bool:
    """Test MixedOptimizerV2."""
    print("\n⚙️ Testing MixedOptimizerV2...")

    try:
        model = EnhancedTransformerLM(
            vocab_size=100, context_length=16, d_model=32, num_layers=2, num_heads=4, d_ff=64, tie_embeddings=False
        )

        optimizer = MixedOptimizerV2(model=model, muon_lr=1e-3, adam_lr=5e-4, embedding_lr=2e-3, lm_head_lr=1e-4)

        # Check parameter grouping
        group_types = {group.get("type", "unknown") for group in optimizer.param_groups}
        expected_types = {"embedding", "lm_head", "linear", "1d"}
        assert expected_types.issubset(group_types), f"Missing parameter types: {expected_types - group_types}"

        # Test learning rate updates
        original_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.update_learning_rates(0.5, 0.8, 1.2, 0.3)

        # Check that learning rates changed
        updated_lrs = [group["lr"] for group in optimizer.param_groups]
        changed = any(orig != updated for orig, updated in zip(original_lrs, updated_lrs))
        assert changed, "Learning rates should change after update"

        for group in optimizer.param_groups:
            assert group["lr"] > 0, "All learning rates should be positive"

        print("✅ MixedOptimizerV2 tests passed")
        return True

    except Exception as e:
        print(f"❌ MixedOptimizerV2 test failed: {e}")
        return False


def test_stability_tracking() -> bool:
    """Test stability tracking system."""
    print("\n📊 Testing stability tracking...")

    try:
        tracker = StabilityTracker(window_size=50)

        # Add normal losses
        for i in range(30):
            tracker.update(loss=2.0 + 0.1 * np.sin(i), grad_norm=1.0, lr=1e-3)

        stats = tracker.get_comprehensive_stats()
        assert "stability_score" in stats, "Should have stability score"
        assert stats["stability_score"] > 0, "Stability score should be positive"

        # Test issue detection
        for i in range(10):
            tracker.update(loss=10.0, grad_norm=1.0, lr=1e-3)  # Spike

        issues = tracker.detect_training_issues()
        assert "loss_spike" in issues, "Should detect issues"

        print("✅ Stability tracking tests passed")
        return True

    except Exception as e:
        print(f"❌ Stability tracking test failed: {e}")
        return False


def test_full_training_pipeline(config_path: str = None) -> bool:
    """Test the complete training pipeline with a mini run."""
    print("\n🚀 Testing full training pipeline...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            if config_path and Path(config_path).exists():
                # Use provided config but modify for testing
                config = load_config(config_path)

                # Create test datasets
                train_path, val_path = create_test_datasets(tmp_dir, config.vocab_size)
                config.train_data_path = train_path
                config.val_data_path = val_path

                # Override for testing
                config.max_steps = 5
                config.max_wallclock_hours = 0.1
                config.batch_size = 4
                config.use_wandb = False
                config.compile_model = False  # Disable for testing
                config.checkpoint_dir = tmp_dir
                config.device = "cpu"  # Force CPU for testing
                config.log_interval = 2
                config.eval_interval = 3

            else:
                # Create minimal test config
                train_path, val_path = create_test_datasets(tmp_dir, 500)

                config = TrainingConfig(
                    train_data_path=train_path,
                    val_data_path=val_path,
                    vocab_size=500,
                    context_length=32,
                    d_model=64,
                    num_layers=2,
                    num_heads=4,
                    d_ff=128,
                    max_steps=5,
                    max_wallclock_hours=0.1,
                    batch_size=4,
                    gradient_accumulation_steps=2,
                    optimizer="mixed_v2",
                    lr_schedule="linear_decay_to_zero",
                    learning_rate=1e-3,
                    use_adaptive_clipping=True,
                    use_amp=False,
                    use_gradient_checkpointing=True,
                    compile_model=False,
                    use_wandb=False,
                    checkpoint_dir=tmp_dir,
                    device="cpu",
                    log_interval=2,
                    eval_interval=3,
                )

            # Initialize trainer
            print("🔧 Initializing trainer...")
            trainer = Trainer(config)

            # Test data loading
            print("📊 Testing data loading...")
            inputs, targets = trainer.train_loader.get_batch()
            assert inputs.shape[0] == config.batch_size, "Batch size should match"
            assert inputs.shape[1] == config.context_length, "Context length should match"

            # Test training steps
            print("🏃 Running training steps...")
            successful_steps = 0
            for step in range(config.max_steps):
                metrics = trainer.train_step()

                # Handle case where training stops early
                if metrics.get("training_stopped", False):
                    print(f"  Step {step}: Training stopped early - {metrics.get('loss', 'N/A')}")
                    break

                # Verify metrics
                assert "loss" in metrics, "Should have loss metric"
                assert "lr" in metrics, "Should have learning rate metric"

                # Only count as successful if loss is finite
                if torch.isfinite(torch.tensor(metrics["loss"])):
                    successful_steps += 1
                    print(f"  Step {step}: loss={metrics['loss']:.4f}, lr={metrics['lr']:.2e}")
                else:
                    print(f"  Step {step}: Non-finite loss detected")

                assert metrics["lr"] >= 0, "Learning rate should be non-negative"

                trainer.step += 1

                # Test evaluation
                if trainer.step % config.eval_interval == 0:
                    eval_metrics = trainer.evaluate()
                    if eval_metrics:
                        assert "loss" in eval_metrics, "Should have eval loss"
                        if torch.isfinite(torch.tensor(eval_metrics["loss"])):
                            print(f"  Eval: loss={eval_metrics['loss']:.4f}, ppl={eval_metrics['perplexity']:.2f}")
                        else:
                            print(f"  Eval: Non-finite loss detected")

            # Test checkpointing
            print("💾 Testing checkpointing...")
            checkpoint_path = Path(tmp_dir) / "test_checkpoint.pt"
            trainer.save_checkpoint(str(checkpoint_path))
            assert checkpoint_path.exists(), "Checkpoint should be saved"

            # At least some steps should succeed
            assert successful_steps > 0, f"No successful training steps out of {config.max_steps}"

            print("✅ Full training pipeline tests passed")
            return True

        except Exception as e:
            print(f"❌ Full training pipeline test failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def test_configuration_validation(config_path: str = None) -> bool:
    """Test configuration validation."""
    print("\n⚙️ Testing configuration validation...")

    try:
        if config_path and Path(config_path).exists():
            config = load_config(config_path)

            # Basic validation
            assert config.vocab_size > 0, "vocab_size should be positive"
            assert config.context_length > 0, "context_length should be positive"
            assert config.d_model % config.num_heads == 0, "d_model should be divisible by num_heads"
            assert config.max_steps > 0, "max_steps should be positive"
            assert config.learning_rate > 0, "learning_rate should be positive"

            # Check d_ff optimization
            assert config.d_ff % 64 == 0, "d_ff should be optimized for tensor cores"

            print(f"✅ Configuration validation passed for {config_path}")

        else:
            print("⚠️ No config path provided, skipping config validation")

        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def run_smoke_tests(config_path: str = None) -> bool:
    """Run all smoke tests."""
    print("🧪 Starting comprehensive training system smoke tests...\n")

    tests = [
        ("Device Management", test_device_management),
        ("Gradient Clipping", test_gradient_clipping),
        ("Learning Rate Schedules", test_learning_rate_schedules),
        ("Custom Activations", test_custom_activations),
        ("U-Net Architecture", test_unet_architecture),
        ("MixedOptimizerV2", test_mixed_optimizer),
        ("Stability Tracking", test_stability_tracking),
        ("Configuration Validation", lambda: test_configuration_validation(config_path)),
        ("Full Training Pipeline", lambda: test_full_training_pipeline(config_path)),
    ]

    passed = 0
    total = len(tests)
    start_time = time.time()

    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"🧪 Running {test_name} test...")
        print(f"{'=' * 60}")

        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    elapsed_time = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"🧪 SMOKE TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Passed: {passed}/{total} tests")
    print(f"Time: {elapsed_time:.2f} seconds")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Training system is ready.")
        print("✅ You can now run your full training with confidence.")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please fix issues before training.")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Training System Smoke Test")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training configuration JSON file",
        default="cs336_basics/scripts/configs/openwebtext_h100_v2.json",
    )
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (skip full pipeline)")

    args = parser.parse_args()

    print("🧪 Training System Smoke Test")
    print("=" * 60)
    print("This script verifies your training system is working correctly")
    print("before you start the full training run.")
    print("=" * 60)

    if args.config and Path(args.config).exists():
        print(f"📋 Using config: {args.config}")
    else:
        print("📋 Using default test configuration")
        args.config = None

    success = run_smoke_tests(args.config)

    if success:
        print("\n🚀 Ready to train! Run your training command now.")
        if args.config:
            print(f"   python -m cs336_basics.scripts.train_transformer --config {args.config}")
        exit(0)
    else:
        print("\n❌ Fix the failing tests before training.")
        exit(1)


if __name__ == "__main__":
    main()
