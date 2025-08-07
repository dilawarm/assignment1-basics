#!/usr/bin/env python3
"""Safe test script for .npy dataloader that works even without proper CUDA setup."""

import os
import sys
import time

# Set environment variable for CUDA debugging if needed
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.data.optimized_dataloader import create_optimized_dataloaders


def test_npy_dataloader_safe():
    """Test the .npy dataloader functionality with safe fallbacks."""
    print("🧪 Testing .npy dataloader (safe mode)...")

    # Check if data files exist
    train_file = "training_data/owt_train_tokens.npy"
    val_file = "training_data/owt_valid_tokens.npy"

    if not os.path.exists(train_file):
        print(f"❌ Training file not found: {train_file}")
        return False
    if not os.path.exists(val_file):
        print(f"❌ Validation file not found: {val_file}")
        return False

    print(f"✅ Found data files!")

    # Try different configurations in order of preference
    configurations = [
        {"num_workers": 0, "pin_memory": False, "name": "Single-threaded CPU"},
        {"num_workers": 2, "pin_memory": False, "name": "Multi-threaded CPU"},
        {"num_workers": 0, "pin_memory": True, "name": "Single-threaded with pinned memory"},
        {"num_workers": 2, "pin_memory": True, "name": "Multi-threaded with pinned memory"},
    ]

    for config in configurations:
        try:
            print(f"\n🔧 Trying configuration: {config['name']}")

            train_loader, val_loader = create_optimized_dataloaders(
                batch_size=4,
                max_length=1024,
                num_workers=config["num_workers"],
                data_dir="training_data",
                seed=42,
                use_memmap=True,
                pin_memory=config["pin_memory"],
            )

            print("✅ Created dataloaders successfully!")

            # Test loading a few batches
            print("📚 Testing data loading...")
            start_time = time.time()

            batch_count = 0
            for i, batch in enumerate(train_loader):
                if i >= 3:  # Test first 3 batches
                    break

                input_ids = batch["input_ids"]
                labels = batch["labels"]

                # Verify data integrity
                assert input_ids.shape == labels.shape
                assert input_ids.shape[1] == 1024  # sequence length
                assert input_ids.dtype == torch.long
                assert torch.all(input_ids >= 0) and torch.all(input_ids < 50257)  # Valid token range

                batch_count += 1
                print(f"  Batch {i}: shape {input_ids.shape}, dtype {input_ids.dtype} ✓")

            load_time = time.time() - start_time
            tokens_loaded = batch_count * 4 * 1024
            tokens_per_sec = tokens_loaded / load_time if load_time > 0 else 0

            print(f"✅ Loaded {batch_count} batches successfully!")
            print(f"📊 Loading speed: {tokens_per_sec:,.0f} tokens/sec")

            # Test validation dataloader briefly
            print("📚 Testing validation dataloader...")
            val_batch = next(iter(val_loader))
            print(f"✅ Validation batch shape: {val_batch['input_ids'].shape}")

            print(f"🎉 Configuration '{config['name']}' works perfectly!")
            return True

        except Exception as e:
            print(f"❌ Configuration '{config['name']}' failed: {e}")
            continue

    print("❌ All configurations failed!")
    return False


if __name__ == "__main__":
    success = test_npy_dataloader_safe()
    if success:
        print("\n🎉 .npy dataloader is working! Ready for training!")
    else:
        print("\n❌ All dataloader tests failed!")
        print("💡 Suggestions:")
        print("   - Check if .npy files are valid")
        print("   - Try running with CUDA_VISIBLE_DEVICES='' (CPU only)")
        print("   - Check CUDA installation if using GPU")
        sys.exit(1)
