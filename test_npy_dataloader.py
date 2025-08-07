#!/usr/bin/env python3
"""Test script to verify .npy dataloader works correctly."""

import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.data.optimized_dataloader import create_optimized_dataloaders


def test_npy_dataloader():
    """Test the .npy dataloader functionality."""
    print("🧪 Testing .npy dataloader...")

    # Initialize CUDA if available
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            print(f"🎯 CUDA initialized: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e:
            print(f"⚠️  CUDA initialization failed: {e}")
            print("📋 Testing will continue with CPU only")
    else:
        print("📋 CUDA not available, testing with CPU only")

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

    # Create dataloaders with appropriate settings
    try:
        # Detect CUDA capabilities and adjust settings accordingly
        use_pinned_memory = False
        num_workers = 2

        if torch.cuda.is_available():
            try:
                # Test if we can actually use CUDA and pinned memory
                test_tensor = torch.tensor([1.0])
                test_tensor.pin_memory()
                # Also test GPU operations
                if torch.cuda.device_count() > 0:
                    test_gpu = test_tensor.cuda()
                    del test_gpu
                del test_tensor
                use_pinned_memory = True
                print("✅ CUDA operations working, enabling pinned memory")
            except RuntimeError as e:
                use_pinned_memory = False
                num_workers = 0  # Disable multiprocessing if CUDA has issues
                print(f"⚠️  CUDA issues detected ({e}), disabling pinned memory and workers")

        train_loader, val_loader = create_optimized_dataloaders(
            batch_size=4,
            max_length=1024,
            num_workers=num_workers,  # Dynamically set based on CUDA
            data_dir="training_data",
            seed=42,
            use_memmap=True,
            pin_memory=use_pinned_memory,
        )
        print("✅ Created dataloaders successfully!")

    except Exception as e:
        print(f"❌ Failed to create dataloaders: {e}")
        return False

    # Test training dataloader
    print("\n📚 Testing training dataloader...")
    start_time = time.time()

    for i, batch in enumerate(train_loader):
        if i >= 5:  # Test first 5 batches
            break

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        print(f"Batch {i}: input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}")
        print(f"         labels shape: {labels.shape}, dtype: {labels.dtype}")

        # Verify data integrity
        assert input_ids.shape == labels.shape
        assert input_ids.shape[1] == 1024  # sequence length
        assert input_ids.dtype == torch.long
        assert torch.all(input_ids >= 0) and torch.all(input_ids < 50257)  # Valid token range

    load_time = time.time() - start_time
    tokens_loaded = 5 * 4 * 1024  # 5 batches * 4 batch_size * 1024 seq_len
    tokens_per_sec = tokens_loaded / load_time

    print(f"✅ Training dataloader test passed!")
    print(f"📊 Loading speed: {tokens_per_sec:,.0f} tokens/sec")

    # Test validation dataloader
    print("\n📚 Testing validation dataloader...")

    for i, batch in enumerate(val_loader):
        if i >= 2:  # Test first 2 batches
            break

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        print(f"Val Batch {i}: input_ids shape: {input_ids.shape}")

        # Verify data integrity
        assert input_ids.shape == labels.shape
        assert input_ids.shape[1] == 1024
        assert input_ids.dtype == torch.long

    print(f"✅ Validation dataloader test passed!")

    # Test memory pinning if CUDA available
    if torch.cuda.is_available():
        print("\n🔧 Testing CUDA memory pinning...")

        batch = next(iter(train_loader))
        input_ids = batch["input_ids"]

        if input_ids.is_pinned():
            print("✅ Memory pinning working!")
        else:
            print("⚠️  Memory pinning not working")

        # Test GPU transfer
        start_time = time.time()
        input_ids_gpu = input_ids.to("cuda", non_blocking=True)
        transfer_time = time.time() - start_time

        print(f"📊 GPU transfer time: {transfer_time * 1000:.2f} ms for {input_ids.numel():,} elements")

    print("\n✅ All tests passed! .npy dataloader is working correctly.")
    return True


if __name__ == "__main__":
    success = test_npy_dataloader()
    if success:
        print("\n🎉 Ready for optimized H100 training!")
    else:
        print("\n❌ Dataloader tests failed!")
        sys.exit(1)
