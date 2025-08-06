#!/usr/bin/env python3
"""Test script for local OpenWebText data loading."""

import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.data.local_dataloader import LocalOpenWebTextDataset, create_local_dataloaders, preprocess_and_save


def test_file_existence():
    """Test if data files exist."""
    print("🧪 Testing file existence...")

    train_file = "training_data/owt_train.txt"
    val_file = "training_data/owt_valid.txt"

    if os.path.exists(train_file):
        size = os.path.getsize(train_file) / 1e9
        print(f"✅ Training file found: {train_file} ({size:.1f}GB)")
    else:
        print(f"❌ Training file not found: {train_file}")
        return False

    if os.path.exists(val_file):
        size = os.path.getsize(val_file) / 1e9
        print(f"✅ Validation file found: {val_file} ({size:.1f}GB)")
    else:
        print(f"❌ Validation file not found: {val_file}")
        return False

    return True


def test_single_dataset():
    """Test loading a single dataset."""
    print("\n🧪 Testing single dataset loading...")

    try:
        dataset = LocalOpenWebTextDataset(
            file_path="training_data/owt_valid.txt",  # Use smaller validation file
            max_length=512,  # Smaller for testing
            buffer_size=10000,
        )

        print("✅ Dataset created successfully")

        # Test iteration
        count = 0
        start_time = time.time()

        for i, sample in enumerate(dataset):
            if i >= 5:  # Just test first 5 samples
                break

            input_ids = sample["input_ids"]
            labels = sample["labels"]

            print(f"Sample {i}: input_ids shape {input_ids.shape}, labels shape {labels.shape}")

            # Verify they're the same
            assert torch.equal(input_ids, labels), "input_ids and labels should be identical"

            # Verify length
            assert len(input_ids) == 512, f"Expected length 512, got {len(input_ids)}"

            count += 1

        elapsed = time.time() - start_time
        print(f"✅ Processed {count} samples in {elapsed:.2f}s")

    except Exception as e:
        print(f"❌ Single dataset test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_dataloader():
    """Test DataLoader functionality."""
    print("\n🧪 Testing DataLoader...")

    try:
        train_loader, val_loader = create_local_dataloaders(
            train_file="training_data/owt_train.txt",
            val_file="training_data/owt_valid.txt",
            batch_size=4,
            max_length=512,
            num_workers=2,  # Reduced for testing
            prefetch_factor=2,
        )

        print("✅ DataLoaders created successfully")

        # Test training loader
        print("Testing training loader...")
        train_iter = iter(train_loader)

        for i in range(3):
            batch = next(train_iter)
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            print(f"Training batch {i}: input_ids {input_ids.shape}, labels {labels.shape}")

            # Verify batch properties
            assert input_ids.shape[0] == 4, f"Expected batch size 4, got {input_ids.shape[0]}"
            assert input_ids.shape[1] == 512, f"Expected seq length 512, got {input_ids.shape[1]}"
            assert torch.equal(input_ids, labels), "input_ids and labels should be identical"

        # Test validation loader
        print("Testing validation loader...")
        val_iter = iter(val_loader)

        for i in range(3):
            batch = next(val_iter)
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            print(f"Validation batch {i}: input_ids {input_ids.shape}, labels {labels.shape}")

            # Verify batch properties
            assert input_ids.shape[0] == 4, f"Expected batch size 4, got {input_ids.shape[0]}"
            assert input_ids.shape[1] == 512, f"Expected seq length 512, got {input_ids.shape[1]}"
            assert torch.equal(input_ids, labels), "input_ids and labels should be identical"

        print("✅ DataLoader test passed")

    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_tokenization_quality():
    """Test tokenization quality."""
    print("\n🧪 Testing tokenization quality...")

    try:
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # Test sample
        test_text = "Hello world! This is a test of the tokenization process."

        # Manual tokenization
        manual_tokens = tokenizer(test_text, return_tensors="pt")["input_ids"].squeeze(0)

        print(f"Original text: {test_text}")
        print(f"Tokenized: {manual_tokens}")
        print(f"Decoded: {tokenizer.decode(manual_tokens)}")

        # Test with dataset
        dataset = LocalOpenWebTextDataset(
            file_path="training_data/owt_valid.txt",
            max_length=128,
            buffer_size=1000,
        )

        # Get a sample and decode it
        sample_iter = iter(dataset)
        sample = next(sample_iter)

        decoded_text = tokenizer.decode(sample["input_ids"])
        print(f"\nSample from dataset:")
        print(f"Token shape: {sample['input_ids'].shape}")
        print(f"First 100 chars: {decoded_text[:100]}...")

        print("✅ Tokenization quality test passed")

    except Exception as e:
        print(f"❌ Tokenization quality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_performance():
    """Test loading performance."""
    print("\n🧪 Testing loading performance...")

    try:
        train_loader, val_loader = create_local_dataloaders(
            train_file="training_data/owt_train.txt",
            val_file="training_data/owt_valid.txt",
            batch_size=8,
            max_length=1024,
            num_workers=4,
            prefetch_factor=2,
        )

        # Time batch loading
        start_time = time.time()
        batch_count = 0
        token_count = 0

        train_iter = iter(train_loader)

        for i in range(20):  # Test 20 batches
            batch = next(train_iter)
            batch_tokens = batch["input_ids"].numel()
            token_count += batch_tokens
            batch_count += 1

            if i % 5 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    tokens_per_sec = token_count / elapsed
                    print(f"Batch {i}: {tokens_per_sec:.0f} tokens/sec")

        elapsed = time.time() - start_time
        tokens_per_sec = token_count / elapsed if elapsed > 0 else 0

        print(f"✅ Performance test complete:")
        print(f"  Processed {batch_count} batches in {elapsed:.2f}s")
        print(f"  Total tokens: {token_count:,}")
        print(f"  Tokens/sec: {tokens_per_sec:.0f}")

        if tokens_per_sec > 100_000:
            print("✅ Good performance!")
        elif tokens_per_sec > 50_000:
            print("⚠️  Moderate performance")
        else:
            print("❌ Poor performance - check data loading bottlenecks")

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_preprocessing():
    """Test preprocessing functionality."""
    print("\n🧪 Testing preprocessing (optional)...")

    # Create a small test file
    test_file = "test_sample.txt"
    test_output = "test_sample_preprocessed.npy"

    try:
        # Create test data
        with open(test_file, "w") as f:
            f.write("This is a test line.\n")
            f.write("Another test line with more content.\n")
            f.write("Final test line for preprocessing.\n")

        # Test preprocessing
        preprocess_and_save(
            input_file=test_file,
            output_file=test_output,
            max_length=64,
            chunk_size=10,
        )

        print(f"✅ Preprocessing test completed")

        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists(test_output):
            os.remove(test_output)

    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")

        # Cleanup on failure
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists(test_output):
            os.remove(test_output)

        return False

    return True


def main():
    """Run all tests."""
    print("🚀 Testing Local OpenWebText Data Loading")
    print("=" * 60)

    tests = [
        ("File Existence", test_file_existence),
        ("Single Dataset", test_single_dataset),
        ("DataLoader", test_dataloader),
        ("Tokenization Quality", test_tokenization_quality),
        ("Performance", test_performance),
        ("Preprocessing", test_preprocessing),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Ready for training.")
    else:
        print("⚠️  Some tests failed. Check your setup.")

    print("\n📋 Next steps:")
    print("  1. Run: python train_h100_local.py")
    print("  2. Monitor training logs for performance metrics")
    print("  3. Expect significant performance improvements")


if __name__ == "__main__":
    main()
