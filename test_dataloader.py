#!/usr/bin/env python3
"""Test script to verify dataloader with local numpy files."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.data import create_dataloaders


def test_dataloader():
    """Test dataloader functionality with local files."""
    print("Testing dataloader with local OpenWebText files...")

    # Check if data files exist
    train_path = "data/encoded/owt_train_tokens.npy"
    val_path = "data/encoded/owt_valid_tokens.npy"

    if not os.path.exists(train_path):
        print(f"Error: Training data not found at {train_path}")
        return

    if not os.path.exists(val_path):
        print(f"Error: Validation data not found at {val_path}")
        return

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        batch_size=4,
        max_length=128,  # Small for testing
        num_workers=2,
    )

    # Test train loader
    print("\nTesting train dataloader:")
    print(f"Number of batches: {len(train_loader)}")

    # Get first batch
    for i, batch in enumerate(train_loader):
        if i >= 3:  # Only test first 3 batches
            break

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        print(f"\nBatch {i + 1}:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Input dtype: {input_ids.dtype}")
        print(f"  Min token: {input_ids.min().item()}")
        print(f"  Max token: {input_ids.max().item()}")

        # Verify shapes
        assert input_ids.shape == labels.shape, "Input and label shapes must match"
        assert input_ids.shape[0] == 4, f"Batch size should be 4, got {input_ids.shape[0]}"
        assert input_ids.shape[1] == 128, f"Sequence length should be 128, got {input_ids.shape[1]}"

    # Test validation loader
    print("\n\nTesting validation dataloader:")
    print(f"Number of batches: {len(val_loader)}")

    # Get first batch
    batch = next(iter(val_loader))
    input_ids = batch["input_ids"]

    print(f"\nFirst validation batch:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Min token: {input_ids.min().item()}")
    print(f"  Max token: {input_ids.max().item()}")

    print("\n✅ Dataloader test passed!")


if __name__ == "__main__":
    test_dataloader()
