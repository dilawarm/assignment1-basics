#!/usr/bin/env python3
"""Test script for local .npy data loading."""

import os
import sys
import time
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.data import create_dataloaders


def test_local_data_loading():
    """Test loading data from local .npy files."""
    print("🧪 Testing local .npy data loading...")
    
    # Check if data files exist
    train_path = "training_data/owt_train_tokens.npy"
    val_path = "training_data/owt_valid_tokens.npy"
    
    if not os.path.exists(train_path):
        print(f"❌ Training data not found: {train_path}")
        return False
        
    if not os.path.exists(val_path):
        print(f"❌ Validation data not found: {val_path}")
        return False
    
    print(f"✅ Found training data: {train_path}")
    print(f"✅ Found validation data: {val_path}")
    
    # Create data loaders
    print("\n📂 Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_data_path=train_path,
        val_data_path=val_path,
        batch_size=4,
        max_length=1024,
        num_workers=2,
        prefetch_factor=2,
        seed=42,
    )
    
    # Test training data loader
    print("\n🚂 Testing training data loader...")
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        if i >= 5:  # Test first 5 batches
            break
            
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        print(f"  Batch {i+1}: input_ids shape = {input_ids.shape}, labels shape = {labels.shape}")
        print(f"    Sample tokens: {input_ids[0, :10].tolist()}")
        
        # Verify labels match input_ids
        assert torch.equal(input_ids, labels), "Labels should match input_ids for language modeling"
        
    train_time = time.time() - start_time
    print(f"✅ Training loader test completed in {train_time:.2f}s")
    
    # Test validation data loader
    print("\n🔍 Testing validation data loader...")
    start_time = time.time()
    
    for i, batch in enumerate(val_loader):
        if i >= 3:  # Test first 3 batches
            break
            
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        print(f"  Batch {i+1}: input_ids shape = {input_ids.shape}, labels shape = {labels.shape}")
        
    val_time = time.time() - start_time
    print(f"✅ Validation loader test completed in {val_time:.2f}s")
    
    # Test performance
    print("\n⚡ Testing data loading performance...")
    start_time = time.time()
    batch_count = 0
    token_count = 0
    
    for i, batch in enumerate(train_loader):
        if i >= 50:  # Test 50 batches
            break
            
        batch_count += 1
        token_count += batch["input_ids"].numel()
        
    elapsed = time.time() - start_time
    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
    
    print(f"📊 Performance:")
    print(f"  Batches processed: {batch_count}")
    print(f"  Total tokens: {token_count:,}")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"  Tokens/sec: {tokens_per_sec:,.0f}")
    
    return True


def main():
    """Main test function."""
    print("🚀 Testing local .npy data loading")
    print("=" * 60)
    
    success = test_local_data_loading()
    
    if success:
        print("\n✅ All tests passed!")
        print("\n📋 Ready for training with local data:")
        print("  python train_h100_optimized.py")
    else:
        print("\n❌ Tests failed!")
        print("  Make sure training_data/ contains owt_train_tokens.npy and owt_valid_tokens.npy")
    
    print("=" * 60)


if __name__ == "__main__":
    main()