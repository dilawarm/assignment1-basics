#!/usr/bin/env python3
"""Estimate training performance and time requirements."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.data import OpenWebTextDataset


def estimate_performance():
    """Estimate training performance for H100."""
    print("=" * 80)
    print("H100 Training Performance Estimation")
    print("=" * 80)

    # Load dataset info
    print("\nDataset Information:")
    train_data = OpenWebTextDataset("data/encoded/owt_train_tokens.npy", max_length=1024)
    val_data = OpenWebTextDataset("data/encoded/owt_valid_tokens.npy", max_length=1024)

    print(f"\nTraining data:")
    print(f"  Total tokens: {train_data.total_tokens:,}")
    print(f"  Sequences (1024 tokens): {train_data.num_sequences:,}")

    print(f"\nValidation data:")
    print(f"  Total tokens: {val_data.total_tokens:,}")
    print(f"  Sequences (1024 tokens): {val_data.num_sequences:,}")

    # Training configuration
    batch_size = 8
    gradient_accumulation = 16
    seq_length = 1024

    effective_batch_size = batch_size * gradient_accumulation * seq_length
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Effective batch size: {effective_batch_size:,} tokens")

    # Performance estimates
    print(f"\nPerformance Estimates (H100 with FP8):")

    # Conservative estimate
    tokens_per_sec_conservative = 700_000
    print(f"\nConservative estimate: {tokens_per_sec_conservative:,} tokens/sec")
    steps_per_sec = tokens_per_sec_conservative / effective_batch_size
    print(f"  Steps per second: {steps_per_sec:.1f}")
    print(f"  Minutes per 1000 steps: {1000 / steps_per_sec / 60:.1f}")

    # Expected estimate
    tokens_per_sec_expected = 900_000
    print(f"\nExpected estimate: {tokens_per_sec_expected:,} tokens/sec")
    steps_per_sec = tokens_per_sec_expected / effective_batch_size
    print(f"  Steps per second: {steps_per_sec:.1f}")
    print(f"  Minutes per 1000 steps: {1000 / steps_per_sec / 60:.1f}")

    # Optimistic estimate
    tokens_per_sec_optimistic = 1_100_000
    print(f"\nOptimistic estimate: {tokens_per_sec_optimistic:,} tokens/sec")
    steps_per_sec = tokens_per_sec_optimistic / effective_batch_size
    print(f"  Steps per second: {steps_per_sec:.1f}")
    print(f"  Minutes per 1000 steps: {1000 / steps_per_sec / 60:.1f}")

    # Training time calculations
    print(f"\nTraining Time Calculations (1.5 hours = 5400 seconds):")

    for name, tokens_per_sec in [
        ("Conservative", tokens_per_sec_conservative),
        ("Expected", tokens_per_sec_expected),
        ("Optimistic", tokens_per_sec_optimistic),
    ]:
        total_tokens = tokens_per_sec * 5400
        total_steps = total_tokens / effective_batch_size
        epochs = total_tokens / train_data.total_tokens

        print(f"\n{name} ({tokens_per_sec:,} tokens/sec):")
        print(f"  Total tokens in 1.5h: {total_tokens:,} ({total_tokens / 1e9:.1f}B)")
        print(f"  Total steps: {int(total_steps):,}")
        print(f"  Training epochs: {epochs:.2f}")
        print(f"  Tokens seen per parameter: {total_tokens / 350e6:.1f}")

    # Memory estimates
    print(f"\n" + "=" * 80)
    print("Memory Estimates (H100 80GB):")
    print(f"  Model parameters: ~1.4GB (FP32)")
    print(f"  Optimizer states: ~5.6GB (AdamW with FP32 moments)")
    print(f"  Activations (with gradient checkpointing): ~8-10GB")
    print(f"  Gradients: ~1.4GB")
    print(f"  Total estimated: ~16-20GB")
    print(f"  Available headroom: ~60-64GB")
    print(f"\nWith FP8:")
    print(f"  50% reduction in activation memory")
    print(f"  Total estimated with FP8: ~12-15GB")
    print(f"  Could potentially increase batch size to 16-32")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    estimate_performance()
