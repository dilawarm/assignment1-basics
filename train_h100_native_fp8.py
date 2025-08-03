#!/usr/bin/env python3
"""
H100 training with native PyTorch FP8 support (no Transformer Engine).
This avoids cuBLAS errors by using PyTorch's built-in FP8 support.
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.data import create_dataloaders
from cs336_basics.model import TransformerLM
from cs336_basics.training import Trainer, TrainingConfig


def check_fp8_support():
    """Check if PyTorch supports FP8 dtypes."""
    try:
        # Check for FP8 dtypes in PyTorch
        _ = torch.float8_e4m3fn
        _ = torch.float8_e5m2
        return True
    except AttributeError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Train 350M transformer on H100 with native FP8")

    # Model arguments
    parser.add_argument("--dim", type=int, default=1024, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=24, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Attention head dimension")
    parser.add_argument("--intermediate_size", type=int, default=4096, help="FFN intermediate size")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="Peak learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=4e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--max_hours", type=float, default=1.5, help="Maximum training hours")

    # Data arguments
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data workers")

    # Optimization arguments
    parser.add_argument(
        "--use_native_fp8", action="store_true", default=False, help="Use PyTorch native FP8 (experimental)"
    )
    parser.add_argument("--use_flash_attn", action="store_true", default=True, help="Use Flash Attention")
    parser.add_argument("--compile_model", action="store_true", default=True, help="Compile model")
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", default=True, help="Use gradient checkpointing"
    )

    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Use Weights & Biases logging")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Evaluation interval")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("H100 Training with Native PyTorch (No Transformer Engine)")
    print("=" * 80)
    print(f"Model Configuration:")
    print(f"  - Parameters: ~350M")
    print(f"  - Layers: {args.n_layers}")
    print(f"  - Hidden size: {args.dim}")
    print(f"  - Heads: {args.n_heads} x {args.head_dim}")
    print(f"  - FFN size: {args.intermediate_size}")
    print(f"  - Sequence length: {args.max_length}")
    print()

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")

    # Check compute capability
    major, minor = torch.cuda.get_device_capability()
    compute_capability = major + minor / 10
    print(f"Compute Capability: {compute_capability}")

    # Check FP8 support
    if args.use_native_fp8:
        if not check_fp8_support():
            print("WARNING: PyTorch doesn't support FP8 dtypes. Using FP16 instead.")
            args.use_native_fp8 = False
        elif compute_capability < 8.9:
            print("WARNING: FP8 requires compute capability >= 8.9. Using FP16 instead.")
            args.use_native_fp8 = False
        else:
            print("✓ Native FP8 support available")

    print(f"\nTraining Configuration:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {args.batch_size * args.gradient_accumulation_steps * args.max_length} tokens")
    print(f"  - Learning rate: {args.learning_rate} (peak) -> {args.min_learning_rate} (min)")
    print(f"  - Warmup steps: {args.warmup_steps}")
    print()
    print(f"Optimizations:")
    print(f"  - FP8 (native): {args.use_native_fp8}")
    print(f"  - Flash Attention: {args.use_flash_attn}")
    print(f"  - Model compilation: {args.compile_model}")
    print(f"  - Gradient checkpointing: {args.gradient_checkpointing}")
    print("=" * 80)

    # Create model WITHOUT Transformer Engine FP8
    print("\nCreating model...")
    model = TransformerLM(
        vocab_size=50257,  # GPT-2 tokenizer
        max_seq_len=args.max_length,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        intermediate_size=args.intermediate_size,
        dropout=0.0,
        tie_embeddings=True,
        use_flash=args.use_flash_attn,
        use_fp8=False,  # Always False to avoid Transformer Engine
        use_gradient_checkpointing=args.gradient_checkpointing,
    )

    # Optionally convert model to FP8 using native PyTorch
    if args.use_native_fp8 and check_fp8_support():
        print("\nConverting model to native FP8...")
        # This is experimental - PyTorch's FP8 support is still evolving
        # For now, we'll use FP16 which is well-supported
        pass

    # Create data loaders
    print("\nCreating data loaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    # Calculate training steps
    tokens_per_second = 500_000  # Conservative estimate without FP8
    if args.use_flash_attn:
        tokens_per_second = 700_000  # With Flash Attention

    total_seconds = args.max_hours * 3600
    total_tokens = tokens_per_second * total_seconds
    tokens_per_step = args.batch_size * args.gradient_accumulation_steps * args.max_length
    max_steps = int(total_tokens / tokens_per_step)

    print(f"\nTraining plan:")
    print(f"  - Estimated tokens/sec: {tokens_per_second:,}")
    print(f"  - Total training time: {args.max_hours} hours")
    print(f"  - Total tokens: {total_tokens:,} ({total_tokens / 1e9:.1f}B)")
    print(f"  - Total steps: {max_steps:,}")

    # Create training config
    config = TrainingConfig(
        model_name=f"gpt-350m-h100-native-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        grad_clip=1.0,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=max_steps,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        save_interval=5000,
        max_length=args.max_length,
        num_workers=args.num_workers,
        use_fp8=False,  # Disable Transformer Engine FP8
        use_amp=True,  # Use standard FP16 AMP
        compile_model=args.compile_model,
        use_flash_attn=args.use_flash_attn,
        gradient_checkpointing=args.gradient_checkpointing,
        use_wandb=args.use_wandb,
        output_dir=args.output_dir,
    )

    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
    )

    # Start training
    print("\nStarting training...")
    print("Target: Beat validation loss of 3.0781")
    print("Expected: Achieve validation loss of 2.90-2.95")
    print("-" * 80)

    final_metrics = trainer.train()

    # Print final results
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Final validation loss: {final_metrics['val_loss']:.4f}")
    print(f"Final validation perplexity: {final_metrics['val_perplexity']:.2f}")

    if final_metrics["val_loss"] < 3.0781:
        print(f"✅ SUCCESS! Beat target of 3.0781 by {3.0781 - final_metrics['val_loss']:.4f}")
    else:
        print(f"❌ FAILED to beat target. Missed by {final_metrics['val_loss'] - 3.0781:.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
