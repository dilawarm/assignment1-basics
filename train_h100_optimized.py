#!/usr/bin/env python3
"""
Ultra-optimized H100 training script for 350M parameter transformer.
Target: Beat validation loss of 3.0781 on OpenWebText in 1.5 hours.

Key optimizations:
- Memory-efficient batch sizing and gradient accumulation
- Selective mixed precision: BF16 for excellent stability and torch.compile() compatibility
- Aggressive hyperparameters for faster convergence
- Local pre-tokenized data for maximum I/O throughput
- Advanced CUDA optimizations with memory management
- Compatible with both FP8 (no compile) and BF16 (with compile) modes
"""

import argparse
import os
import sys
from datetime import datetime

import torch

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.data import create_dataloaders
from cs336_basics.model import TransformerLM, apply_selective_mixed_precision
from cs336_basics.training import H100OptimizedTrainer, TrainingConfig


def print_system_info():
    """Print system and GPU information."""
    print("=" * 80)
    print("🚀 H100-Optimized 350M Transformer Training")
    print("=" * 80)

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")

        if "H100" in gpu_name:
            print("✅ H100 detected - optimizations enabled")
        elif "A100" in gpu_name:
            print("⚠️  A100 detected - some optimizations may be suboptimal")
        else:
            print("⚠️  Non-H100 GPU detected - performance may be limited")
    else:
        print("❌ CUDA not available!")
        sys.exit(1)

    # PyTorch info
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    # Check for optimizations
    try:
        import torchao

        print("✅ TorchAO available for FP8")
    except ImportError:
        print("❌ TorchAO not available")

    try:
        from flash_attn import flash_attn_func

        print("✅ Flash Attention available")
    except ImportError:
        print("❌ Flash Attention not available")

    # Check BF16 support
    if torch.cuda.is_bf16_supported():
        print("✅ BF16 mixed precision supported")
    else:
        print("⚠️  BF16 not supported, falling back to FP16")

    print("=" * 80)


def calculate_memory_efficient_batch_size():
    """
    Calculate memory-efficient batch size for 80GB H100.

    Based on PyTorch memory optimization best practices:
    https://paulbridger.com/posts/pytorch-memory-tuning/
    """
    # Model memory estimation (350M parameters with BF16)
    model_params = 350e6

    # Memory breakdown for transformer models:
    # 1. Model parameters: 350M * 2 bytes = 0.7GB (BF16)
    # 2. Gradients: 350M * 2 bytes = 0.7GB (BF16)
    # 3. Optimizer states: 350M * 8 bytes = 2.8GB (Adam: momentum + variance)
    # 4. Activations: batch_size * seq_len * hidden_dim * layers * multiplier

    base_memory = 0.7 + 0.7 + 2.8  # Model + gradients + optimizer = 4.2GB

    # Available memory for activations (leave 10GB safety margin)
    available_memory = 80 - base_memory - 10  # ~65GB for activations

    # Activation memory per sequence with gradient checkpointing
    # With checkpointing: activations ≈ 2 * sqrt(n_layers) * hidden_dim * seq_len * 2 bytes
    n_layers = 24
    hidden_dim = 1024
    seq_len = 1024

    # Memory per sequence with gradient checkpointing (much more efficient)
    activation_memory_per_seq = 2 * (n_layers**0.5) * hidden_dim * seq_len * 2 / 1e9  # GB

    # Calculate max batch size
    max_batch_size = int(available_memory / activation_memory_per_seq)

    # Be conservative: use 60% of calculated max
    optimal_batch_size = max(8, int(max_batch_size * 0.6))

    # Cap at reasonable limit for stability
    optimal_batch_size = min(optimal_batch_size, 32)

    print(f"💾 Memory Analysis:")
    print(f"  Base memory (model + optimizer): {base_memory:.1f}GB")
    print(f"  Available for activations: {available_memory:.1f}GB")
    print(f"  Memory per sequence: {activation_memory_per_seq * 1000:.1f}MB")
    print(f"  Calculated max batch size: {max_batch_size}")
    print(f"  Conservative optimal: {optimal_batch_size}")

    return optimal_batch_size


def setup_memory_optimizations():
    """Setup memory optimization environment variables."""
    # Enable expandable segments to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

    # Enable memory pool for better allocation patterns
    torch.cuda.empty_cache()

    print("✅ Memory optimizations configured")


def main():
    parser = argparse.ArgumentParser(description="Memory-optimized H100 transformer training")

    # Model arguments
    parser.add_argument("--dim", type=int, default=1024, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=24, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Attention head dimension")
    parser.add_argument("--intermediate_size", type=int, default=4096, help="FFN intermediate size")

    # Training arguments - Memory-optimized defaults
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (auto-calculated if None)")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (increased for memory efficiency)",
    )
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=6e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--max_hours", type=float, default=1.5, help="Maximum training hours")

    # Data arguments
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="training_data/owt_train_tokens.npy",
        help="Path to training data .npy file",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="training_data/owt_valid_tokens.npy",
        help="Path to validation data .npy file",
    )
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data workers (reduced for memory)")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor (reduced for memory)")

    # Optimization arguments
    parser.add_argument("--use_bf16", action="store_true", default=True, help="Use BF16 mixed precision")
    parser.add_argument("--use_flash_attn", action="store_true", default=True, help="Use Flash Attention")
    parser.add_argument("--compile_model", action="store_true", default=True, help="Compile model")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing (essential for memory)",
    )

    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Use W&B logging")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_optimized", help="Output directory")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--log_interval", type=int, default=50, help="Logging interval")

    args = parser.parse_args()

    # Setup memory optimizations first
    setup_memory_optimizations()

    # Print system information
    print_system_info()

    # Check that data files exist
    if not os.path.exists(args.train_data_path):
        print(f"❌ Training data file not found: {args.train_data_path}")
        sys.exit(1)
    if not os.path.exists(args.val_data_path):
        print(f"❌ Validation data file not found: {args.val_data_path}")
        sys.exit(1)

    print(f"✅ Found training data: {args.train_data_path}")
    print(f"✅ Found validation data: {args.val_data_path}")

    # Calculate memory-efficient batch size if not provided
    if args.batch_size is None:
        args.batch_size = calculate_memory_efficient_batch_size()
        print(f"🧮 Auto-calculated memory-efficient batch size: {args.batch_size}")
    else:
        print(f"⚠️  Using manual batch size: {args.batch_size}")

    # Validate memory usage
    estimated_memory_gb = 4.2 + args.batch_size * 0.15  # Base + batch memory
    if estimated_memory_gb > 70:
        print(f"⚠️  WARNING: Estimated memory usage {estimated_memory_gb:.1f}GB may cause OOM!")
        print("Consider reducing batch size or enabling gradient checkpointing.")

    print(f"\n📊 Memory-Optimized Training Configuration:")
    print(f"  Model: 350M parameters ({args.n_layers} layers, {args.dim} hidden)")
    print(f"  Batch size: {args.batch_size} (memory-optimized)")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps * args.max_length:,} tokens")
    print(f"  Learning rate: {args.learning_rate} -> {args.min_learning_rate}")
    print(f"  Max training time: {args.max_hours} hours")
    print(f"  Mixed precision: {'BF16' if args.use_bf16 else 'FP32'}")
    print(f"  Model compilation: {'Enabled' if args.compile_model else 'Disabled'}")
    print(f"  Gradient checkpointing: {'Enabled' if args.gradient_checkpointing else 'Disabled'}")
    print(f"  Estimated memory usage: {estimated_memory_gb:.1f}GB")
    print()

    # Create model
    print("🏗️  Creating memory-optimized model...")
    model = TransformerLM(
        vocab_size=50257,
        max_seq_len=args.max_length,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        intermediate_size=args.intermediate_size,
        dropout=0.0,  # No dropout for maximum performance
        tie_embeddings=True,
        use_flash=args.use_flash_attn,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )

    # Apply selective mixed precision optimizations
    if args.use_bf16:
        try:
            # Pass compile flag to determine precision strategy
            model = apply_selective_mixed_precision(model, use_compile=args.compile_model)
        except Exception as e:
            print(f"⚠️  Failed to apply mixed precision optimizations: {e}")
            args.use_bf16 = False

    # Create memory-optimized data loaders
    print("📂 Creating memory-optimized data loaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=42,
    )

    # Calculate training steps
    estimated_tokens_per_sec = 300_000  # Conservative estimate with memory optimizations
    total_seconds = args.max_hours * 3600
    total_tokens = estimated_tokens_per_sec * total_seconds
    tokens_per_step = args.batch_size * args.gradient_accumulation_steps * args.max_length
    max_steps = int(total_tokens / tokens_per_step)

    print(f"\n📈 Training Plan:")
    print(f"  Estimated tokens/sec: {estimated_tokens_per_sec:,}")
    print(f"  Total training time: {args.max_hours} hours")
    print(f"  Target total tokens: {total_tokens:,} ({total_tokens / 1e9:.1f}B)")
    print(f"  Planned steps: {max_steps:,}")
    print(f"  Tokens per step: {tokens_per_step:,}")

    # Create memory-optimized training config
    config = TrainingConfig(
        model_name=f"gpt-350m-h100-memory-optimized-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
        save_interval=2000,
        max_length=args.max_length,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        use_mixed_precision=args.use_bf16,
        use_bf16=args.use_bf16,
        compile_model=args.compile_model,
        use_flash_attn=args.use_flash_attn,
        gradient_checkpointing=args.gradient_checkpointing,
        use_fused_adamw=True,
        use_wandb=args.use_wandb,
        output_dir=args.output_dir,
    )

    # Create memory-optimized trainer
    print("🎯 Creating H100 memory-optimized trainer...")
    trainer = H100OptimizedTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
    )

    # Print optimization status
    print(f"\n⚙️  Memory Optimizations:")
    print(f"  Mixed Precision: {'✅ BF16' if config.use_bf16 else '❌'}")
    print(f"  Flash Attention: {'✅' if config.use_flash_attn else '❌'}")
    print(f"  Model Compilation: {'✅' if config.compile_model else '❌'}")
    print(f"  Gradient Checkpointing: {'✅ ENABLED' if config.gradient_checkpointing else '❌ DISABLED'}")
    print(f"  Fused AdamW: {'✅' if config.use_fused_adamw else '❌'}")
    print(
        f"  Memory Allocation: {'✅ Optimized' if 'expandable_segments' in os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '') else '❌'}"
    )

    # Memory estimate
    model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"\n💾 Memory Breakdown:")
    print(f"  Model parameters: {model_size_gb:.1f} GB")
    print(f"  Estimated total usage: {estimated_memory_gb:.1f} GB")
    print(f"  Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Memory efficiency: {(estimated_memory_gb / 80) * 100:.1f}%")

    # Start training
    print("\n" + "🚀 Starting memory-optimized training...")
    print("🎯 Target: Beat validation loss of 3.0781")
    print("🔥 Expected: Achieve validation loss of 2.8-2.9")
    print("💾 Memory-optimized for stability and efficiency")
    print("-" * 80)

    try:
        # Clear any cached memory before training
        torch.cuda.empty_cache()

        final_metrics = trainer.train()

        # Print success/failure
        success = trainer.best_val_loss < 3.0781
        print(f"\n{'🎉' if success else '😞'} Training {'SUCCEEDED' if success else 'FAILED'}")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback

        traceback.print_exc()

        # Try to free memory for debugging
        torch.cuda.empty_cache()
        sys.exit(1)


if __name__ == "__main__":
    main()
