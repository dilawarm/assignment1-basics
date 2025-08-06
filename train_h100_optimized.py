#!/usr/bin/env python3
"""
Ultra-optimized H100 training script for 350M parameter transformer.
Target: Beat validation loss of 3.0781 on OpenWebText in 1.5 hours.

Key optimizations:
- Larger batch sizes to utilize 80GB H100 memory
- TorchAO FP8 for 2x speedup
- Aggressive hyperparameters for faster convergence
- Streaming data loading to prevent I/O bottlenecks
- Advanced CUDA optimizations
"""

import argparse
import os
import sys
from datetime import datetime

import torch

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.data import create_dataloaders
from cs336_basics.model import TransformerLM, apply_torchao_optimizations
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

    print("=" * 80)


def calculate_optimal_batch_size():
    """Calculate optimal batch size for 80GB H100."""
    # Model memory estimation (350M parameters)
    # - Model weights: 350M * 4 bytes = 1.4GB (FP32)
    # - With FP8: ~0.7GB
    # - Activations: batch_size * seq_len * hidden_dim * layers * 4
    # - With gradient checkpointing: much less activation memory

    # Conservative estimate for 80GB H100
    # Leave ~20GB for system/other processes
    available_memory = 60  # GB

    # With FP8 + gradient checkpointing, we can fit much larger batches
    # Each sequence (1024 tokens) uses roughly 100MB with all optimizations
    estimated_memory_per_batch = 0.1  # GB per sequence

    max_batch_size = int(available_memory / estimated_memory_per_batch)

    # Be conservative and use 80% of calculated max
    optimal_batch_size = max(16, int(max_batch_size * 0.8))

    return min(optimal_batch_size, 64)  # Cap at 64 for stability


def main():
    parser = argparse.ArgumentParser(description="H100-optimized 350M transformer training")

    # Model arguments
    parser.add_argument("--dim", type=int, default=1024, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=24, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Attention head dimension")
    parser.add_argument("--intermediate_size", type=int, default=4096, help="FFN intermediate size")

    # Training arguments - Optimized defaults
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (auto-calculated if None)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Peak learning rate (increased)")
    parser.add_argument("--min_learning_rate", type=float, default=6e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps (reduced)")
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
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data workers")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="Prefetch factor")

    # Optimization arguments
    parser.add_argument("--use_fp8", action="store_true", default=True, help="Use FP8 precision")
    parser.add_argument("--use_flash_attn", action="store_true", default=True, help="Use Flash Attention")
    parser.add_argument("--compile_model", action="store_true", default=True, help="Compile model")
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", default=True, help="Use gradient checkpointing"
    )

    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Use W&B logging")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_optimized", help="Output directory")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--log_interval", type=int, default=50, help="Logging interval")

    args = parser.parse_args()

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

    # Calculate optimal batch size if not provided
    if args.batch_size is None:
        args.batch_size = calculate_optimal_batch_size()
        print(f"🧮 Auto-calculated batch size: {args.batch_size}")

    # Validate batch size isn't too large
    if args.batch_size > 64:
        print(f"⚠️  Large batch size ({args.batch_size}) detected. Consider reducing if OOM occurs.")

    print(f"\n📊 Training Configuration:")
    print(f"  Model: 350M parameters ({args.n_layers} layers, {args.dim} hidden)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps * args.max_length:,} tokens")
    print(f"  Learning rate: {args.learning_rate} -> {args.min_learning_rate}")
    print(f"  Max training time: {args.max_hours} hours")
    print()

    # Create model
    print("🏗️  Creating model...")
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

    # Apply TorchAO optimizations if available
    if args.use_fp8:
        try:
            model = apply_torchao_optimizations(model, torch.device("cuda"))
        except Exception as e:
            print(f"⚠️  Failed to apply TorchAO optimizations: {e}")
            args.use_fp8 = False

    # Create optimized data loaders
    print("📂 Creating optimized data loaders from local .npy files...")
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
    estimated_tokens_per_sec = 500_000  # Conservative estimate with all optimizations
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

    # Create optimized training config
    config = TrainingConfig(
        model_name=f"gpt-350m-h100-ultra-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
        use_fp8=args.use_fp8,
        use_amp=False,  # Don't use both FP8 and AMP
        compile_model=args.compile_model,
        use_flash_attn=args.use_flash_attn,
        gradient_checkpointing=args.gradient_checkpointing,
        use_fused_adamw=True,
        use_cuda_graphs=True,
        use_wandb=args.use_wandb,
        output_dir=args.output_dir,
    )

    # Create optimized trainer
    print("🎯 Creating H100-optimized trainer...")
    trainer = H100OptimizedTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
    )

    # Print optimization status
    print(f"\n⚙️  Optimizations:")
    print(f"  FP8: {'✅' if config.use_fp8 else '❌'}")
    print(f"  Flash Attention: {'✅' if config.use_flash_attn else '❌'}")
    print(f"  Model Compilation: {'✅' if config.compile_model else '❌'}")
    print(f"  Gradient Checkpointing: {'✅' if config.gradient_checkpointing else '❌'}")
    print(f"  Fused AdamW: {'✅' if config.use_fused_adamw else '❌'}")

    # Memory estimate
    model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"\n💾 Memory Estimate:")
    print(f"  Model size: {model_size_gb:.1f} GB")
    print(f"  Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Expected utilization: ~60-70%")

    # Start training
    print("\n" + "🚀 Starting optimized training...")
    print("🎯 Target: Beat validation loss of 3.0781")
    print("🔥 Expected: Achieve validation loss of 2.8-2.9")
    print("-" * 80)

    try:
        final_metrics = trainer.train()

        # Print success/failure
        success = trainer.best_val_loss < 3.0781
        print(f"\n{'🎉' if success else '😞'} Training {'SUCCEEDED' if success else 'FAILED'}")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
