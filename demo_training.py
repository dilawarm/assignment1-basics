#!/usr/bin/env python3
"""
Demonstration of the H100-optimized training pipeline.
This script shows how to use the implementation with a small example.
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.model import TransformerLM
from cs336_basics.training import Trainer, TrainingConfig


def create_dummy_data(num_samples=100, seq_len=128, vocab_size=50257):
    """Create dummy data for demonstration."""
    from torch.utils.data import DataLoader, Dataset

    class DummyDataset(Dataset):
        def __init__(self, num_samples, seq_len, vocab_size):
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.vocab_size = vocab_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
            return {
                "input_ids": input_ids,
                "labels": input_ids,
            }

    dataset = DummyDataset(num_samples, seq_len, vocab_size)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    return dataloader


def main():
    print("=" * 80)
    print("H100-Optimized Transformer Training Demo")
    print("=" * 80)

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create a smaller model for demonstration
    print("\nCreating demonstration model (smaller for testing)...")
    model = TransformerLM(
        vocab_size=50257,
        max_seq_len=128,
        dim=512,  # Smaller for demo
        n_layers=6,  # Fewer layers for demo
        n_heads=8,
        head_dim=64,
        intermediate_size=2048,
        dropout=0.0,
        tie_embeddings=True,
        use_flash=False,  # Disable for compatibility
        use_fp8=False,  # Disable for compatibility
        use_gradient_checkpointing=False,
    )

    # Create dummy data
    print("\nCreating dummy data loaders...")
    train_loader = create_dummy_data(num_samples=100, seq_len=128)
    val_loader = create_dummy_data(num_samples=20, seq_len=128)

    # Create training config
    config = TrainingConfig(
        model_name="demo-transformer",
        learning_rate=1e-3,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=10,  # Just 10 steps for demo
        warmup_steps=2,
        eval_interval=5,
        log_interval=1,
        use_fp8=False,
        use_amp=device == "cuda",
        compile_model=False,  # Disable for faster demo
        use_flash_attn=False,
        gradient_checkpointing=False,
        use_wandb=False,  # Disable for demo
        output_dir="./demo_checkpoints",
    )

    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
    )

    # Run training
    print("\nStarting demo training (10 steps)...")
    print("-" * 80)

    final_metrics = trainer.train()

    print("\n" + "=" * 80)
    print("Demo complete!")
    print(f"Final loss: {final_metrics['val_loss']:.4f}")
    print(f"Final perplexity: {final_metrics['val_perplexity']:.2f}")
    print("=" * 80)

    # Test generation
    print("\nTesting generation...")
    model.eval()
    with torch.no_grad():
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        generated = model.generate(prompt, max_length=20, temperature=0.8)
        print(f"Generated tokens: {generated.shape}")

    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()
