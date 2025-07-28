"""
Simple Transformer Training Script

A clean and straightforward training setup similar to the Jupyter notebook implementation,
but using the existing repository components.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from tqdm import tqdm

from cs336_basics.data import get_batch
from cs336_basics.loss.cross_entropy import cross_entropy
from cs336_basics.nn.models import TransformerLM
from cs336_basics.tokenization.tokenizer import Tokenizer
from cs336_basics.training.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.training.gradient_clipping import gradient_clipping
from cs336_basics.training.lr_schedules import cosine_learning_rate_schedule
from cs336_basics.training.optimizers import AdamW


@dataclass
class TrainModelArgs:
    """Training configuration - simplified like the notebook."""

    # Model parameters
    vocab_size: int = 32000
    context_length: int = 256
    num_layers: int = 4
    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000

    # Optimizer parameters
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)

    # Learning rate schedule
    max_learning_rate: float = 2e-3
    min_learning_rate: float = 1e-5
    warmup_iters: int = 2000
    cosine_cycle_iters: int = 40960

    # Data paths
    training_set: str = "owt_train_tokens.npy"
    validation_set: str = "owt_valid_tokens.npy"
    tokenizer_vocab: str = "openwebtext_vocab.json"
    tokenizer_merges: str = "openwebtext_merges.pkl"

    # Training configuration
    validation_step_interval: int = 500
    checkpoint_step_interval: int = 10000
    steps: int = 40960
    batch_size: int = 32
    gradient_clipping: float = 1.0

    # Device and optimization
    device: str = "cuda"
    compile_model: bool = True

    # Wandb settings
    wandb_active: bool = False
    wandb_project: str = "cs336-assignment1"
    wandb_run: str = ""

    def __post_init__(self):
        """Validate configuration."""
        assert self.vocab_size > 0
        assert self.context_length > 0
        assert self.d_model > 0
        assert self.d_model % self.num_heads == 0
        assert self.steps > 0
        assert self.batch_size > 0
        assert self.max_learning_rate > 0


class TrainModel:
    """Simple trainer class similar to the notebook implementation."""

    def __init__(self, args: TrainModelArgs):
        self.args = args
        self.step = 0
        self.device = torch.device(args.device)

        # Setup model
        self.model = TransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            device=self.device,
        )

        # Compile model if requested
        if args.compile_model and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model)
                print("✅ Model compiled successfully")
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}")

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(), lr=args.max_learning_rate, weight_decay=args.weight_decay, betas=args.betas
        )

        # Setup tokenizer
        self.tokenizer = Tokenizer.from_files(args.tokenizer_vocab, args.tokenizer_merges, ["<|endoftext|>"])

        # Load datasets
        print(f"Loading training data from {args.training_set}")
        self.training_set = np.load(args.training_set, mmap_mode="r")
        print(f"Training set size: {len(self.training_set):,} tokens")

        if Path(args.validation_set).exists():
            print(f"Loading validation data from {args.validation_set}")
            self.validation_set = np.load(args.validation_set, mmap_mode="r")
            print(f"Validation set size: {len(self.validation_set):,} tokens")
        else:
            self.validation_set = None
            print("No validation set found")

        # Initialize wandb if requested
        if args.wandb_active:
            wandb.init(project=args.wandb_project, name=args.wandb_run, config=asdict(args))
            wandb.watch(self.model, log="gradients", log_freq=10)

    def get_lr(self, step: int) -> float:
        """Get learning rate using cosine schedule."""
        return cosine_learning_rate_schedule(
            iteration=step,
            max_learning_rate=self.args.max_learning_rate,
            min_learning_rate=self.args.min_learning_rate,
            warmup_iters=self.args.warmup_iters,
            cosine_cycle_iters=self.args.cosine_cycle_iters,
        )

    def evaluate(self) -> tuple[float, float]:
        """Evaluate the model on validation set."""
        if self.validation_set is None:
            return float("inf"), float("inf")

        self.model.eval()
        total_loss = 0.0
        num_batches = 50  # Fixed number of eval batches

        with torch.no_grad():
            for _ in range(num_batches):
                try:
                    inputs, targets = get_batch(
                        self.validation_set, self.args.batch_size, self.args.context_length, device=str(self.device)
                    )

                    logits = self.model(inputs)
                    loss = cross_entropy(logits, targets)

                    if torch.isfinite(loss):
                        total_loss += loss.item()
                    else:
                        num_batches -= 1

                except Exception as e:
                    print(f"⚠️ Evaluation batch failed: {e}")
                    num_batches -= 1

        if num_batches == 0:
            return float("inf"), float("inf")

        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 10))  # Cap to prevent overflow

        return avg_loss, perplexity

    def train_step(self) -> dict[str, Any]:
        """Single training step."""
        self.model.train()

        # Get learning rate
        lr = self.get_lr(self.step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # Zero gradients
        self.optimizer.zero_grad()

        # Get batch
        inputs, targets = get_batch(
            self.training_set, self.args.batch_size, self.args.context_length, device=str(self.device)
        )

        # Forward pass
        logits = self.model(inputs)
        loss = cross_entropy(logits, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        grad_norm = gradient_clipping(self.model.parameters(), self.args.gradient_clipping)

        # Optimizer step
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "lr": lr,
            "grad_norm": grad_norm,
        }

    def train(self):
        """Main training loop."""
        print(f"\n🚀 Starting training for {self.args.steps} steps")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")

        # Initial evaluation
        val_loss, val_perplexity = self.evaluate()
        print(f"Initial validation - loss: {val_loss:.4f}, perplexity: {val_perplexity:.2f}")

        if self.args.wandb_active:
            wandb.log({"val_loss": val_loss, "val_perplexity": val_perplexity, "step": 0})

        # Training loop
        pbar = tqdm(range(self.args.steps), desc="Training")
        start_time = time.time()

        for step in pbar:
            self.step = step

            # Training step
            metrics = self.train_step()

            # Calculate tokens per second
            elapsed_time = time.time() - start_time
            tokens_processed = (step + 1) * self.args.batch_size * self.args.context_length
            tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0

            # Update progress bar
            pbar.set_postfix(
                {"loss": f"{metrics['loss']:.4f}", "lr": f"{metrics['lr']:.2e}", "tok/s": f"{tokens_per_sec:.0f}"}
            )

            # Evaluation
            if step > 0 and step % self.args.validation_step_interval == 0:
                val_loss, val_perplexity = self.evaluate()

                print(
                    f"\nStep {step}: train_loss={metrics['loss']:.4f}, "
                    f"val_loss={val_loss:.4f}, val_perplexity={val_perplexity:.2f}"
                )

                if self.args.wandb_active:
                    wandb.log({"val_loss": val_loss, "val_perplexity": val_perplexity, "step": step})

            # Checkpoint saving
            if step > 0 and step % self.args.checkpoint_step_interval == 0:
                checkpoint_path = f"checkpoint_step_{step}.pt"
                save_checkpoint(self.model, self.optimizer, step, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

            # Log to wandb
            if self.args.wandb_active:
                wandb.log(
                    {
                        "train_loss": metrics["loss"],
                        "train_perplexity": math.exp(min(metrics["loss"], 10)),
                        "lr": metrics["lr"],
                        "grad_norm": metrics["grad_norm"],
                        "tokens_per_sec": tokens_per_sec,
                        "step": step,
                    }
                )

        # Final evaluation and checkpoint
        val_loss, val_perplexity = self.evaluate()
        print(f"\nFinal evaluation - val_loss: {val_loss:.4f}, val_perplexity: {val_perplexity:.2f}")

        # Save final checkpoint
        final_checkpoint = f"final_checkpoint_step_{self.args.steps}.pt"
        save_checkpoint(self.model, self.optimizer, self.args.steps, final_checkpoint)
        print(f"Saved final checkpoint: {final_checkpoint}")

        if self.args.wandb_active:
            wandb.log({"final_val_loss": val_loss, "final_val_perplexity": val_perplexity, "step": self.args.steps})
            wandb.finish()


def load_config_from_json(config_path: str) -> TrainModelArgs:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return TrainModelArgs(**config_dict)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple Transformer Training")

    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--val_data", type=str, help="Path to validation data")
    parser.add_argument("--steps", type=int, default=40960, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1", help="Wandb project")
    parser.add_argument("--wandb_run", type=str, default="transformer_training", help="Wandb run name")
    parser.add_argument("--no_compile", action="store_true", help="Disable model compilation")

    args = parser.parse_args()

    # Load config or create from args
    if args.config:
        config = load_config_from_json(args.config)
        # Override with command line args if provided
        if args.train_data:
            config.training_set = args.train_data
        if args.val_data:
            config.validation_set = args.val_data
    else:
        config = TrainModelArgs(
            training_set=args.train_data or "owt_train_tokens.npy",
            validation_set=args.val_data or "owt_valid_tokens.npy",
            steps=args.steps,
            batch_size=args.batch_size,
            max_learning_rate=args.lr,
            wandb_active=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run=args.wandb_run,
            compile_model=not args.no_compile,
        )

    # Create trainer and start training
    trainer = TrainModel(config)
    trainer.train()


if __name__ == "__main__":
    main()
