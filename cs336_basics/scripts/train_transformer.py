"""
Simple Transformer Training Script
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from cs336_basics.data import get_batch
from cs336_basics.experiments.exp_logging import ExperimentLogger, TrainingIntegrator
from cs336_basics.loss.cross_entropy import cross_entropy
from cs336_basics.nn.models import TransformerLM
from cs336_basics.training.checkpoint import save_checkpoint
from cs336_basics.training.gradient_clipping import gradient_clipping
from cs336_basics.training.lr_schedules import cosine_learning_rate_schedule
from cs336_basics.training.optimizers import AdamW


@dataclass
class TrainModelArgs:
    """Training configuration"""

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
    training_set: str = "data/encoded/owt_train_tokens.npy"
    validation_set: str = "data/encoded/owt_valid_tokens.npy"

    # Training configuration
    validation_step_interval: int = 500
    checkpoint_step_interval: int = 10000
    steps: int = 40960
    batch_size: int = 32
    gradient_clipping: float = 1.0

    # Device and optimization
    device: str = "cuda"
    compile_model: bool = True

    # Experiment logging settings
    experiment_name: str = "transformer_training"
    experiment_description: str = "Transformer language model training"
    use_wandb: bool = True
    wandb_project: str = "cs336-assignment1"
    wandb_entity: str = ""
    log_dir: str = "experiments"

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
    """Simple trainer class with comprehensive production logging."""

    def __init__(self, args: TrainModelArgs):
        self.args = args
        self.step = 0
        self.device = torch.device(args.device)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_experiment_name = f"{args.experiment_name}_{timestamp}"
        self.experiment_timestamp = timestamp

        self.experiment_logger = ExperimentLogger(
            experiment_name=timestamped_experiment_name,
            description=args.experiment_description,
            log_dir=args.log_dir,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity if args.wandb_entity else None,
        )

        print(f"🔬 Experiment: {timestamped_experiment_name}")

        self.experiment_logger.log_hyperparameters(**asdict(args))

        self.training_integrator = TrainingIntegrator(
            experiment_logger=self.experiment_logger,
            hardware_log_interval=50,
        )

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

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.experiment_logger.add_note(
            f"Model initialized: {total_params:,} total parameters, {trainable_params:,} trainable"
        )

        if args.compile_model and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model)
                self.experiment_logger.add_note("Model compiled successfully with torch.compile")
                print("✅ Model compiled successfully")
            except Exception as e:
                self.experiment_logger.add_note(f"Model compilation failed: {e}")
                print(f"⚠️ Model compilation failed: {e}")

        self.optimizer = AdamW(
            self.model.parameters(), lr=args.max_learning_rate, weight_decay=args.weight_decay, betas=args.betas
        )

        print(f"Loading training data from {args.training_set}")
        self.training_set = np.load(args.training_set, mmap_mode="r")
        print(f"Training set size: {len(self.training_set):,} tokens")
        self.experiment_logger.add_note(f"Training data loaded: {len(self.training_set):,} tokens")

        if Path(args.validation_set).exists():
            print(f"Loading validation data from {args.validation_set}")
            self.validation_set = np.load(args.validation_set, mmap_mode="r")
            print(f"Validation set size: {len(self.validation_set):,} tokens")
            self.experiment_logger.add_note(f"Validation data loaded: {len(self.validation_set):,} tokens")
        else:
            self.validation_set = None
            self.experiment_logger.add_note("No validation set found")
            print("No validation set found")

    def calculate_mfu(self, tokens_per_sec: float) -> float:
        """Calculate Model FLOPs Utilization (MFU) for performance monitoring."""
        try:
            N = sum(p.numel() for p in self.model.parameters())
            L = self.args.num_layers
            H = self.args.d_model
            Q = self.args.context_length
            V = self.args.vocab_size

            flops_per_token_forward = 6 * N + 12 * L * H * Q

            flops_per_token_training = 3 * flops_per_token_forward

            model_flops_per_sec = tokens_per_sec * flops_per_token_training

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
                if "H100" in gpu_name:
                    peak_flops = 1979e12
                elif "A100" in gpu_name:
                    peak_flops = 624e12
                elif "V100" in gpu_name:
                    peak_flops = 125e12
                else:
                    peak_flops = 100e12
            else:
                peak_flops = 1e12

            mfu = min(model_flops_per_sec / peak_flops, 1.0)
            return mfu

        except Exception as e:
            self.experiment_logger.add_note(f"MFU calculation failed: {e}")
            return 0.0

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
        num_batches = 50

        eval_start_time = time.time()
        with torch.no_grad():
            for batch_idx in range(num_batches):
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
                    self.experiment_logger.add_note(f"Evaluation batch {batch_idx} failed: {e}")
                    print(f"⚠️ Evaluation batch failed: {e}")
                    num_batches -= 1

        eval_time = time.time() - eval_start_time

        if num_batches == 0:
            return float("inf"), float("inf")

        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 10))

        eval_tokens_per_sec = (num_batches * self.args.batch_size * self.args.context_length) / eval_time
        self.experiment_logger.add_note(
            f"Evaluation completed: {num_batches} batches, {eval_tokens_per_sec:.0f} tokens/sec"
        )

        return avg_loss, perplexity

    def train_step(self) -> dict[str, Any]:
        """Single training step with comprehensive logging."""
        self.model.train()
        step_start_time = time.time()

        lr = self.get_lr(self.step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.optimizer.zero_grad()

        batch_start_time = time.time()
        inputs, targets = get_batch(
            self.training_set, self.args.batch_size, self.args.context_length, device=str(self.device)
        )
        batch_time = time.time() - batch_start_time

        forward_start_time = time.time()
        logits = self.model(inputs)
        loss = cross_entropy(logits, targets)
        forward_time = time.time() - forward_start_time

        backward_start_time = time.time()
        loss.backward()
        backward_time = time.time() - backward_start_time

        grad_norm = gradient_clipping(self.model.parameters(), self.args.gradient_clipping)

        optimizer_start_time = time.time()
        self.optimizer.step()
        optimizer_time = time.time() - optimizer_start_time

        step_time = time.time() - step_start_time

        return {
            "loss": loss.item(),
            "lr": lr,
            "grad_norm": grad_norm,
            "step_time": step_time,
            "batch_time": batch_time,
            "forward_time": forward_time,
            "backward_time": backward_time,
            "optimizer_time": optimizer_time,
        }

    def train(self):
        """Main training loop with comprehensive logging."""
        print(f"\n🚀 Starting training for {self.args.steps} steps")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")

        self.experiment_logger.add_note("Training started")
        self.training_integrator.start_epoch(0)

        val_loss, val_perplexity = self.evaluate()
        print(f"Initial validation - loss: {val_loss:.4f}, perplexity: {val_perplexity:.2f}")

        self.training_integrator.log_validation_step(
            step=0,
            val_loss=val_loss,
            perplexity=val_perplexity,
            wallclock_time=0.0,
        )

        pbar = tqdm(range(self.args.steps), desc="Training")
        training_start_time = time.time()

        for step in pbar:
            self.step = step

            metrics = self.train_step()

            elapsed_time = time.time() - training_start_time
            tokens_processed = (step + 1) * self.args.batch_size * self.args.context_length
            tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0
            mfu = self.calculate_mfu(tokens_per_sec)

            self.training_integrator.log_training_step(
                step=step,
                train_loss=metrics["loss"],
                learning_rate=metrics["lr"],
                tokens_processed=self.args.batch_size * self.args.context_length,
                samples_processed=self.args.batch_size,
                step_time=metrics["step_time"],
                tokens_per_sec=tokens_per_sec,
                wallclock_time=elapsed_time / 3600,
                grad_norm=metrics["grad_norm"],
                mfu=mfu,
                batch_time=metrics["batch_time"],
                forward_time=metrics["forward_time"],
                backward_time=metrics["backward_time"],
                optimizer_time=metrics["optimizer_time"],
                effective_batch_size=self.args.batch_size,
                sequence_length=self.args.context_length,
            )

            pbar.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "lr": f"{metrics['lr']:.2e}",
                    "tok/s": f"{tokens_per_sec:.0f}",
                    "MFU": f"{mfu:.3f}",
                    "grad_norm": f"{metrics['grad_norm']:.3f}",
                }
            )

            if step > 0 and step % self.args.validation_step_interval == 0:
                val_loss, val_perplexity = self.evaluate()

                print(
                    f"\nStep {step}: train_loss={metrics['loss']:.4f}, "
                    f"val_loss={val_loss:.4f}, val_perplexity={val_perplexity:.2f}, "
                    f"MFU={mfu:.3f}, tok/s={tokens_per_sec:.0f}"
                )

                self.training_integrator.log_validation_step(
                    step=step,
                    val_loss=val_loss,
                    perplexity=val_perplexity,
                    wallclock_time=elapsed_time / 3600,
                    mfu_at_validation=mfu,
                    tokens_per_sec_at_validation=tokens_per_sec,
                )

            if step > 0 and step % self.args.checkpoint_step_interval == 0:
                checkpoint_path = f"checkpoint_{self.experiment_timestamp}_step_{step}.pt"
                save_checkpoint(self.model, self.optimizer, step, checkpoint_path)
                self.experiment_logger.add_note(f"Checkpoint saved: {checkpoint_path}")
                print(f"Saved checkpoint: {checkpoint_path}")

        final_elapsed_time = time.time() - training_start_time
        val_loss, val_perplexity = self.evaluate()
        final_tokens_per_sec = (self.args.steps * self.args.batch_size * self.args.context_length) / final_elapsed_time
        final_mfu = self.calculate_mfu(final_tokens_per_sec)

        print(f"\nFinal evaluation - val_loss: {val_loss:.4f}, val_perplexity: {val_perplexity:.2f}")
        print(
            f"Training completed - {final_elapsed_time / 3600:.2f} hours, {final_tokens_per_sec:.0f} tok/s, MFU: {final_mfu:.3f}"
        )

        self.training_integrator.log_validation_step(
            step=self.args.steps,
            val_loss=val_loss,
            perplexity=val_perplexity,
            wallclock_time=final_elapsed_time / 3600,
            final_mfu=final_mfu,
            final_tokens_per_sec=final_tokens_per_sec,
            training_hours=final_elapsed_time / 3600,
        )

        final_checkpoint = f"final_checkpoint_{self.experiment_timestamp}_step_{self.args.steps}.pt"
        save_checkpoint(self.model, self.optimizer, self.args.steps, final_checkpoint)
        print(f"Saved final checkpoint: {final_checkpoint}")

        self.experiment_logger.add_note(
            f"Training completed successfully. Final metrics: "
            f"val_loss={val_loss:.4f}, MFU={final_mfu:.3f}, "
            f"tokens/sec={final_tokens_per_sec:.0f}"
        )
        self.experiment_logger.mark_completed(success=True)


def load_config_from_json(config_path: str) -> TrainModelArgs:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return TrainModelArgs(**config_dict)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple Transformer Training with Production Logging")

    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--val_data", type=str, help="Path to validation data")
    parser.add_argument("--steps", type=int, default=40960, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")

    parser.add_argument("--experiment_name", type=str, default="transformer_training", help="Experiment name")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1", help="Wandb project")
    parser.add_argument("--wandb_entity", type=str, default="", help="Wandb entity")
    parser.add_argument("--log_dir", type=str, default="experiments", help="Local logging directory")
    parser.add_argument("--no_compile", action="store_true", help="Disable model compilation")

    args = parser.parse_args()

    if args.config:
        config = load_config_from_json(args.config)
        if args.train_data:
            config.training_set = args.train_data
        if args.val_data:
            config.validation_set = args.val_data
        if args.experiment_name != "transformer_training":
            config.experiment_name = args.experiment_name
        if args.wandb:
            config.use_wandb = True
        if args.wandb_project != "cs336-assignment1":
            config.wandb_project = args.wandb_project
        if args.wandb_entity:
            config.wandb_entity = args.wandb_entity
        if args.log_dir != "experiments":
            config.log_dir = args.log_dir
    else:
        config = TrainModelArgs(
            training_set=args.train_data or "data/encoded/owt_train_tokens.npy",
            validation_set=args.val_data or "data/encoded/owt_valid_tokens.npy",
            steps=args.steps,
            batch_size=args.batch_size,
            max_learning_rate=args.lr,
            experiment_name=args.experiment_name,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            log_dir=args.log_dir,
            compile_model=not args.no_compile,
        )

    trainer = TrainModel(config)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        trainer.experiment_logger.add_note("Training interrupted by user")
        trainer.experiment_logger.mark_completed(success=False)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        trainer.experiment_logger.add_note(f"Training failed with error: {e}")
        trainer.experiment_logger.mark_completed(success=False)
        raise


if __name__ == "__main__":
    main()
