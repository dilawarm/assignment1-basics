"""
Simplified Transformer Training Script
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from cs336_basics.data import get_batch
from cs336_basics.experiments.exp_logging import ExperimentLogger, TrainingIntegrator
from cs336_basics.loss.cross_entropy import cross_entropy
from cs336_basics.nn.models import EnhancedTransformerLM
from cs336_basics.training.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.training.gradient_clipping import advanced_gradient_clipping
from cs336_basics.training.lr_schedules import linear_decay_to_zero_schedule, warmup_schedule
from cs336_basics.training.optimizers import Adam, AdamW, MixedOptimizerV2, Muon


class StabilityTracker:
    """Simple stability tracker for tests compatibility."""

    def __init__(self, window_size: int = 200):
        self.losses = []
        self.grad_norms = []
        self.lr_history = []
        self.window_size = window_size

    def update(self, loss: float, grad_norm: float = 0.0, lr: float = 0.0) -> None:
        """Update with new training metrics."""
        self.losses.append(loss)
        self.grad_norms.append(grad_norm)
        self.lr_history.append(lr)

        if len(self.losses) > self.window_size:
            self.losses.pop(0)
            self.grad_norms.pop(0)
            self.lr_history.pop(0)

    def detect_training_issues(self) -> dict[str, bool]:
        """Basic issue detection."""
        if len(self.losses) < 20:
            return {"stable": True, "loss_spike": False, "gradient_explosion": False, "training_collapse": False}

        recent_losses = self.losses[-50:]
        very_recent = self.losses[-10:]

        issues = {
            "loss_spike": False,
            "gradient_explosion": False,
            "training_collapse": False,
            "stable": True,
        }

        if len(recent_losses) >= 10:
            recent_mean = np.mean(recent_losses[:-5])
            current_mean = np.mean(very_recent)
            if current_mean > recent_mean * 2.0:
                issues["loss_spike"] = True
                issues["stable"] = False

        if len(self.grad_norms) >= 10:
            recent_grads = self.grad_norms[-10:]
            if any(g > 10.0 for g in recent_grads):
                issues["gradient_explosion"] = True
                issues["stable"] = False

        if len(very_recent) >= 5:
            if any(math.isnan(l) or math.isinf(l) for l in very_recent):
                issues["training_collapse"] = True
                issues["stable"] = False

        return issues

    def get_comprehensive_stats(self) -> dict[str, float]:
        """Get basic stability statistics."""
        if len(self.losses) < 10:
            return {"stability_score": 1.0}

        recent_losses = self.losses[-100:]
        stats = {
            "loss_variance": float(np.var(recent_losses)),
            "stability_score": 1.0 / (1.0 + np.std(recent_losses)) if len(recent_losses) > 1 else 1.0,
        }
        return stats


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Data parameters
    train_data_path: str
    val_data_path: str | None = None
    vocab_size: int = 32000
    context_length: int = 512

    # Model parameters
    d_model: int = 1024
    num_layers: int = 16
    num_heads: int = 8
    d_ff: int = 4096
    rope_theta: float = 10000.0
    eps: float = 1e-5
    tie_embeddings: bool = False
    activation: str = "custom"
    use_unet_architecture: bool = True

    # Training parameters
    max_steps: int = 20000
    max_wallclock_hours: float = 1.5
    batch_size: int = 128
    gradient_accumulation_steps: int = 4

    # Optimizer settings
    optimizer: str = "mixed_v2"
    lr_schedule: str = "linear_decay_to_zero"
    learning_rate: float = 8e-3
    muon_lr: float = 8e-3
    adam_lr: float = 6e-3
    embedding_lr: float = 12e-3
    lm_head_lr: float = 4e-3
    min_learning_rate: float = 0.0
    warmup_steps: int = 100
    weight_decay: float = 0.015

    # Muon-specific parameters
    momentum: float = 0.97
    ns_iters: int = 6

    # Adam-specific parameters
    beta1: float = 0.9
    beta2: float = 0.95

    # Gradient clipping
    use_adaptive_clipping: bool = True
    grad_clip_norm: float = 1.0

    # Mixed precision
    use_amp: bool = True
    use_bfloat16: bool = True

    # Model compilation and optimization features (for test compatibility)
    compile_model: bool = True
    torch_compile_backend: str = "inductor"
    use_tf32: bool = True
    use_gradient_checkpointing: bool = False
    gradient_checkpointing_layers: int = 0  # For config file compatibility
    channels_last: bool = False

    # Data loading
    num_workers: int = 12
    pin_memory: bool = True
    prefetch_factor: int = 8

    # Additional optimizer settings (for config file compatibility)
    use_fused_adamw: bool = True
    min_learning_rate: float = 0.0

    # Logging and evaluation
    log_interval: int = 20
    eval_interval: int = 200
    eval_batches: int = 50
    save_interval: int = 1000

    # Directories and experiment tracking
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "openwebtext_h100_v2"
    experiment_description: str = "OpenWebText H100 training"
    use_wandb: bool = True
    wandb_project: str = "cs336-assignment1"

    # Device settings
    device: str = "cuda"
    resume_from: str | None = None
    auto_resume: bool = True

    # Emergency mode settings (for test compatibility)
    max_consecutive_failures: int = 3
    emergency_lr_reduction: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration."""
        import warnings

        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.context_length > 0, "context_length must be positive"
        assert self.d_model > 0, "d_model must be positive"
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert self.max_steps > 0, "max_steps must be positive"
        assert self.max_wallclock_hours > 0, "max_wallclock_hours must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Optimize d_ff for tensor cores
        if self.d_ff % 64 != 0:
            original_d_ff = self.d_ff
            self.d_ff = ((self.d_ff + 63) // 64) * 64
            warnings.warn(f"Adjusted d_ff from {original_d_ff} to {self.d_ff} for tensor core optimization")

        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        self.total_tokens = self.effective_batch_size * self.max_steps * self.context_length


class DataLoader:
    """Data loader for training."""

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        context_length: int,
        device: str,
        pin_memory: bool = True,
        prefetch_factor: int = 8,
        persistent_workers: bool = True,
        channels_last: bool = False,
    ) -> None:
        self.data_path = data_path
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = torch.device(device)
        self.pin_memory = pin_memory

        data_path_obj = Path(data_path)
        if not data_path_obj.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if data_path.endswith(".npy"):
            try:
                self.data = np.load(data_path, mmap_mode="r")
            except (ValueError, pickle.UnpicklingError):
                self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        else:
            self.data = np.memmap(data_path, dtype=np.uint16, mode="r")

        self.data_size = len(self.data)

        if self.data_size < context_length + 1:
            raise ValueError(f"Dataset too small: {self.data_size} < {context_length + 1}")

        print(f"Loaded dataset: {self.data_size:,} tokens from {data_path}")

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data."""
        inputs, targets = get_batch(
            dataset=self.data,
            batch_size=self.batch_size,
            context_length=self.context_length,
            device=str(self.device),
        )

        if self.device.type == "cuda":
            inputs = inputs.to(device=self.device, non_blocking=True)
            targets = targets.to(device=self.device, non_blocking=True)

        return inputs, targets


class Trainer:
    """Simplified trainer."""

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize the trainer."""
        self.config = config
        self.step = 0
        self.start_time = time.time()
        self.stability_tracker = StabilityTracker()
        self.emergency_mode = False
        self.consecutive_failures = 0

        # Setup experiment logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_experiment_name = f"{config.experiment_name}_{timestamp}"

        self.experiment_logger = ExperimentLogger(
            experiment_name=timestamped_experiment_name,
            description=config.experiment_description,
            log_dir="experiments",
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
        )

        from dataclasses import asdict

        self.experiment_logger.log_hyperparameters(**asdict(config))

        self.training_integrator = TrainingIntegrator(
            self.experiment_logger,
            hardware_log_interval=self.config.log_interval,
        )

        self._setup_device()
        self._setup_model()
        self._setup_optimizer()
        self._setup_amp()
        self._setup_data_loaders()

        if config.resume_from or config.auto_resume:
            self._try_resume()

        print(f"Trainer initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def _setup_device(self) -> None:
        """Setup device."""
        if self.config.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
            self.config.compile_model = False
            self.config.use_amp = False
            self.config.use_tf32 = False
        else:
            self.device = torch.device(self.config.device)

        if self.device.type == "cuda":
            if self.config.use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cudnn.benchmark = True

            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        # Note: TF32 setting is preserved even for CPU device for test compatibility

    def _setup_model(self) -> None:
        """Setup model."""
        self.model = EnhancedTransformerLM(
            vocab_size=self.config.vocab_size,
            context_length=self.config.context_length,
            d_model=self.config.d_model,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            d_ff=self.config.d_ff,
            rope_theta=self.config.rope_theta,
            eps=self.config.eps,
            tie_embeddings=self.config.tie_embeddings,
            activation=self.config.activation,
            use_unet_architecture=self.config.use_unet_architecture,
            device=self.device,
        )

        self.original_model = self.model

        if self.config.compile_model and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, backend=self.config.torch_compile_backend)
                print(f"Model compiled with {self.config.torch_compile_backend} backend")
            except Exception as e:
                print(f"Model compilation failed: {e}")
                self.config.compile_model = False

    def _setup_optimizer(self) -> None:
        """Setup optimizer."""
        if self.config.optimizer == "mixed_v2":
            self.optimizer = MixedOptimizerV2(
                model=self.original_model,
                muon_lr=self.config.muon_lr,
                adam_lr=self.config.adam_lr,
                embedding_lr=self.config.embedding_lr,
                lm_head_lr=self.config.lm_head_lr,
                muon_momentum=self.config.momentum,
                adam_betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
                eps=self.config.eps,
                ns_iters=self.config.ns_iters,
            )
            print("Using MixedOptimizerV2 (Muon + Adam)")
        elif self.config.optimizer == "adam":
            self.optimizer = Adam(
                self.original_model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
                eps=self.config.eps,
            )
            print("Using Adam optimizer")
        elif self.config.optimizer == "muon":
            self.optimizer = Muon(
                self.original_model.parameters(),
                lr=self.config.muon_lr,
                momentum=self.config.momentum,
                ns_iters=self.config.ns_iters,
                weight_decay=self.config.weight_decay,
                eps=self.config.eps,
            )
            print("Using Muon optimizer")
        else:
            self.optimizer = AdamW(
                self.original_model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
                eps=self.config.eps,
            )
            print("Using AdamW optimizer")

    def _setup_amp(self) -> None:
        """Setup Automatic Mixed Precision."""
        if self.config.use_amp and self.device.type == "cuda":
            if self.config.use_bfloat16:
                self.scaler = None
                self.amp_dtype = torch.bfloat16
                print("Enabled AMP with bfloat16")
            else:
                self.scaler = GradScaler()
                self.amp_dtype = torch.float16
                print("Enabled AMP with float16")
        else:
            self.scaler = None
            self.amp_dtype = torch.float32
            print("Using float32 precision")

    def _setup_data_loaders(self) -> None:
        """Setup data loaders."""
        dataloader_kwargs = {
            "batch_size": self.config.batch_size,
            "context_length": self.config.context_length,
            "device": str(self.device),
            "pin_memory": self.config.pin_memory,
            "prefetch_factor": self.config.prefetch_factor,
        }

        self.train_loader = DataLoader(data_path=self.config.train_data_path, **dataloader_kwargs)

        self.val_loader = None
        if self.config.val_data_path and Path(self.config.val_data_path).exists():
            self.val_loader = DataLoader(data_path=self.config.val_data_path, **dataloader_kwargs)
            print("Validation data loader ready")

    def _try_resume(self) -> None:
        """Try to resume from checkpoint."""
        checkpoint_path = None

        if self.config.resume_from and Path(self.config.resume_from).exists():
            checkpoint_path = self.config.resume_from
        elif self.config.auto_resume:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
                if checkpoints:
                    checkpoint_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))

        if checkpoint_path:
            try:
                self.step = load_checkpoint(checkpoint_path, self.original_model, self.optimizer)
                print(f"Resumed from {checkpoint_path} at step {self.step}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting fresh training...")

    def get_lr(self, step: int) -> dict[str, float]:
        """Get learning rate schedule."""
        if self.config.lr_schedule == "linear_decay_to_zero":
            base_lr = linear_decay_to_zero_schedule(
                iteration=step,
                max_learning_rate=self.config.learning_rate,
                warmup_iters=self.config.warmup_steps,
                total_iters=self.config.max_steps,
            )
        else:
            base_lr = warmup_schedule(
                iteration=step,
                max_learning_rate=self.config.learning_rate,
                warmup_iters=self.config.warmup_steps,
                total_iters=self.config.max_steps,
                warmup_type="linear",
            )

        # Apply emergency mode reduction if needed
        if self.emergency_mode:
            base_lr *= self.config.emergency_lr_reduction

        lr_factor = base_lr / self.config.learning_rate if self.config.learning_rate > 0 else 0

        return {
            "base_lr": base_lr,
            "muon_lr": self.config.muon_lr * lr_factor,
            "adam_lr": self.config.adam_lr * lr_factor,
            "embedding_lr": self.config.embedding_lr * lr_factor,
            "lm_head_lr": self.config.lm_head_lr * lr_factor,
        }

    def train_step(self) -> dict[str, Any]:
        """Execute one training step."""
        self.model.train()
        step_start_time = time.time()

        lrs = self.get_lr(self.step)

        # Update learning rates for mixed optimizer
        if self.config.optimizer == "mixed_v2":
            muon_lr_factor = lrs["muon_lr"] / self.config.muon_lr if self.config.muon_lr > 0 else 0
            adam_lr_factor = lrs["adam_lr"] / self.config.adam_lr if self.config.adam_lr > 0 else 0
            embedding_lr_factor = lrs["embedding_lr"] / self.config.embedding_lr if self.config.embedding_lr > 0 else 0
            lm_head_lr_factor = lrs["lm_head_lr"] / self.config.lm_head_lr if self.config.lm_head_lr > 0 else 0

            self.optimizer.update_learning_rates(muon_lr_factor, adam_lr_factor, embedding_lr_factor, lm_head_lr_factor)
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lrs["base_lr"]

        self.optimizer.zero_grad()

        total_loss = 0.0
        for accumulation_step in range(self.config.gradient_accumulation_steps):
            inputs, targets = self.train_loader.get_batch()

            if self.config.use_amp:
                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    logits = self.model(inputs)
                    loss = cross_entropy(logits, targets)
                    loss = loss / self.config.gradient_accumulation_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                logits = self.model(inputs)
                loss = cross_entropy(logits, targets)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

            total_loss += loss.item()

        # Gradient clipping
        grad_norm = advanced_gradient_clipping(
            self.original_model,
            max_global_norm=self.config.grad_clip_norm,
            use_adaptive=self.config.use_adaptive_clipping,
        )

        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        step_time = time.time() - step_start_time

        return {
            "loss": total_loss,
            "lr": lrs["base_lr"],
            "grad_norm": grad_norm,
            "step_time": step_time,
            **lrs,
        }

    def evaluate(self) -> dict[str, Any]:
        """Evaluate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for _ in range(self.config.eval_batches):
                try:
                    inputs, targets = self.val_loader.get_batch()

                    if self.config.use_amp:
                        with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                            logits = self.model(inputs)
                            loss = cross_entropy(logits, targets)
                    else:
                        logits = self.model(inputs)
                        loss = cross_entropy(logits, targets)

                    if torch.isfinite(loss):
                        total_loss += loss.item()
                        num_batches += 1

                except Exception as e:
                    print(f"Evaluation batch failed: {e}")
                    break

        if num_batches == 0:
            return {}

        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 10))

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "eval_batches": num_batches,
        }

    def get_enhanced_metrics(self, base_metrics: dict[str, Any], step_time: float) -> dict[str, Any]:
        """Get enhanced training metrics for test compatibility."""
        tokens_per_sec = (self.config.effective_batch_size * self.config.context_length) / step_time

        # Simple MFU calculation for test compatibility
        N = sum(p.numel() for p in self.original_model.parameters()) if hasattr(self, "original_model") else 1000000
        flops_per_token = 6 * N  # Simplified calculation
        model_flops_per_sec = tokens_per_sec * flops_per_token
        peak_flops = 1979e12 if self.config.use_bfloat16 else 67e12  # H100 estimates
        mfu = model_flops_per_sec / peak_flops

        enhanced_metrics = {
            **base_metrics,
            "tokens_per_sec": tokens_per_sec,
            "samples_per_sec": tokens_per_sec / self.config.context_length,
            "effective_batch_size": self.config.effective_batch_size,
            "training_progress": self.step / self.config.max_steps,
            "wallclock_hours": (time.time() - self.start_time) / 3600,
            "mfu": min(mfu, 1.0),
        }

        # Add simple stability stats
        if hasattr(self, "stability_tracker"):
            stability_stats = self.stability_tracker.get_comprehensive_stats()
            enhanced_metrics.update(stability_stats)

        return enhanced_metrics

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint."""
        save_checkpoint(self.original_model, self.optimizer, self.step, path)

    def train(self) -> None:
        """Main training loop."""
        max_time_seconds = self.config.max_wallclock_hours * 3600
        best_val_loss = float("inf")

        print(f"Starting training for {self.config.max_steps} steps")
        print(f"Schedule: {self.config.lr_schedule} with {self.config.warmup_steps} warmup")
        print(f"Optimizer: {self.config.optimizer}")

        self.training_integrator.start_epoch(0)

        while self.step < self.config.max_steps:
            step_start_time = time.time()
            elapsed_time = step_start_time - self.start_time

            if elapsed_time >= max_time_seconds:
                print(f"Reached time limit of {self.config.max_wallclock_hours:.1f} hours")
                break

            metrics = self.train_step()

            step_time = time.time() - step_start_time
            elapsed_hours = elapsed_time / 3600
            tokens_per_sec = (self.config.effective_batch_size * self.config.context_length) / step_time

            # Logging
            if self.step % self.config.log_interval == 0:
                self.training_integrator.log_training_step(
                    wallclock_time=elapsed_hours,
                    step=self.step,
                    train_loss=metrics["loss"],
                    learning_rate=metrics["lr"],
                    tokens_processed=self.config.effective_batch_size * self.config.context_length,
                    samples_processed=self.config.effective_batch_size,
                    step_time=step_time,
                    tokens_per_sec=tokens_per_sec,
                )

                print(
                    f"Step {self.step:5d}: loss={metrics['loss']:.4f}, lr={metrics['lr']:.2e}, "
                    f"{tokens_per_sec:.0f} tok/s, grad_norm={metrics['grad_norm']:.3f}, "
                    f"{elapsed_hours:.2f}h"
                )

            # Evaluation
            if self.step % self.config.eval_interval == 0 and self.step > 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    val_loss = eval_metrics["loss"]

                    self.training_integrator.log_validation_step(
                        wallclock_time=elapsed_hours,
                        step=self.step,
                        val_loss=val_loss,
                        perplexity=eval_metrics["perplexity"],
                    )

                    status = ""
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        status = "NEW BEST!"
                        best_path = Path(self.config.checkpoint_dir) / f"best_model_step_{self.step}.pt"
                        self.save_checkpoint(str(best_path))

                    print(
                        f"Eval Step {self.step}: val_loss={val_loss:.4f}, "
                        f"perplexity={eval_metrics['perplexity']:.2f} {status}"
                    )

            # Checkpointing
            if self.step % self.config.save_interval == 0 and self.step > 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{self.step}.pt"
                self.save_checkpoint(str(checkpoint_path))

            self.step += 1

        # Final evaluation and checkpointing
        final_elapsed_hours = (time.time() - self.start_time) / 3600
        final_eval = self.evaluate()

        print(f"Training completed in {final_elapsed_hours:.2f} hours")
        if final_eval:
            print(f"Final validation loss: {final_eval['loss']:.4f}")

        final_checkpoint = Path(self.config.checkpoint_dir) / f"final_checkpoint_step_{self.step}.pt"
        self.save_checkpoint(str(final_checkpoint))

        self.experiment_logger.add_note(f"Training completed in {final_elapsed_hours:.2f} hours")
        self.experiment_logger.mark_completed()


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return TrainingConfig(**config_dict)


def main() -> None:
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Transformer Training System")

    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--train_data", type=str, help="Path to training data (.npy)")
    parser.add_argument("--val_data", type=str, help="Path to validation data (.npy)")
    parser.add_argument("--max_steps", type=int, default=20000, help="Maximum training steps")
    parser.add_argument("--max_hours", type=float, default=1.5, help="Maximum training time in hours")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-3, help="Learning rate")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        if args.train_data:
            config.train_data_path = args.train_data
        if args.val_data:
            config.val_data_path = args.val_data
    else:
        if not args.train_data:
            raise ValueError("Must specify either --config or --train_data")

        config = TrainingConfig(
            train_data_path=args.train_data,
            val_data_path=args.val_data,
            max_steps=args.max_steps,
            max_wallclock_hours=args.max_hours,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    if args.no_compile:
        config.compile_model = False
    if args.no_wandb:
        config.use_wandb = False

    print(f"Configuration: {config.experiment_name}")

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
