"""H100-optimized trainer with selective mixed precision and maximum throughput."""

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Training configuration optimized for H100."""

    # Model
    model_name: str = "gpt-350m-h100-optimized"

    # Optimization - Aggressive settings for H100
    learning_rate: float = 6e-4  # Increased for faster convergence
    min_learning_rate: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # Training - Optimized for 80GB H100
    batch_size: int = 32  # Increased significantly
    gradient_accumulation_steps: int = 4  # Reduced since batch_size increased
    max_steps: Optional[int] = None
    warmup_steps: int = 1000  # Reduced for faster ramp-up

    # Evaluation
    eval_interval: int = 500  # More frequent evaluation
    log_interval: int = 50  # More frequent logging
    save_interval: int = 2000

    # Data
    max_length: int = 1024
    num_workers: int = 8
    prefetch_factor: int = 4

    # Precision - Use selective mixed precision instead of aggressive FP8
    use_mixed_precision: bool = True
    use_bf16: bool = True  # BF16 is more stable than FP16 for large models

    # Hardware optimizations
    compile_model: bool = True
    use_flash_attn: bool = True
    gradient_checkpointing: bool = True

    # Advanced optimizations
    use_fused_adamw: bool = True

    # Logging
    use_wandb: bool = True
    project_name: str = "cs336-h100-optimized"

    # Paths
    output_dir: str = "./checkpoints"
    resume_from: Optional[str] = None


class H100OptimizedTrainer:
    """H100-optimized trainer with maximum throughput focus."""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: TrainingConfig,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("H100 trainer requires CUDA!")

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup mixed precision (selective approach)
        self.scaler = None
        if config.use_mixed_precision:
            if config.use_bf16:
                # BF16 is more stable for large models and works well with torch.compile()
                print("🔧 Using BF16 mixed precision training...")
                # Model dtype conversion is handled in apply_selective_mixed_precision
            else:
                # FP16 with gradient scaling
                print("🔧 Using FP16 mixed precision training...")
                self.scaler = GradScaler()

        # Compile model for maximum performance
        if config.compile_model:
            print("🔧 Compiling model with torch.compile...")
            self.model = torch.compile(
                self.model,
                mode="max-autotune",  # Maximum optimization
                fullgraph=True,  # Compile entire graph
                backend="inductor",  # Use TorchInductor backend
            )
            print("✅ Model compiled successfully")

        # Setup optimizer with optimizations
        self.optimizer = self._create_optimizer()

        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Performance tracking
        self.total_tokens = 0
        self.start_time = time.time()

        # Setup logging
        if config.use_wandb:
            self._setup_wandb()

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # CUDA optimizations
        self._setup_cuda_optimizations()

    def _setup_cuda_optimizations(self):
        """Setup CUDA-specific optimizations."""
        # Enable TensorFloat-32 for maximum performance on H100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable Flash Attention optimizations
        torch.backends.cuda.enable_flash_sdp(True)

        # Optimize memory allocation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        print("✅ Enabled CUDA optimizations for H100")

    def _create_optimizer(self):
        """Create optimized AdamW optimizer."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ["bias", "norm", "embedding"]):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use fused AdamW for better performance
        optimizer = AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
            fused=self.config.use_fused_adamw,
        )

        return optimizer

    def _create_scheduler(self):
        """Create optimized learning rate scheduler."""

        def lr_lambda(step):
            # Warmup
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps

            # Cosine decay
            progress = (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)
            return self.config.min_learning_rate / self.config.learning_rate + (
                1 - self.config.min_learning_rate / self.config.learning_rate
            ) * 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        wandb.init(
            project=self.config.project_name,
            name=self.config.model_name,
            config=self.config.__dict__,
        )

    def _get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU memory and utilization stats."""
        if not NVML_AVAILABLE:
            return {}

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)

            return {
                "gpu_memory_used_gb": mem_info.used / 1e9,
                "gpu_memory_total_gb": mem_info.total / 1e9,
                "gpu_memory_percent": (mem_info.used / mem_info.total) * 100,
                "gpu_utilization": util_info.gpu,
            }
        except:
            return {}

    def _calculate_mfu(self, tokens_per_sec: float, batch_size: int, seq_len: int) -> float:
        """Calculate Model FLOPS Utilization (MFU)."""
        # Model parameters (350M)
        model_params = 350e6

        # FLOPs per token for transformer
        # Forward: 6 * params (attention + mlp)
        # Backward: 2 * forward (roughly)
        flops_per_token = 6 * model_params * 3  # 3x for forward + backward

        # Total FLOPs per second
        model_flops_per_sec = tokens_per_sec * flops_per_token

        # H100 peak FLOPs (theoretical)
        # BF16: ~1000 TFLOPs/s, FP16: ~1000 TFLOPs/s
        peak_flops = 1000e12

        # MFU calculation
        mfu = (model_flops_per_sec / peak_flops) * 100
        return min(mfu, 100.0)  # Cap at 100%

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Optimized training step."""
        # Move batch to device efficiently
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        # Forward pass with mixed precision
        if self.config.use_mixed_precision:
            if self.config.use_bf16:
                # BF16 mixed precision (no gradient scaling needed)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model(input_ids=input_ids, labels=labels)
            else:
                # FP16 mixed precision with gradient scaling
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids=input_ids, labels=labels)
        else:
            # Full precision
            outputs = self.model(input_ids=input_ids, labels=labels)

        loss = outputs["loss"]

        # Scale loss by gradient accumulation steps
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {"loss": loss.item() * self.config.gradient_accumulation_steps}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Fast evaluation on validation set."""
        self.model.eval()

        total_loss = 0
        total_tokens = 0
        num_batches = 0
        max_eval_batches = 200  # Limit evaluation time

        eval_iter = iter(self.val_dataloader)

        for _ in range(min(max_eval_batches, len(self.val_dataloader))):
            try:
                batch = next(eval_iter)
            except StopIteration:
                break

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            if self.config.use_mixed_precision:
                if self.config.use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = self.model(input_ids=input_ids, labels=labels)
                else:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids=input_ids, labels=labels)
            else:
                outputs = self.model(input_ids=input_ids, labels=labels)

            loss = outputs["loss"]

            batch_size, seq_len = input_ids.shape
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len
            num_batches += 1

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = math.exp(min(avg_loss, 10))  # Cap for numerical stability

        self.model.train()

        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
            "eval_batches": num_batches,
        }

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}.pt")

        # Get raw model (unwrap compiled model if needed)
        raw_model = self.model
        if hasattr(self.model, "_orig_mod"):
            raw_model = self.model._orig_mod

        checkpoint = {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint to {path}")

    def train(self, max_steps: Optional[int] = None):
        """Optimized training loop for H100."""
        if max_steps is not None:
            self.config.max_steps = max_steps

        # Calculate total steps for 1.5 hours
        if self.config.max_steps is None:
            # Target: 1.5 hours of training
            # With optimizations, aim for 500K+ tokens/sec
            estimated_tokens_per_sec = 500_000
            total_seconds = 1.5 * 3600
            total_tokens = estimated_tokens_per_sec * total_seconds
            tokens_per_step = self.config.batch_size * self.config.gradient_accumulation_steps * self.config.max_length
            self.config.max_steps = int(total_tokens / tokens_per_step)

        print(f"🚀 Starting training for {self.config.max_steps} steps...")
        print(
            f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps * self.config.max_length:,} tokens"
        )

        # Training loop
        self.model.train()

        # Warmup CUDA kernels (more conservative for compatibility)
        print("🔥 Warming up CUDA kernels...")
        warmup_steps = 2  # Reduced warmup to avoid potential issues
        for i, batch in enumerate(self.train_dataloader):
            if i >= warmup_steps:
                break
            try:
                _ = self.train_step(batch)
                if (i + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            except Exception as e:
                print(f"⚠️  Warmup step {i} failed: {e}")
                break

        # Reset for actual training
        self.optimizer.zero_grad()
        self.global_step = 0
        self.start_time = time.time()

        print("✅ Warmup complete. Starting actual training...")

        while self.global_step < self.config.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                try:
                    # Training step
                    step_start = time.time()
                    metrics = self.train_step(batch)

                    # Track tokens
                    batch_tokens = batch["input_ids"].numel()
                    self.total_tokens += batch_tokens

                    # Gradient accumulation
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        # Gradient clipping
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)

                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                        # Optimizer step
                        if self.scaler is not None:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        self.scheduler.step()
                        self.optimizer.zero_grad()

                        self.global_step += 1

                        # Performance calculations
                        elapsed = time.time() - self.start_time
                        tokens_per_sec = self.total_tokens / elapsed if elapsed > 0 else 0
                        mfu = self._calculate_mfu(
                            tokens_per_sec,
                            self.config.batch_size * self.config.gradient_accumulation_steps,
                            self.config.max_length,
                        )

                        # Logging
                        if self.global_step % self.config.log_interval == 0:
                            log_metrics = {
                                "train_loss": metrics["loss"],
                                "learning_rate": self.scheduler.get_last_lr()[0],
                                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                                "tokens_per_second": tokens_per_sec,
                                "mfu_percent": mfu,
                                "global_step": self.global_step,
                            }

                            # Add GPU stats
                            gpu_stats = self._get_gpu_stats()
                            log_metrics.update(gpu_stats)

                            # Log to wandb
                            if self.config.use_wandb:
                                wandb.log(log_metrics)

                            # Print progress
                            print(
                                f"Step {self.global_step}: loss={metrics['loss']:.4f}, "
                                f"lr={log_metrics['learning_rate']:.2e}, "
                                f"tokens/sec={tokens_per_sec:.0f}, "
                                f"MFU={mfu:.1f}%"
                            )

                        # Evaluation
                        if self.global_step % self.config.eval_interval == 0:
                            print(f"\n🔍 Running evaluation at step {self.global_step}...")
                            eval_start = time.time()
                            eval_metrics = self.evaluate()
                            eval_time = time.time() - eval_start

                            print(
                                f"✅ Val loss: {eval_metrics['val_loss']:.4f} | "
                                f"Perplexity: {eval_metrics['val_perplexity']:.2f}"
                            )

                            # Log to wandb
                            if self.config.use_wandb:
                                wandb.log(eval_metrics)

                            # Save best model
                            if eval_metrics["val_loss"] < self.best_val_loss:
                                self.best_val_loss = eval_metrics["val_loss"]
                                best_path = os.path.join(self.config.output_dir, "best_model.pt")
                                self.save_checkpoint(best_path)
                                print(f"💾 New best model saved! Loss: {self.best_val_loss:.4f}")

                        # Save checkpoint
                        if self.global_step % self.config.save_interval == 0:
                            self.save_checkpoint()

                        # Check target loss
                        if hasattr(self, "best_val_loss") and self.best_val_loss < 3.0781:
                            print(f"\n🎯 TARGET ACHIEVED! Validation loss {self.best_val_loss:.4f} < 3.0781")
                            break

                        # Check if done
                        if self.global_step >= self.config.max_steps:
                            break

                except Exception as e:
                    print(f"⚠️  Training step failed: {e}")
                    # Continue training instead of crashing
                    continue

            self.epoch += 1

            # Early exit if target achieved
            if hasattr(self, "best_val_loss") and self.best_val_loss < 3.0781:
                break

        # Final evaluation
        print("\n🏁 Training complete! Running final evaluation...")
        final_metrics = self.evaluate()

        # Calculate final performance stats
        total_time = time.time() - self.start_time
        final_tokens_per_sec = self.total_tokens / total_time
        final_mfu = self._calculate_mfu(
            final_tokens_per_sec,
            self.config.batch_size * self.config.gradient_accumulation_steps,
            self.config.max_length,
        )

        print(f"\n" + "=" * 80)
        print("🏆 FINAL RESULTS")
        print("=" * 80)
        print(f"Final validation loss: {final_metrics['val_loss']:.4f}")
        print(f"Final validation perplexity: {final_metrics['val_perplexity']:.2f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total training time: {total_time / 3600:.2f} hours")
        print(f"Total tokens processed: {self.total_tokens:,}")
        print(f"Average tokens/sec: {final_tokens_per_sec:.0f}")
        print(f"Average MFU: {final_mfu:.1f}%")

        target_achieved = self.best_val_loss < 3.0781
        if target_achieved:
            print(f"✅ SUCCESS! Beat target of 3.0781 by {3.0781 - self.best_val_loss:.4f}")
        else:
            print(f"❌ Target not achieved. Missed by {self.best_val_loss - 3.0781:.4f}")

        print("=" * 80)

        # Save final model
        final_path = os.path.join(self.config.output_dir, "final_model.pt")
        self.save_checkpoint(final_path)

        # Close wandb
        if self.config.use_wandb:
            wandb.finish()

        return final_metrics


# Alias for backward compatibility
Trainer = H100OptimizedTrainer