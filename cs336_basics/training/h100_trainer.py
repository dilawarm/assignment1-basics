"""H100OptimizedTrainer with 2025 state-of-the-art optimizations."""

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

from ..model import apply_selective_mixed_precision


@dataclass
class H100TrainingConfig:
    """H100-optimized training configuration with 2025 best practices."""

    # Model
    model_name: str = "gpt-350m-h100-optimized"

    # Core optimization settings
    learning_rate: float = 6e-4  # Increased from 4e-4 based on 2025 research
    min_learning_rate: float = 6e-5  # 10% of max_lr
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # Advanced batch configuration for H100
    batch_size: int = 16  # Increased for better H100 utilization
    gradient_accumulation_steps: int = 8  # Adjusted to maintain ~1M tokens
    effective_batch_size_tokens: int = 1_048_576  # 1M tokens target

    # Training schedule
    max_steps: Optional[int] = None
    warmup_steps: int = 1000  # Reduced for faster convergence
    cooldown_steps: int = 500  # Add cooldown phase

    # Evaluation and logging
    eval_interval: int = 500  # More frequent evaluation
    log_interval: int = 50  # More frequent logging
    save_interval: int = 2000

    # Data configuration
    max_length: int = 1024
    num_workers: int = 8  # Increased for better data throughput

    # H100-specific optimizations
    use_flash_attention: bool = True
    use_fp8: bool = True
    use_compile: bool = True
    gradient_checkpointing: bool = True

    # Advanced optimizations
    use_fused_adamw: bool = True
    use_lr_warmup_cosine_decay: bool = True
    use_weight_decay_schedule: bool = False  # Disable for now to avoid complexity

    # Memory optimizations
    pin_memory: bool = True
    non_blocking: bool = True

    # Logging
    use_wandb: bool = True
    project_name: str = "cs336-h100-optimized"

    # Paths
    output_dir: str = "./checkpoints_optimized"
    resume_from: Optional[str] = None


class H100OptimizedTrainer:
    """
    H100OptimizedTrainer with 2025 state-of-the-art optimizations.

    Key improvements over base trainer:
    1. Aggressive H100-specific optimizations
    2. Dynamic batch size scaling
    3. Advanced learning rate scheduling
    4. FP8 mixed precision with TorchAO
    5. Optimized data pipeline
    6. Better memory management
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: H100TrainingConfig,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config

        # Setup device and verify H100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._verify_h100_setup()

        # Move model to device
        self.model = self.model.to(self.device)

        # Apply H100-specific optimizations
        self._apply_h100_optimizations()

        # Setup optimizer with H100 optimizations
        self.optimizer = self._create_optimized_optimizer()

        # Setup advanced learning rate scheduler
        self.scheduler = self._create_advanced_scheduler()

        # Setup mixed precision
        self.scaler = GradScaler() if not config.use_fp8 else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.tokens_processed = 0
        self.start_time = time.time()

        # Performance tracking
        self.step_times = []
        self.throughput_history = []

        # Setup logging
        if config.use_wandb:
            self._setup_advanced_wandb()

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        print("🚀 H100OptimizedTrainer initialized with 2025 optimizations!")

    def _verify_h100_setup(self):
        """Verify H100 GPU and optimal configuration."""
        if self.device.type != "cuda":
            print("⚠️  WARNING: Not using CUDA! Performance will be severely degraded.")
            return

        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎯 GPU: {gpu_name}")

        # Check for H100 specific features
        compute_capability = torch.cuda.get_device_capability(0)
        print(f"🔍 Compute capability: {compute_capability[0]}.{compute_capability[1]}")

        # Verify H100 optimizations are available
        if compute_capability[0] >= 9:  # H100 has compute capability 9.0
            print("✅ H100 detected! Enabling all optimizations.")
        elif compute_capability[0] >= 8:  # A100 has compute capability 8.0
            print("🔶 A100/RTX detected. Some H100 optimizations may not be available.")
        else:
            print("⚠️  Older GPU detected. Performance will be limited.")

        # Check memory
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"💾 GPU Memory: {memory_gb:.1f} GB")

        if memory_gb < 70:
            print("⚠️  Less than 80GB memory. Consider reducing batch size.")

    def _apply_h100_optimizations(self):
        """Apply H100-specific model optimizations."""
        print("🔧 Applying H100-specific optimizations...")

        # Apply selective mixed precision (FP8/BF16)
        self.model = apply_selective_mixed_precision(self.model, use_compile=self.config.use_compile)

        # Compile model for maximum performance
        if self.config.use_compile:
            print("⚡ Compiling model with torch.compile() for H100...")
            try:
                # Use aggressive compilation for H100
                self.model = torch.compile(
                    self.model,
                    mode="max-autotune",  # Most aggressive optimization
                    dynamic=False,  # Static shapes for H100 optimization
                    fullgraph=True,  # Compile the full model
                )
                print("✅ Model compiled successfully!")
            except Exception as e:
                print(f"⚠️  Compilation failed: {e}. Using non-compiled model.")

        # Enable optimized attention
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)  # Disable math attention
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # Disable memory efficient attention

    def _create_optimized_optimizer(self):
        """Create optimizer with H100-specific optimizations."""
        # Separate parameters for different weight decay schedules
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ["bias", "norm", "embedding"]):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer_groups = [
            {
                "params": decay_params,
                "weight_decay": self.config.weight_decay,
                "lr": self.config.learning_rate,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
                "lr": self.config.learning_rate,
            },
        ]

        # Use fused AdamW for H100 performance
        # Note: fused and foreach cannot both be True simultaneously
        optimizer_kwargs = {
            "lr": self.config.learning_rate,
            "betas": (self.config.beta1, self.config.beta2),
            "eps": self.config.eps,
        }

        # Try optimizations in order of preference for H100
        optimizer = None

        if self.config.use_fused_adamw:
            # First try: Fused AdamW (best for H100)
            try:
                optimizer = AdamW(
                    optimizer_groups,
                    fused=True,
                    foreach=False,  # Must be False when fused=True
                    **optimizer_kwargs,
                )
                print("✅ Using fused AdamW optimizer (optimal for H100)")
            except RuntimeError as e:
                print(f"⚠️  Fused AdamW failed: {e}")

        if optimizer is None:
            # Second try: Foreach AdamW (good fallback)
            try:
                optimizer = AdamW(optimizer_groups, fused=False, foreach=True, **optimizer_kwargs)
                print("✅ Using foreach AdamW optimizer")
            except RuntimeError as e:
                print(f"⚠️  Foreach AdamW failed: {e}")

        if optimizer is None:
            # Final fallback: Standard AdamW (always works)
            print("🔄 Using standard AdamW optimizer (no advanced optimizations)")
            optimizer = AdamW(optimizer_groups, fused=False, foreach=False, **optimizer_kwargs)
            print("✅ Using standard AdamW optimizer")

        # Verify the optimizer was created successfully
        if hasattr(optimizer, "param_groups"):
            total_params = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
            print(f"✅ Created AdamW optimizer with {total_params:,} parameters")

        return optimizer

    def _create_advanced_scheduler(self):
        """Create advanced learning rate scheduler with warmup, cosine decay, and cooldown."""

        def lr_lambda(step):
            # Warmup phase
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps

            # Main training phase (cosine decay)
            main_steps = self.config.max_steps - self.config.warmup_steps - self.config.cooldown_steps
            if step < self.config.max_steps - self.config.cooldown_steps:
                progress = (step - self.config.warmup_steps) / max(1, main_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return (
                    self.config.min_learning_rate / self.config.learning_rate
                    + (1 - self.config.min_learning_rate / self.config.learning_rate) * cosine_decay
                )

            # Cooldown phase (exponential decay)
            else:
                cooldown_progress = (
                    step - (self.config.max_steps - self.config.cooldown_steps)
                ) / self.config.cooldown_steps
                decay_factor = 0.5 ** (cooldown_progress * 2)  # Decay to 25% of min_lr
                return (self.config.min_learning_rate / self.config.learning_rate) * decay_factor

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _setup_advanced_wandb(self):
        """Setup W&B with comprehensive tracking."""
        # Calculate expected performance
        expected_tokens_per_sec = 900_000  # Based on H100 FP8 performance
        expected_mfu = 85.0  # Model FLOPS Utilization target

        wandb.init(
            project=self.config.project_name,
            name=self.config.model_name,
            config={
                **self.config.__dict__,
                "expected_tokens_per_sec": expected_tokens_per_sec,
                "expected_mfu": expected_mfu,
                "gpu_name": torch.cuda.get_device_name(0),
                "compute_capability": torch.cuda.get_device_capability(0),
            },
            tags=["h100", "fp8", "optimized", "2025"],
        )

        # Watch model with detailed logging
        wandb.watch(self.model, log="all", log_freq=500)

    def _calculate_mfu(self, tokens_per_sec: float) -> float:
        """Calculate Model FLOPs Utilization (MFU) for H100."""
        # H100 theoretical peak: ~1000 TFLOPs for FP8
        h100_peak_tflops = 1000e12  # 1000 TFLOPs in FP8

        # Estimate FLOPs per token for our 350M model
        # Transformer FLOP calculation: 6 * N * L (N=params, L=seq_len)
        params = 350e6  # 350M parameters
        seq_len = self.config.max_length
        flops_per_token = 6 * params * seq_len

        # Calculate actual FLOP/s
        actual_flops = tokens_per_sec * flops_per_token

        # MFU = actual FLOP/s / peak FLOP/s
        mfu = (actual_flops / h100_peak_tflops) * 100
        return min(mfu, 100.0)  # Cap at 100%

    def _get_comprehensive_gpu_stats(self) -> Dict[str, float]:
        """Get comprehensive GPU statistics."""
        stats = {}

        if not NVML_AVAILABLE:
            return stats

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats.update(
                {
                    "gpu_memory_used_gb": mem_info.used / 1e9,
                    "gpu_memory_total_gb": mem_info.total / 1e9,
                    "gpu_memory_percent": (mem_info.used / mem_info.total) * 100,
                }
            )

            # Utilization
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats.update(
                {
                    "gpu_utilization": util_info.gpu,
                    "memory_utilization": util_info.memory,
                }
            )

            # Temperature and power
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                stats.update(
                    {
                        "gpu_temperature_c": temp,
                        "gpu_power_watts": power,
                    }
                )
            except:
                pass  # Some stats might not be available

        except Exception as e:
            print(f"Warning: Could not get GPU stats: {e}")

        return stats

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Optimized training step with H100-specific improvements."""
        step_start = time.time()

        # Move batch to device with non-blocking transfer
        input_ids = batch["input_ids"].to(self.device, non_blocking=self.config.non_blocking)
        labels = batch["labels"].to(self.device, non_blocking=self.config.non_blocking)

        # Forward pass with appropriate precision
        if self.config.use_fp8:
            # FP8 forward pass (no autocast needed with TorchAO)
            outputs = self.model(input_ids=input_ids, labels=labels)
        elif self.scaler is not None:
            # FP16 mixed precision
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(input_ids=input_ids, labels=labels)
        else:
            # Regular precision
            outputs = self.model(input_ids=input_ids, labels=labels)

        loss = outputs["loss"]

        # Scale loss by gradient accumulation steps
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Track step time and throughput
        step_time = time.time() - step_start
        self.step_times.append(step_time)

        # Keep only recent step times
        if len(self.step_times) > 100:
            self.step_times = self.step_times[-100:]

        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "step_time": step_time,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Fast evaluation optimized for H100."""
        self.model.eval()

        total_loss = 0
        total_tokens = 0
        eval_steps = 0
        max_eval_steps = 100  # Limit evaluation time

        eval_start = time.time()

        for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
            if eval_steps >= max_eval_steps:
                break

            input_ids = batch["input_ids"].to(self.device, non_blocking=self.config.non_blocking)
            labels = batch["labels"].to(self.device, non_blocking=self.config.non_blocking)

            # Forward pass (use same precision as training)
            if self.config.use_fp8:
                outputs = self.model(input_ids=input_ids, labels=labels)
            elif self.scaler is not None:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = self.model(input_ids=input_ids, labels=labels)
            else:
                outputs = self.model(input_ids=input_ids, labels=labels)

            loss = outputs["loss"]

            batch_size, seq_len = input_ids.shape
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len
            eval_steps += 1

        eval_time = time.time() - eval_start
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")

        self.model.train()

        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
            "eval_time": eval_time,
            "eval_steps": eval_steps,
        }

    def save_checkpoint(self, path: Optional[str] = None):
        """Save optimized checkpoint."""
        if path is None:
            path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}.pt")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "tokens_processed": self.tokens_processed,
            "config": self.config,
            "performance_stats": {
                "avg_step_time": sum(self.step_times) / len(self.step_times) if self.step_times else 0,
                "throughput_history": self.throughput_history[-100:],  # Keep recent history
            },
        }

        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint to {path}")

    def train(self, max_steps: Optional[int] = None):
        """Main training loop with H100 optimizations."""
        if max_steps is not None:
            self.config.max_steps = max_steps

        # Calculate steps for 1.5 hours if not specified
        if self.config.max_steps is None:
            # More aggressive estimate based on improved performance
            target_tokens_per_sec = 500_000  # Conservative estimate
            total_seconds = 1.5 * 3600  # 1.5 hours
            total_tokens = target_tokens_per_sec * total_seconds
            tokens_per_step = self.config.batch_size * self.config.gradient_accumulation_steps * self.config.max_length
            self.config.max_steps = int(total_tokens / tokens_per_step)

        print(f"🚀 Starting H100-optimized training for {self.config.max_steps} steps...")
        print(f"📊 Target effective batch size: {self.config.effective_batch_size_tokens:,} tokens")
        print(f"🎯 Expected performance: >500K tokens/sec, >80% MFU")

        # Training loop
        self.model.train()
        self.start_time = time.time()

        # Pre-compile with a dummy batch to avoid first-step slowdown
        if self.config.use_compile:
            print("🔥 Warming up compiled model...")
            dummy_batch = {
                "input_ids": torch.randint(0, 1000, (self.config.batch_size, self.config.max_length)).to(self.device),
                "labels": torch.randint(0, 1000, (self.config.batch_size, self.config.max_length)).to(self.device),
            }
            with torch.no_grad():
                _ = self.model(input_ids=dummy_batch["input_ids"], labels=dummy_batch["labels"])
            print("✅ Model warmed up!")

        while self.global_step < self.config.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Training step
                step_metrics = self.train_step(batch)

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
                    step_tokens = (
                        self.config.batch_size * self.config.gradient_accumulation_steps * self.config.max_length
                    )
                    self.tokens_processed += step_tokens

                    # Calculate performance metrics
                    elapsed_time = time.time() - self.start_time
                    tokens_per_sec = self.tokens_processed / elapsed_time
                    mfu = self._calculate_mfu(tokens_per_sec)

                    self.throughput_history.append(tokens_per_sec)
                    if len(self.throughput_history) > 1000:
                        self.throughput_history = self.throughput_history[-1000:]

                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        log_metrics = {
                            "train_loss": step_metrics["loss"],
                            "learning_rate": self.scheduler.get_last_lr()[0],
                            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "tokens_per_second": tokens_per_sec,
                            "mfu_percent": mfu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "step_time": step_metrics["step_time"],
                        }

                        # Add GPU stats
                        log_metrics.update(self._get_comprehensive_gpu_stats())

                        # Log to wandb
                        if self.config.use_wandb:
                            wandb.log(log_metrics)

                        # Enhanced console logging
                        print(
                            f"Step {self.global_step}: "
                            f"loss={step_metrics['loss']:.4f}, "
                            f"lr={log_metrics['learning_rate']:.2e}, "
                            f"tokens/sec={tokens_per_sec:.0f}, "
                            f"MFU={mfu:.1f}%, "
                            f"mem={log_metrics.get('gpu_memory_percent', 0):.1f}%"
                        )

                    # Evaluation
                    if self.global_step % self.config.eval_interval == 0:
                        print(f"\n🔍 Running evaluation at step {self.global_step}...")
                        eval_metrics = self.evaluate()

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

                        # Check if we've beaten the target
                        if eval_metrics["val_loss"] < 3.0781:
                            print(f"🎉 TARGET ACHIEVED! Validation loss {eval_metrics['val_loss']:.4f} < 3.0781")

                    # Save checkpoint
                    if self.global_step % self.config.save_interval == 0:
                        self.save_checkpoint()

                    # Check if done
                    if self.global_step >= self.config.max_steps:
                        break

            self.epoch += 1

        # Final evaluation
        print("\n🏁 Training complete! Running final evaluation...")
        final_metrics = self.evaluate()
        print(f"🎯 Final validation loss: {final_metrics['val_loss']:.4f}")
        print(f"📊 Final validation perplexity: {final_metrics['val_perplexity']:.2f}")

        # Final performance summary
        final_tokens_per_sec = self.tokens_processed / (time.time() - self.start_time)
        final_mfu = self._calculate_mfu(final_tokens_per_sec)

        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"   Tokens processed: {self.tokens_processed:,}")
        print(f"   Average throughput: {final_tokens_per_sec:.0f} tokens/sec")
        print(f"   Model FLOPs Utilization: {final_mfu:.1f}%")
        print(f"   Training time: {(time.time() - self.start_time) / 3600:.2f} hours")

        if final_metrics["val_loss"] < 3.0781:
            improvement = 3.0781 - final_metrics["val_loss"]
            print(f"✅ SUCCESS! Beat target by {improvement:.4f}")
        else:
            deficit = final_metrics["val_loss"] - 3.0781
            print(f"❌ Missed target by {deficit:.4f}")

        # Save final model
        final_path = os.path.join(self.config.output_dir, "final_model.pt")
        self.save_checkpoint(final_path)

        # Close wandb
        if self.config.use_wandb:
            wandb.finish()

        return final_metrics
