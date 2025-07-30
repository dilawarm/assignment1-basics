"""
Advanced gradient clipping utilities with state-of-the-art adaptive methods.

This module provides:
- ZClip: Adaptive gradient clipping with z-score anomaly detection (2025)
- AdaGC: Parameter-specific adaptive gradient clipping with EMA (2025)
- Traditional gradient clipping with robust device handling
- Proactive loss spike detection and recovery
"""

import math
import warnings
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class ZClipGradientClipper:
    """
    ZClip: Adaptive gradient clipping using z-score anomaly detection.

    Based on "ZClip: Adaptive Spike Mitigation for LLM Pre-Training" (2025).
    Dynamically adjusts clipping threshold based on statistical properties
    of gradient norms to prevent malignant loss spikes.
    """

    def __init__(
        self,
        history_size: int = 100,
        zscore_threshold: float = 3.0,
        min_clip_value: float = 0.1,
        max_clip_value: float = 10.0,
        warmup_steps: int = 100,
    ):
        self.history_size = history_size
        self.zscore_threshold = zscore_threshold
        self.min_clip_value = min_clip_value
        self.max_clip_value = max_clip_value
        self.warmup_steps = warmup_steps

        self.grad_norm_history = []
        self.step_count = 0
        self.current_threshold = max_clip_value

    def clip_gradients(self, parameters: Iterable[torch.nn.Parameter]) -> Tuple[float, float, bool]:
        """
        Apply ZClip adaptive gradient clipping.

        Returns:
            Tuple of (global_norm, clip_threshold, was_clipped)
        """
        self.step_count += 1
        param_list = list(parameters)

        # Compute global gradient norm
        global_norm = self._compute_global_norm(param_list)

        if math.isnan(global_norm) or math.isinf(global_norm):
            warnings.warn(f"NaN/Inf gradient norm detected at step {self.step_count}")
            return global_norm, self.current_threshold, False

        # Update history
        self.grad_norm_history.append(global_norm)
        if len(self.grad_norm_history) > self.history_size:
            self.grad_norm_history.pop(0)

        # Use warmup phase with conservative clipping
        if self.step_count <= self.warmup_steps:
            self.current_threshold = self.min_clip_value * 2.0
            was_clipped = self._apply_clipping(param_list, global_norm, self.current_threshold)
            return global_norm, self.current_threshold, was_clipped

        # Adaptive threshold based on z-score anomaly detection
        if len(self.grad_norm_history) >= 10:
            mean_norm = np.mean(self.grad_norm_history)
            std_norm = np.std(self.grad_norm_history) + 1e-8

            # Detect anomaly using z-score
            z_score = (global_norm - mean_norm) / std_norm

            if z_score > self.zscore_threshold:
                # Adaptive threshold based on recent statistics
                self.current_threshold = mean_norm + self.zscore_threshold * std_norm
                self.current_threshold = max(self.min_clip_value, min(self.current_threshold, self.max_clip_value))
            else:
                # Gradually increase threshold when stable
                self.current_threshold = min(self.current_threshold * 1.01, self.max_clip_value)

        was_clipped = self._apply_clipping(param_list, global_norm, self.current_threshold)
        return global_norm, self.current_threshold, was_clipped

    def _compute_global_norm(self, param_list) -> float:
        """Compute global gradient norm."""
        total_norm_squared = 0.0
        device = None

        for param in param_list:
            if param.grad is not None:
                if device is None:
                    device = param.grad.device

                param_norm = torch.norm(param.grad.data, dtype=torch.float32)
                total_norm_squared += param_norm.item() ** 2

        return math.sqrt(total_norm_squared + 1e-8)

    def _apply_clipping(self, param_list, global_norm: float, threshold: float) -> bool:
        """Apply gradient clipping if needed."""
        if global_norm > threshold:
            clip_coeff = threshold / global_norm
            for param in param_list:
                if param.grad is not None:
                    param.grad.data.mul_(clip_coeff)
            return True
        return False


class AdaGCGradientClipper:
    """
    AdaGC: Adaptive Gradient Clipping with parameter-specific thresholds.

    Based on "AdaGC: Improving Training Stability for Large Language Model Pretraining" (2025).
    Maintains per-parameter gradient norm statistics using exponential moving averages.
    """

    def __init__(
        self,
        beta: float = 0.99,
        eps: float = 1e-8,
        init_threshold: float = 1.0,
        threshold_scale: float = 2.0,
    ):
        self.beta = beta
        self.eps = eps
        self.init_threshold = init_threshold
        self.threshold_scale = threshold_scale

        self.param_stats = {}
        self.step_count = 0

    def clip_gradients(self, parameters: Iterable[torch.nn.Parameter]) -> Tuple[float, Dict[str, float], int]:
        """
        Apply AdaGC adaptive gradient clipping.

        Returns:
            Tuple of (global_norm, param_thresholds, num_clipped)
        """
        self.step_count += 1
        param_list = list(parameters)

        global_norm_squared = 0.0
        param_thresholds = {}
        num_clipped = 0

        for i, param in enumerate(param_list):
            if param.grad is None:
                continue

            param_id = f"param_{i}"
            param_norm = torch.norm(param.grad.data, dtype=torch.float32).item()

            # Initialize statistics if needed
            if param_id not in self.param_stats:
                self.param_stats[param_id] = {
                    "ema_norm": param_norm,
                    "ema_norm_sq": param_norm**2,
                }

            # Update EMA statistics
            stats = self.param_stats[param_id]
            stats["ema_norm"] = self.beta * stats["ema_norm"] + (1 - self.beta) * param_norm
            stats["ema_norm_sq"] = self.beta * stats["ema_norm_sq"] + (1 - self.beta) * (param_norm**2)

            # Bias correction
            bias_correction_1 = 1 - self.beta**self.step_count
            bias_correction_2 = 1 - self.beta**self.step_count

            corrected_mean = stats["ema_norm"] / bias_correction_1
            corrected_second_moment = stats["ema_norm_sq"] / bias_correction_2

            # Adaptive threshold
            variance = corrected_second_moment - corrected_mean**2
            std = math.sqrt(max(variance, 0) + self.eps)
            threshold = corrected_mean + self.threshold_scale * std
            threshold = max(threshold, self.init_threshold)

            param_thresholds[param_id] = threshold

            # Apply clipping if needed
            if param_norm > threshold:
                clip_coeff = threshold / param_norm
                param.grad.data.mul_(clip_coeff)
                param_norm = threshold
                num_clipped += 1

            global_norm_squared += param_norm**2

        global_norm = math.sqrt(global_norm_squared + 1e-8)
        return global_norm, param_thresholds, num_clipped


class StabilityMonitor:
    """
    Advanced stability monitoring with proactive loss spike detection.
    """

    def __init__(
        self,
        loss_history_size: int = 50,
        spike_threshold: float = 2.0,
        nan_tolerance: int = 3,
        recovery_lr_scale: float = 0.1,
    ):
        self.loss_history_size = loss_history_size
        self.spike_threshold = spike_threshold
        self.nan_tolerance = nan_tolerance
        self.recovery_lr_scale = recovery_lr_scale

        self.loss_history = []
        self.nan_count = 0
        self.spike_count = 0
        self.last_stable_state = None

    def update(
        self, loss: float, grad_norm: float, step: int, model_state: Optional[Dict] = None
    ) -> Tuple[bool, Dict, Optional[Dict]]:
        """
        Update stability monitoring.

        Returns:
            Tuple of (should_continue, stats, recovery_state)
        """
        # Check for NaN
        if math.isnan(loss) or math.isnan(grad_norm):
            self.nan_count += 1
            if self.nan_count > self.nan_tolerance:
                return False, self._get_stats(), self.last_stable_state
        else:
            # Reset NaN count on stable step
            self.nan_count = 0

            # Save stable state
            if model_state is not None and len(self.loss_history) > 10:
                recent_avg = np.mean(self.loss_history[-5:])
                if not math.isnan(recent_avg) and recent_avg < float("inf"):
                    self.last_stable_state = {
                        "step": step,
                        "loss": loss,
                        "model_state": model_state.copy() if hasattr(model_state, "copy") else model_state,
                        "lr_scale": 1.0,
                    }

        # Update loss history
        if not math.isnan(loss):
            self.loss_history.append(loss)
            if len(self.loss_history) > self.loss_history_size:
                self.loss_history.pop(0)

        # Detect loss spikes
        if len(self.loss_history) >= 10:
            recent_avg = np.mean(self.loss_history[-3:])
            baseline_avg = np.mean(self.loss_history[-10:-3])

            if recent_avg > baseline_avg * self.spike_threshold:
                self.spike_count += 1

                # Prepare recovery state with reduced learning rate
                recovery_state = None
                if self.last_stable_state is not None:
                    recovery_state = self.last_stable_state.copy()
                    recovery_state["lr_scale"] = self.recovery_lr_scale

        return True, self._get_stats(), None

    def _get_stats(self) -> Dict:
        """Get monitoring statistics."""
        return {
            "nan_count": self.nan_count,
            "spike_count": self.spike_count,
            "avg_recent_loss": float(np.mean(self.loss_history[-5:]) if len(self.loss_history) >= 5 else 0.0),
            "loss_trend": float(
                np.mean(self.loss_history[-3:]) - np.mean(self.loss_history[-6:-3])
                if len(self.loss_history) >= 6
                else 0.0
            ),
        }


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, device: Optional[torch.device] = None
) -> float:
    """
    Traditional gradient clipping by global L2 norm with robust device handling.

    Args:
        parameters: Model parameters with gradients
        max_l2_norm: Maximum L2 norm for clipping
        device: Device for computation (auto-detected if None)

    Returns:
        Global gradient norm before clipping
    """
    eps = 1e-8

    param_list = list(parameters)

    if device is None:
        for param in param_list:
            if param.grad is not None:
                device = param.grad.device
                break
        else:
            device = torch.device("cpu")

    total_norm_squared = torch.tensor(0.0, device=device, dtype=torch.float32)
    param_count = 0

    for param in param_list:
        if param.grad is not None:
            if param.grad.device != device:
                param.grad = param.grad.to(device=device)

            param_norm = torch.norm(param.grad.data, dtype=torch.float32)
            total_norm_squared += param_norm**2
            param_count += 1

    if param_count == 0:
        return 0.0

    global_norm = torch.sqrt(total_norm_squared + eps)

    if global_norm > max_l2_norm:
        clip_coeff = max_l2_norm / global_norm
        for param in param_list:
            if param.grad is not None:
                param.grad.data.mul_(clip_coeff)

    return global_norm.item()


def create_adaptive_clipper(method: str = "zclip", **kwargs) -> object:
    """
    Create an adaptive gradient clipper.

    Args:
        method: "zclip" or "adagc"
        **kwargs: Method-specific parameters

    Returns:
        Gradient clipper instance
    """
    if method.lower() == "zclip":
        return ZClipGradientClipper(**kwargs)
    elif method.lower() == "adagc":
        return AdaGCGradientClipper(**kwargs)
    else:
        raise ValueError(f"Unknown adaptive clipping method: {method}")


def safe_gradient_step(
    optimizer: torch.optim.Optimizer,
    clipper: object,
    parameters: Iterable[torch.nn.Parameter],
    fallback_clip: float = 1.0,
) -> Tuple[float, bool]:
    """
    Perform a safe gradient step with adaptive clipping and fallback.

    Args:
        optimizer: PyTorch optimizer
        clipper: Adaptive gradient clipper (ZClip or AdaGC)
        parameters: Model parameters
        fallback_clip: Fallback clipping value if adaptive fails

    Returns:
        Tuple of (gradient_norm, was_clipped)
    """
    try:
        if hasattr(clipper, "clip_gradients"):
            if isinstance(clipper, ZClipGradientClipper):
                grad_norm, threshold, was_clipped = clipper.clip_gradients(parameters)
            elif isinstance(clipper, AdaGCGradientClipper):
                grad_norm, thresholds, num_clipped = clipper.clip_gradients(parameters)
                was_clipped = num_clipped > 0
            else:
                grad_norm = gradient_clipping(parameters, fallback_clip)
                was_clipped = grad_norm > fallback_clip
        else:
            grad_norm = gradient_clipping(parameters, fallback_clip)
            was_clipped = grad_norm > fallback_clip

        # Check for problematic gradients
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            warnings.warn(f"Problematic gradient norm: {grad_norm}, skipping step")
            return grad_norm, False

        optimizer.step()
        return grad_norm, was_clipped

    except Exception as e:
        warnings.warn(f"Gradient step failed: {e}, applying fallback clipping")
        grad_norm = gradient_clipping(parameters, fallback_clip)
        optimizer.step()
        return grad_norm, True
