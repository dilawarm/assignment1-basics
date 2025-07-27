"""
Advanced Gradient Clipping Techniques for Training Stability

This module implements state-of-the-art gradient clipping methods based on:
- ZClip: Adaptive Spike Mitigation for LLM Pre-Training (2025)
- AdaGC: Improving Training Stability for Large Language Model Pretraining (2025)
- Best practices for preventing NaN/inf gradients in deep learning
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_


class ZClip:
    """
    ZClip: Adaptive gradient clipping using z-score based anomaly detection.

    Proactively detects and mitigates gradient spikes before they cause training instability.
    Based on "ZClip: Adaptive Spike Mitigation for LLM Pre-Training" (2025).
    """

    def __init__(
        self,
        model: nn.Module,
        window_size: int = 200,
        z_threshold: float = 3.0,
        min_threshold: float = 0.1,
        max_threshold: float = 10.0,
        ema_decay: float = 0.99,
        warmup_steps: int = 100,
        enable_logging: bool = True,
    ):
        """
        Initialize ZClip adaptive gradient clipping.

        Args:
            model: The model to apply gradient clipping to
            window_size: Size of the sliding window for gradient norm history
            z_threshold: Z-score threshold for anomaly detection
            min_threshold: Minimum clipping threshold
            max_threshold: Maximum clipping threshold
            ema_decay: Exponential moving average decay factor
            warmup_steps: Number of warmup steps before full ZClip activation
            enable_logging: Whether to enable detailed logging
        """
        self.model = model
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self.enable_logging = enable_logging

        # Gradient norm history
        self.grad_norm_history = deque(maxlen=window_size)

        # Statistics
        self.mean_grad_norm = 0.0
        self.var_grad_norm = 1.0
        self.ema_mean = 0.0
        self.ema_var = 1.0

        # Counters
        self.step_count = 0
        self.spike_count = 0
        self.total_clips = 0

        # Current adaptive threshold
        self.current_threshold = max_threshold

    def compute_grad_norm(self, parameters: Union[torch.Tensor, List[torch.Tensor]]) -> float:
        """Compute the global gradient norm."""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        total_norm = 0.0
        param_count = 0

        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(dtype=torch.float32)
                total_norm += param_norm.item() ** 2
                param_count += 1

        if param_count == 0:
            return 0.0

        return math.sqrt(total_norm)

    def update_statistics(self, grad_norm: float) -> None:
        """Update gradient norm statistics."""
        self.grad_norm_history.append(grad_norm)

        # Update EMA statistics
        if self.step_count == 0:
            self.ema_mean = grad_norm
            self.ema_var = 1.0
        else:
            delta = grad_norm - self.ema_mean
            self.ema_mean += (1 - self.ema_decay) * delta
            self.ema_var = self.ema_decay * self.ema_var + (1 - self.ema_decay) * delta * delta

        # Update window-based statistics if we have enough history
        if len(self.grad_norm_history) >= 20:
            history_list = list(self.grad_norm_history)
            self.mean_grad_norm = sum(history_list) / len(history_list)
            self.var_grad_norm = sum((x - self.mean_grad_norm) ** 2 for x in history_list) / len(history_list)

    def detect_anomaly(self, grad_norm: float) -> bool:
        """Detect if current gradient norm is anomalous using z-score."""
        if self.step_count < self.warmup_steps or len(self.grad_norm_history) < 20:
            return False

        # Use EMA statistics for more responsive detection
        std_dev = math.sqrt(max(self.ema_var, 1e-8))
        z_score = abs(grad_norm - self.ema_mean) / std_dev

        return z_score > self.z_threshold

    def compute_adaptive_threshold(self, grad_norm: float) -> float:
        """Compute adaptive clipping threshold."""
        if self.step_count < self.warmup_steps:
            return self.max_threshold

        if self.detect_anomaly(grad_norm):
            # Use more conservative threshold for anomalies
            threshold = max(self.ema_mean + 2 * math.sqrt(self.ema_var), self.min_threshold)
            self.spike_count += 1
        else:
            # Use more permissive threshold for normal gradients
            threshold = max(self.ema_mean + 3 * math.sqrt(self.ema_var), self.min_threshold)

        # Clamp to reasonable bounds
        threshold = min(max(threshold, self.min_threshold), self.max_threshold)

        # Smooth threshold changes
        if hasattr(self, "current_threshold"):
            alpha = 0.1  # Smoothing factor
            threshold = alpha * threshold + (1 - alpha) * self.current_threshold

        return threshold

    def clip_gradients(self, parameters: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[float, float]:
        """
        Apply ZClip adaptive gradient clipping.

        Returns:
            Tuple of (original_grad_norm, clipped_grad_norm)
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        # Compute current gradient norm
        grad_norm = self.compute_grad_norm(parameters)

        # Update statistics
        self.update_statistics(grad_norm)

        # Compute adaptive threshold
        threshold = self.compute_adaptive_threshold(grad_norm)
        self.current_threshold = threshold

        # Apply clipping if necessary
        clipped_grad_norm = grad_norm
        if grad_norm > threshold:
            clip_grad_norm_(parameters, threshold)
            clipped_grad_norm = threshold
            self.total_clips += 1

            if self.enable_logging and self.step_count % 100 == 0:
                print(
                    f"ZClip: Clipped grad_norm {grad_norm:.3f} -> {threshold:.3f} "
                    f"(z-score: {abs(grad_norm - self.ema_mean) / math.sqrt(self.ema_var):.2f})"
                )

        self.step_count += 1
        return grad_norm, clipped_grad_norm

    def get_stats(self) -> Dict[str, float]:
        """Get clipping statistics."""
        spike_rate = self.spike_count / max(self.step_count, 1)
        clip_rate = self.total_clips / max(self.step_count, 1)

        return {
            "zclip_threshold": self.current_threshold,
            "zclip_mean_grad_norm": self.ema_mean,
            "zclip_grad_norm_std": math.sqrt(self.ema_var),
            "zclip_spike_rate": spike_rate,
            "zclip_clip_rate": clip_rate,
            "zclip_total_spikes": self.spike_count,
            "zclip_total_clips": self.total_clips,
        }


class AdaptiveGradientClipper:
    """
    Enhanced AdaGC: Per-parameter adaptive gradient clipping.

    Based on "AdaGC: Improving Training Stability for Large Language Model Pretraining" (2025).
    Maintains separate adaptive thresholds for each parameter group.
    """

    def __init__(
        self,
        model: nn.Module,
        max_global_norm: float = 1.0,
        beta: float = 0.98,
        eps: float = 1e-8,
        per_param_clipping: bool = True,
        device: Optional[torch.device] = None,
        enable_logging: bool = True,
    ):
        """
        Initialize enhanced AdaGC.

        Args:
            model: The model to apply gradient clipping to
            max_global_norm: Maximum global gradient norm
            beta: EMA decay factor for adaptive thresholds
            eps: Small constant for numerical stability
            per_param_clipping: Whether to use per-parameter adaptive clipping
            device: Device to run computations on
            enable_logging: Whether to enable detailed logging
        """
        self.model = model
        self.max_global_norm = max_global_norm
        self.beta = beta
        self.eps = eps
        self.per_param_clipping = per_param_clipping
        self.device = device or next(model.parameters()).device
        self.enable_logging = enable_logging

        # Parameter-specific adaptive thresholds
        self.param_ema_norms = {}
        self.param_thresholds = {}
        self.step_count = 0
        self.total_clips = 0

        # Initialize parameter tracking
        self._initialize_param_tracking()

    def _initialize_param_tracking(self) -> None:
        """Initialize per-parameter tracking."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_ema_norms[name] = 0.0
                self.param_thresholds[name] = self.max_global_norm

    def _update_param_threshold(self, name: str, grad_norm: float) -> float:
        """Update adaptive threshold for a specific parameter."""
        if name not in self.param_ema_norms:
            self.param_ema_norms[name] = grad_norm
            self.param_thresholds[name] = self.max_global_norm
        else:
            # Update EMA of gradient norm
            self.param_ema_norms[name] = self.beta * self.param_ema_norms[name] + (1 - self.beta) * grad_norm

            # Compute adaptive threshold
            ema_norm = self.param_ema_norms[name]

            # Use a multiple of the EMA norm as threshold, bounded by global max
            adaptive_threshold = min(max(ema_norm * 2.0, self.max_global_norm * 0.1), self.max_global_norm)

            self.param_thresholds[name] = adaptive_threshold

        return self.param_thresholds[name]

    def clip_gradients(self, model: Optional[nn.Module] = None) -> float:
        """
        Apply enhanced AdaGC gradient clipping.

        Returns:
            The computed gradient norm after clipping
        """
        if model is None:
            model = self.model

        parameters = [p for p in model.parameters() if p.grad is not None]

        if len(parameters) == 0:
            return 0.0

        # Compute global gradient norm
        total_norm = 0.0
        param_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(dtype=torch.float32).item()
                param_norms[name] = param_norm
                total_norm += param_norm**2

        total_norm = math.sqrt(total_norm)

        if self.per_param_clipping:
            # Apply per-parameter adaptive clipping
            any_clipped = False

            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param_norms[name]
                    threshold = self._update_param_threshold(name, param_norm)

                    if param_norm > threshold:
                        scaling_factor = threshold / (param_norm + self.eps)
                        param.grad.data.mul_(scaling_factor)
                        any_clipped = True

            if any_clipped:
                self.total_clips += 1

        # Apply global clipping as final safety net
        if total_norm > self.max_global_norm:
            clip_grad_norm_(parameters, self.max_global_norm)
            final_norm = self.max_global_norm
            self.total_clips += 1
        else:
            final_norm = total_norm

        self.step_count += 1

        if self.enable_logging and self.step_count % 100 == 0:
            avg_threshold = sum(self.param_thresholds.values()) / len(self.param_thresholds)
            print(f"AdaGC: Global norm {total_norm:.3f}, Avg param threshold {avg_threshold:.3f}")

        return final_norm

    def get_stats(self) -> Dict[str, float]:
        """Get clipping statistics."""
        if self.param_thresholds:
            avg_threshold = sum(self.param_thresholds.values()) / len(self.param_thresholds)
            max_threshold = max(self.param_thresholds.values())
            min_threshold = min(self.param_thresholds.values())
        else:
            avg_threshold = max_threshold = min_threshold = self.max_global_norm

        clip_rate = self.total_clips / max(self.step_count, 1)

        return {
            "adagc_avg_threshold": avg_threshold,
            "adagc_max_threshold": max_threshold,
            "adagc_min_threshold": min_threshold,
            "adagc_clip_rate": clip_rate,
            "adagc_total_clips": self.total_clips,
        }


class HybridGradientClipper:
    """
    Hybrid gradient clipping combining ZClip and AdaGC for maximum stability.

    Uses ZClip for anomaly detection and spike prevention, and AdaGC for
    fine-grained per-parameter control.
    """

    def __init__(
        self,
        model: nn.Module,
        zclip_config: Optional[Dict] = None,
        adagc_config: Optional[Dict] = None,
        enable_logging: bool = True,
    ):
        """
        Initialize hybrid gradient clipping.

        Args:
            model: The model to apply gradient clipping to
            zclip_config: Configuration for ZClip
            adagc_config: Configuration for AdaGC
            enable_logging: Whether to enable detailed logging
        """
        self.model = model
        self.enable_logging = enable_logging

        # Initialize ZClip with default or provided config
        zclip_defaults = {
            "window_size": 200,
            "z_threshold": 2.5,  # More conservative for stability
            "min_threshold": 0.1,
            "max_threshold": 5.0,  # Lower max for better stability
            "ema_decay": 0.99,
            "warmup_steps": 100,
            "enable_logging": enable_logging,
        }
        zclip_config = {**zclip_defaults, **(zclip_config or {})}
        self.zclip = ZClip(model, **zclip_config)

        # Initialize AdaGC with default or provided config
        adagc_defaults = {
            "max_global_norm": 1.0,
            "beta": 0.98,
            "eps": 1e-8,
            "per_param_clipping": True,
            "enable_logging": enable_logging,
        }
        adagc_config = {**adagc_defaults, **(adagc_config or {})}
        self.adagc = AdaptiveGradientClipper(model, **adagc_config)

        self.step_count = 0

    def clip_gradients(self, model: Optional[nn.Module] = None) -> Tuple[float, float]:
        """
        Apply hybrid gradient clipping.

        Returns:
            Tuple of (original_norm, final_norm)
        """
        if model is None:
            model = self.model

        parameters = [p for p in model.parameters() if p.grad is not None]

        if len(parameters) == 0:
            return 0.0, 0.0

        # First apply ZClip for anomaly detection and spike prevention
        original_norm, zclip_norm = self.zclip.clip_gradients(parameters)

        # Then apply AdaGC for fine-grained per-parameter control
        final_norm = self.adagc.clip_gradients(model)

        self.step_count += 1

        if self.enable_logging and self.step_count % 100 == 0:
            print(f"Hybrid Clipping: {original_norm:.3f} -> ZClip: {zclip_norm:.3f} -> AdaGC: {final_norm:.3f}")

        return original_norm, final_norm

    def get_stats(self) -> Dict[str, float]:
        """Get comprehensive clipping statistics."""
        stats = {}
        stats.update(self.zclip.get_stats())
        stats.update(self.adagc.get_stats())
        stats["hybrid_step_count"] = self.step_count
        return stats


def advanced_gradient_clipping(
    model: nn.Module,
    max_global_norm: float = 1.0,
    use_adaptive: bool = True,
    clipper: Optional[Union[ZClip, AdaptiveGradientClipper, HybridGradientClipper]] = None,
) -> float:
    """
    Advanced gradient clipping function with multiple strategies.

    Args:
        model: The model to apply gradient clipping to
        max_global_norm: Maximum global gradient norm for fallback clipping
        use_adaptive: Whether to use adaptive clipping techniques
        clipper: Pre-initialized clipper instance

    Returns:
        The final gradient norm after clipping
    """
    parameters = [p for p in model.parameters() if p.grad is not None]

    if len(parameters) == 0:
        return 0.0

    if clipper is not None:
        if isinstance(clipper, HybridGradientClipper):
            _, final_norm = clipper.clip_gradients(model)
            return final_norm
        elif isinstance(clipper, ZClip):
            _, final_norm = clipper.clip_gradients(parameters)
            return final_norm
        elif isinstance(clipper, AdaptiveGradientClipper):
            return clipper.clip_gradients(model)

    # Fallback to standard clipping
    if use_adaptive:
        # Use simple adaptive clipping based on gradient norm history
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in parameters]))
        clip_coef = max_global_norm / (total_norm + 1e-6)

        if clip_coef < 1.0:
            for p in parameters:
                p.grad.detach().mul_(clip_coef)
            return max_global_norm
        else:
            return total_norm.item()
    else:
        # Standard gradient clipping
        return clip_grad_norm_(parameters, max_global_norm)


# Legacy function for backward compatibility
def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    Simple gradient clipping function for backward compatibility.

    Args:
        model: The model to apply gradient clipping to
        max_norm: Maximum gradient norm

    Returns:
        The gradient norm after clipping
    """
    warnings.warn(
        "clip_gradients is deprecated. Use advanced_gradient_clipping instead.", DeprecationWarning, stacklevel=2
    )
    return advanced_gradient_clipping(model, max_norm, use_adaptive=False)
