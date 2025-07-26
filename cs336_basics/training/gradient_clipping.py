"""
Gradient clipping utilities with advanced AdaGC support.

This module provides both traditional gradient clipping and Adaptive Gradient Clipping (AdaGC)
which tracks per-parameter gradient statistics for more stable training.
"""

import math
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


class AdaptiveGradientClipper:
    """
    Adaptive Gradient Clipping (AdaGC) implementation.

    Tracks exponential moving averages of gradient norms per parameter
    and applies adaptive thresholds. Based on arXiv:2502.11034.

    This prevents loss spikes more effectively than global gradient clipping.
    """

    def __init__(
        self,
        model: nn.Module,
        max_global_norm: float = 1.0,
        beta: float = 0.95,
        eps: float = 1e-8,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize AdaGC clipper.

        Args:
            model: Model to clip gradients for
            max_global_norm: Maximum global gradient norm
            beta: Exponential moving average decay
            eps: Small value for numerical stability
            device: Device to place moving averages on
        """
        self.max_global_norm = max_global_norm
        self.beta = beta
        self.eps = eps
        self.device = device or next(model.parameters()).device

        self.moving_averages: Dict[str, torch.Tensor] = {}
        self.step_count = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.moving_averages[name] = torch.zeros(1, device=self.device)

    def update_moving_averages(self, model: nn.Module) -> None:
        """Update moving averages of gradient norms."""
        self.step_count += 1

        for name, param in model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue

            grad = param.grad.to(device=self.device)

            param_grad_norm = torch.norm(grad, dtype=torch.float32)

            if name not in self.moving_averages:
                self.moving_averages[name] = torch.zeros(1, device=self.device)

            self.moving_averages[name] = self.beta * self.moving_averages[name] + (1 - self.beta) * param_grad_norm

    def clip_gradients(self, model: nn.Module) -> float:
        """
        Apply adaptive gradient clipping.

        Returns:
            The global gradient norm before clipping
        """
        self.update_moving_averages(model)

        total_norm_squared = 0.0
        param_count = 0

        for name, param in model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue

            if param.grad.device != self.device:
                param.grad = param.grad.to(device=self.device)

            param_norm = torch.norm(param.grad, dtype=torch.float32)
            total_norm_squared += param_norm.item() ** 2
            param_count += 1

        global_norm = math.sqrt(total_norm_squared) if param_count > 0 else 0.0

        if global_norm > self.max_global_norm:
            clip_coeff = self.max_global_norm / (global_norm + self.eps)

            for name, param in model.named_parameters():
                if param.grad is None or not param.requires_grad:
                    continue

                bias_corrected_ma = self.moving_averages[name] / (1 - self.beta**self.step_count)
                adaptive_threshold = bias_corrected_ma + self.eps

                param_norm = torch.norm(param.grad, dtype=torch.float32)

                param_norm_scalar = param_norm.item()
                adaptive_threshold_scalar = adaptive_threshold.item()

                if param_norm_scalar > adaptive_threshold_scalar:
                    local_clip_coeff = adaptive_threshold_scalar / (param_norm_scalar + self.eps)
                    param.grad.mul_(min(clip_coeff, local_clip_coeff))
                else:
                    param.grad.mul_(clip_coeff)

        return global_norm


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, device: Optional[torch.device] = None
) -> float:
    """
    Clip gradients by global L2 norm with robust device handling.

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


def advanced_gradient_clipping(
    model: nn.Module,
    max_global_norm: float = 1.0,
    use_adaptive: bool = True,
    beta: float = 0.95,
    eps: float = 1e-8,
    adaptive_clipper: Optional[AdaptiveGradientClipper] = None,
) -> float:
    """
    Apply advanced gradient clipping with optional AdaGC.

    Args:
        model: Model to clip gradients for
        max_global_norm: Maximum global gradient norm
        use_adaptive: Whether to use AdaGC
        beta: AdaGC exponential moving average decay
        eps: Small value for numerical stability
        adaptive_clipper: Pre-existing AdaGC clipper to reuse

    Returns:
        Global gradient norm before clipping
    """
    try:
        if adaptive_clipper is not None:
            return adaptive_clipper.clip_gradients(model)
        elif use_adaptive:
            temp_clipper = AdaptiveGradientClipper(model=model, max_global_norm=max_global_norm, beta=beta, eps=eps)
            return temp_clipper.clip_gradients(model)
        else:
            return gradient_clipping(model.parameters(), max_global_norm)
    except Exception as e:
        print(f"Advanced gradient clipping failed, using standard clipping: {e}")
        return gradient_clipping(model.parameters(), max_global_norm)
