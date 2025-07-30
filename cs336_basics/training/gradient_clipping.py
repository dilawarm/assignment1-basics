"""
Gradient clipping utilities with advanced AdaGC support.

This module provides both traditional gradient clipping and Adaptive Gradient Clipping (AdaGC)
which tracks per-parameter gradient statistics for more stable training.
"""

from typing import Iterable

import torch


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, device: torch.device | None = None
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
