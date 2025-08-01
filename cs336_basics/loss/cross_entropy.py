"""
Simple and stable cross-entropy loss using standard PyTorch implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int


def cross_entropy(
    logits: Float[torch.Tensor, "... vocab_size"], targets: Int[torch.Tensor, "..."]
) -> Float[torch.Tensor, ""]:
    """
    Compute cross-entropy loss using standard PyTorch implementation.

    Args:
        logits: Logits tensor of shape (..., vocab_size)
        targets: Target indices tensor of shape (...)

    Returns:
        Scalar cross-entropy loss tensor

    Raises:
        ValueError: If tensor shapes are incompatible
    """
    if logits.dim() < 1:
        raise ValueError(f"Logits must have at least 1 dimension, got {logits.dim()}")

    vocab_size = logits.shape[-1]
    expected_shape = logits.shape[:-1]

    if targets.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch: logits {logits.shape} implies targets should have shape "
            f"{expected_shape}, but got {targets.shape}"
        )

    device = logits.device
    if targets.device != device:
        targets = targets.to(device=device, non_blocking=True)

    if targets.dtype != torch.long:
        targets = targets.to(dtype=torch.long)

    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)

    return F.cross_entropy(logits_flat.float(), targets_flat, reduction="mean")
