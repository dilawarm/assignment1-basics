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

    # Ensure tensors are on same device and correct dtypes
    device = logits.device
    if targets.device != device:
        targets = targets.to(device=device, non_blocking=True)

    if targets.dtype != torch.long:
        targets = targets.to(dtype=torch.long)

    # Flatten tensors for cross-entropy computation
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    # Clamp targets to valid range
    targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)

    # Standard PyTorch cross-entropy
    return F.cross_entropy(logits_flat, targets_flat, reduction="mean")


def robust_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0, ignore_index: int = -100
) -> torch.Tensor:
    """
    Cross-entropy with label smoothing support.

    Args:
        logits: Logits tensor of shape (..., vocab_size)
        targets: Target indices tensor of shape (...)
        label_smoothing: Label smoothing factor (0.0 to 1.0)
        ignore_index: Index to ignore in loss computation

    Returns:
        Scalar cross-entropy loss tensor
    """
    device = logits.device
    vocab_size = logits.shape[-1]

    # Ensure correct device and dtype
    logits = logits.to(device=device)
    targets = targets.to(device=device, dtype=torch.long)

    # Flatten tensors
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    # Handle ignore_index
    if ignore_index >= 0:
        mask = targets_flat != ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=device, dtype=logits.dtype)
        logits_flat = logits_flat[mask]
        targets_flat = targets_flat[mask]

    # Clamp targets to valid range
    targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)

    # Use PyTorch's built-in label smoothing if available
    if hasattr(F, "cross_entropy") and label_smoothing > 0.0:
        try:
            return F.cross_entropy(logits_flat, targets_flat, reduction="mean", label_smoothing=label_smoothing)
        except TypeError:
            # Fallback for older PyTorch versions without label_smoothing parameter
            pass

    if label_smoothing > 0.0:
        # Manual label smoothing implementation
        log_probs = F.log_softmax(logits_flat, dim=-1)
        nll_loss = -log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - label_smoothing) * nll_loss + label_smoothing * smooth_loss
        return loss.mean()
    else:
        return F.cross_entropy(logits_flat, targets_flat, reduction="mean")
