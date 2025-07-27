"""
Cross-entropy loss functions for language model training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int


def cross_entropy(
    logits: Float[torch.Tensor, "... vocab_size"], targets: Int[torch.Tensor, "..."]
) -> Float[torch.Tensor, ""]:
    """
    Compute cross-entropy loss between logits and targets.

    Args:
        logits: Model predictions of shape (..., vocab_size)
        targets: Ground truth token IDs of shape (...)

    Returns:
        Scalar cross-entropy loss
    """
    # Flatten inputs for F.cross_entropy
    logits_flat = logits.view(-1, logits.size(-1))  # (batch * seq_len, vocab_size)
    targets_flat = targets.view(-1)  # (batch * seq_len,)

    # Handle invalid targets by clamping to valid range [0, vocab_size-1]
    vocab_size = logits_flat.size(-1)
    targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")

    return loss


def robust_cross_entropy(
    logits: Float[torch.Tensor, "... vocab_size"], targets: Int[torch.Tensor, "..."], label_smoothing: float = 0.1
) -> Float[torch.Tensor, ""]:
    """
    Compute robust cross-entropy loss with label smoothing.

    Label smoothing helps prevent overconfident predictions and can improve
    generalization by encouraging the model to be less certain about its predictions.

    Args:
        logits: Model predictions of shape (..., vocab_size)
        targets: Ground truth token IDs of shape (...)
        label_smoothing: Label smoothing factor (0.0 = no smoothing, typical: 0.1)

    Returns:
        Scalar cross-entropy loss with label smoothing
    """
    if label_smoothing <= 0.0:
        return cross_entropy(logits, targets)

    # Flatten inputs
    logits_flat = logits.view(-1, logits.size(-1))  # (batch * seq_len, vocab_size)
    targets_flat = targets.view(-1)  # (batch * seq_len,)

    vocab_size = logits_flat.size(-1)

    # Convert targets to one-hot and apply label smoothing
    one_hot = torch.zeros_like(logits_flat).scatter_(1, targets_flat.unsqueeze(1), 1)
    smooth_targets = one_hot * (1.0 - label_smoothing) + label_smoothing / vocab_size

    # Compute log probabilities
    log_probs = F.log_softmax(logits_flat, dim=-1)

    # Compute smoothed cross-entropy
    loss = -(smooth_targets * log_probs).sum(dim=-1).mean()

    return loss


def focal_loss(
    logits: Float[torch.Tensor, "... vocab_size"],
    targets: Int[torch.Tensor, "..."],
    alpha: float = 1.0,
    gamma: float = 2.0,
) -> Float[torch.Tensor, ""]:
    """
    Compute focal loss for handling class imbalance.

    Focal loss down-weights easy examples and focuses on hard examples.
    Useful for vocabularies with highly imbalanced token frequencies.

    Args:
        logits: Model predictions of shape (..., vocab_size)
        targets: Ground truth token IDs of shape (...)
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0, higher = more focus on hard examples)

    Returns:
        Scalar focal loss
    """
    # Flatten inputs
    logits_flat = logits.view(-1, logits.size(-1))  # (batch * seq_len, vocab_size)
    targets_flat = targets.view(-1)  # (batch * seq_len,)

    # Compute cross-entropy
    ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")

    # Compute probabilities and focal weight
    probs = torch.exp(-ce_loss)  # p_t
    focal_weight = alpha * (1 - probs) ** gamma

    # Apply focal weight
    focal_loss = focal_weight * ce_loss

    return focal_loss.mean()
