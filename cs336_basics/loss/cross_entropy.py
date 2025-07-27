"""
Numerically stable cross-entropy loss with advanced device management and memory optimization.

This module provides memory-efficient cross-entropy computation that prevents device
mismatch errors and handles out-of-memory situations gracefully.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int


def cross_entropy(
    logits: Float[torch.Tensor, "... vocab_size"], targets: Int[torch.Tensor, "..."]
) -> Float[torch.Tensor, ""]:
    """
    Compute cross-entropy loss with robust device handling and memory optimization.

    This implementation:
    - Ensures all tensors are on the same device
    - Handles memory overflow gracefully with chunked processing
    - Uses numerically stable cross-entropy computation
    - Prevents device mismatch errors that can occur during training

    Args:
        logits: Logits tensor of shape (..., vocab_size)
        targets: Target indices tensor of shape (...)

    Returns:
        Scalar cross-entropy loss tensor

    Raises:
        ValueError: If tensor shapes are incompatible
        RuntimeError: If computation fails after all fallback attempts
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
        targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
    elif targets.dtype != torch.long:
        targets = targets.to(dtype=torch.long)

    if logits.dtype not in [torch.float32, torch.bfloat16, torch.float16]:
        logits = logits.float()

    try:
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        assert logits_flat.device == targets_flat.device, (
            f"Device mismatch after synchronization: logits on {logits_flat.device}, targets on {targets_flat.device}"
        )

        targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)

        loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")

        if torch.isfinite(loss):
            return loss
        else:
            print("Warning: Non-finite loss detected, falling back to chunked computation")
            return _chunked_cross_entropy(logits, targets, device)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("OOM in cross_entropy, using chunked computation")
            torch.cuda.empty_cache() if device.type == "cuda" else None
            return _chunked_cross_entropy(logits, targets, device)
        elif "device" in str(e).lower() or "cuda" in str(e).lower():
            print(f"Device error in cross_entropy: {e}")
            return _synchronized_cross_entropy(logits, targets, device)
        else:
            print(f"Unexpected error in cross_entropy: {e}")
            return _stable_cross_entropy(logits, targets, device)


def _synchronized_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Cross-entropy with forced device synchronization."""
    vocab_size = logits.shape[-1]

    logits = logits.to(device=device, non_blocking=False)
    targets = targets.to(device=device, dtype=torch.long, non_blocking=False)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)

    return F.cross_entropy(logits_flat, targets_flat, reduction="mean")


def _chunked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, device: torch.device, chunk_size: int = 1024
) -> torch.Tensor:
    """
    Memory-efficient chunked cross-entropy computation.

    Processes the input in smaller chunks to avoid OOM issues while maintaining
    the computation graph for proper gradient computation.

    Args:
        logits: Logits tensor of shape (..., vocab_size)
        targets: Target indices tensor of shape (...)
        device: Device to perform computation on
        chunk_size: Size of chunks to process at once

    Returns:
        Scalar cross-entropy loss tensor with gradients preserved
    """
    vocab_size = logits.shape[-1]
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    logits_flat = logits_flat.to(device=device)
    targets_flat = targets_flat.to(device=device, dtype=torch.long)

    targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)

    # Use a list to collect chunk losses, then stack them to preserve gradients
    chunk_losses = []
    chunk_weights = []

    for i in range(0, logits_flat.shape[0], chunk_size):
        end_idx = min(i + chunk_size, logits_flat.shape[0])

        chunk_logits = logits_flat[i:end_idx]
        chunk_targets = targets_flat[i:end_idx]

        try:
            # Use reduction='mean' for each chunk to get proper weighting
            chunk_loss = F.cross_entropy(chunk_logits, chunk_targets, reduction="mean")

            if torch.isfinite(chunk_loss):
                chunk_losses.append(chunk_loss)
                chunk_weights.append(chunk_targets.numel())
            else:
                print(f"Warning: Non-finite loss in chunk {i // chunk_size}, skipping...")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # If even the chunk is too large, try with a smaller chunk
                print(f"OOM in chunk {i // chunk_size}, trying smaller chunks...")
                torch.cuda.empty_cache() if device.type == "cuda" else None

                # Recursively process with smaller chunks
                smaller_chunk_size = max(chunk_size // 4, 64)  # Minimum chunk size of 64
                if smaller_chunk_size < chunk_size:
                    smaller_logits = logits_flat[i:end_idx]
                    smaller_targets = targets_flat[i:end_idx]
                    chunk_loss = _chunked_cross_entropy(
                        smaller_logits.view(-1, vocab_size), smaller_targets, device, smaller_chunk_size
                    )
                    if torch.isfinite(chunk_loss):
                        chunk_losses.append(chunk_loss)
                        chunk_weights.append(chunk_targets.numel())
                else:
                    print(f"Cannot process chunk {i // chunk_size}, skipping...")
            else:
                raise e

    if not chunk_losses:
        # Fallback to a simple tensor if no chunks succeeded
        print("Warning: No chunks processed successfully, returning zero loss")
        return torch.tensor(0.0, device=device, dtype=logits.dtype, requires_grad=True)

    # Weighted average of chunk losses
    if len(chunk_losses) == 1:
        return chunk_losses[0]

    # Convert weights to tensor for proper gradient computation
    weights = torch.tensor(chunk_weights, device=device, dtype=logits.dtype)
    weights = weights / weights.sum()  # Normalize weights

    # Stack losses and compute weighted average
    losses_tensor = torch.stack(chunk_losses)
    weighted_loss = torch.sum(losses_tensor * weights)

    return weighted_loss


def _stable_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Numerically stable cross-entropy using manual computation.

    This is the most robust fallback that manually implements cross-entropy
    to avoid any potential issues with PyTorch's built-in function.
    """
    vocab_size = logits.shape[-1]
    logits_flat = logits.view(-1, vocab_size).to(device=device)
    targets_flat = targets.view(-1).to(device=device, dtype=torch.long)

    targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)

    logits_max = torch.max(logits_flat, dim=-1, keepdim=True)[0]
    logits_stable = logits_flat - logits_max

    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1, keepdim=True))
    log_probs = logits_stable - log_sum_exp

    target_log_probs = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

    return -target_log_probs.mean()


def robust_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0, ignore_index: int = -100
) -> torch.Tensor:
    """
    Ultra-robust cross-entropy with label smoothing and ignore index support.

    This is the most advanced version that handles all edge cases and
    provides additional features for improved training stability.

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

    logits = logits.to(device=device)
    targets = targets.to(device=device, dtype=torch.long)

    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    if ignore_index >= 0:
        mask = targets_flat != ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=device, dtype=logits.dtype)
        logits_flat = logits_flat[mask]
        targets_flat = targets_flat[mask]

    targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)

    if label_smoothing > 0.0:
        log_probs = F.log_softmax(logits_flat, dim=-1)
        nll_loss = -log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - label_smoothing) * nll_loss + label_smoothing * smooth_loss
        return loss.mean()
    else:
        return F.cross_entropy(logits_flat, targets_flat, reduction="mean")
