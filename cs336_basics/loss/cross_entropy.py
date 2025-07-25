"""
Cross-entropy loss function.
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
        logits: predicted logits of shape (..., vocab_size)
        targets: target indices of shape (...)

    Returns:
        scalar cross-entropy loss averaged over the batch
    """
    device = logits.device
    dtype = logits.dtype

    if targets.device != device:
        targets = targets.to(device=device, non_blocking=True)

    batch_elements = logits.numel() // logits.size(-1)
    vocab_size = logits.size(-1)

    if batch_elements == 0 or vocab_size == 0:
        print("WARNING: Empty batch in cross_entropy, returning zero loss")
        return torch.tensor(0.0, device=device, dtype=dtype)

    memory_threshold = 8 * 1024**3
    tensor_memory = batch_elements * vocab_size * 4

    if tensor_memory > memory_threshold:
        return _chunked_cross_entropy(logits, targets)

    try:
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        assert logits_flat.device == targets_flat.device, (
            f"Device mismatch: logits on {logits_flat.device}, targets on {targets_flat.device}"
        )

        loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")

        if torch.isfinite(loss):
            return loss

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("OOM in standard path, falling back to chunked processing")
            return _chunked_cross_entropy(logits, targets)
        elif "device" in str(e).lower():
            print(f"Device mismatch error in cross_entropy: {e}")
            targets = targets.to(device=device, dtype=torch.long)
            logits_flat = logits.view(-1, vocab_size).to(device=device)
            targets_flat = targets.view(-1).to(device=device)
            loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
            return loss
        raise e

    return _stable_cross_entropy(logits, targets)


def _chunked_cross_entropy(
    logits: Float[torch.Tensor, "... vocab_size"], targets: Int[torch.Tensor, "..."], chunk_size: int = 1024
) -> Float[torch.Tensor, ""]:
    """
    Compute cross-entropy loss using chunked processing for memory efficiency.

    Args:
        logits: predicted logits of shape (..., vocab_size)
        targets: target indices of shape (...)
        chunk_size: number of elements to process at once

    Returns:
        scalar cross-entropy loss averaged over the batch
    """
    device = logits.device
    dtype = logits.dtype

    if targets.device != device:
        targets = targets.to(device=device, non_blocking=True)

    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    total_loss = 0.0
    total_elements = logits_flat.size(0)

    for i in range(0, total_elements, chunk_size):
        end_idx = min(i + chunk_size, total_elements)

        logits_chunk = logits_flat[i:end_idx]
        targets_chunk = targets_flat[i:end_idx]

        assert logits_chunk.device == targets_chunk.device, (
            f"Chunk device mismatch: logits on {logits_chunk.device}, targets on {targets_chunk.device}"
        )

        chunk_loss = F.cross_entropy(logits_chunk, targets_chunk, reduction="sum")
        total_loss += chunk_loss.item()

    return torch.tensor(total_loss / total_elements, device=device, dtype=dtype)


def _stable_cross_entropy(
    logits: Float[torch.Tensor, "... vocab_size"], targets: Int[torch.Tensor, "..."]
) -> Float[torch.Tensor, ""]:
    """
    Fallback numerically stable cross-entropy implementation.

    Args:
        logits: predicted logits of shape (..., vocab_size)
        targets: target indices of shape (...)

    Returns:
        scalar cross-entropy loss averaged over the batch
    """
    logits_clipped = torch.clamp(logits, min=-700.0, max=700.0)

    logits_max = logits_clipped.max(dim=-1, keepdim=True)[0]
    logits_stable = logits_clipped - logits_max

    log_sum_exp = torch.logsumexp(logits_stable, dim=-1, keepdim=True)
    log_softmax = logits_stable - log_sum_exp

    target_log_probs = log_softmax.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    epsilon = 1e-10
    target_log_probs = torch.clamp(target_log_probs, min=torch.log(torch.tensor(epsilon)))

    loss = -target_log_probs.mean()

    if torch.isnan(loss) or torch.isinf(loss):
        print("WARNING: NaN/Inf detected in stable cross_entropy computation")
        return torch.tensor(10.0, device=logits.device, dtype=logits.dtype)

    return loss
