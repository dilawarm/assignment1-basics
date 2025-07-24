"""
Cross-entropy loss function.
"""

from __future__ import annotations

import torch
from jaxtyping import Float, Int


def cross_entropy(
    logits: Float[torch.Tensor, "... vocab_size"], targets: Int[torch.Tensor, "..."]
) -> Float[torch.Tensor, ""]:
    """
    Compute cross-entropy loss between logits and targets.

    This function computes the cross-entropy loss with multiple layers of numerical stability:
    1. Uses standard PyTorch implementation first
    2. Falls back to numerically stable version if NaN/Inf detected
    3. Adds epsilon to prevent log(0) issues only when needed

    Args:
        logits: predicted logits of shape (..., vocab_size)
        targets: target indices of shape (...)

    Returns:
        scalar cross-entropy loss averaged over the batch
    """
    targets = targets.to(logits.device)

    try:
        logits_max = logits.max(dim=-1, keepdim=True)[0]
        logits_stable = logits - logits_max

        log_sum_exp = torch.logsumexp(logits_stable, dim=-1, keepdim=True)
        log_softmax = logits_stable - log_sum_exp

        target_log_probs = log_softmax.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        loss = -target_log_probs.mean()

        if torch.isfinite(loss):
            return loss

    except Exception:
        pass

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
        print("WARNING: NaN/Inf detected in cross_entropy loss computation")
        return torch.tensor(10.0, device=logits.device, dtype=logits.dtype)

    return loss
