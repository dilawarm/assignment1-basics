"""
Data loading utilities for transformer training.
"""

from __future__ import annotations

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.long,
    pin_memory: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batch generation for language model training.

    This function efficiently samples random sequences from the dataset and
    prepares input-target pairs for autoregressive training.

    Args:
        dataset: Memory-mapped numpy array containing tokenized text
        batch_size: Number of sequences in the batch
        context_length: Length of each sequence
        device: Target device for tensors ('cuda' or 'cpu')
        dtype: Data type for tensors (default: torch.long)
        pin_memory: Whether to use pinned memory for faster GPU transfer

    Returns:
        Tuple of (inputs, targets) where:
        - inputs: (batch_size, context_length) tensor of token IDs
        - targets: (batch_size, context_length) tensor of next token IDs
    """
    data_size = len(dataset)

    if data_size < context_length + 1:
        raise ValueError(f"Dataset too small: {data_size} < {context_length + 1}")

    max_start_idx = data_size - context_length
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)

    input_batch = np.empty((batch_size, context_length), dtype=np.int64)
    target_batch = np.empty((batch_size, context_length), dtype=np.int64)

    for i, start_idx in enumerate(start_indices):
        input_batch[i] = dataset[start_idx : start_idx + context_length]
        target_batch[i] = dataset[start_idx + 1 : start_idx + context_length + 1]

    if pin_memory and device == "cuda":
        inputs = torch.from_numpy(input_batch).pin_memory().to(device=device, dtype=dtype, non_blocking=True)
        targets = torch.from_numpy(target_batch).pin_memory().to(device=device, dtype=dtype, non_blocking=True)
    else:
        inputs = torch.from_numpy(input_batch).to(device=device, dtype=dtype)
        targets = torch.from_numpy(target_batch).to(device=device, dtype=dtype)

    return inputs, targets
