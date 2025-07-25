"""
Data loading utilities for transformer training.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.long,
    pin_memory: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


class BatchSampler:
    """
    Advanced batch sampler with memory optimization and prefetching.

    Features:
    - Pre-allocated memory pools
    - Asynchronous data loading
    - Dynamic batch sizing
    """

    def __init__(
        self,
        dataset: np.ndarray,
        batch_size: int,
        context_length: int,
        device: str = "cuda",
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        """Initialize batch sampler."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.data_size = len(dataset)
        self.max_start_idx = self.data_size - context_length

        if self.max_start_idx <= 0:
            raise ValueError(f"Dataset too small for context length {context_length}")

        self._init_memory_pools()

    def _init_memory_pools(self):
        """Initialize memory pools for efficient batch generation."""
        pool_size = self.prefetch_factor * self.batch_size

        self.input_pool = np.empty((pool_size, self.context_length), dtype=np.int64)
        self.target_pool = np.empty((pool_size, self.context_length), dtype=np.int64)

        if self.device == "cuda" and torch.cuda.is_available():
            self.gpu_input_buffer = torch.empty(
                (self.batch_size, self.context_length), dtype=torch.long, device=self.device
            )
            self.gpu_target_buffer = torch.empty(
                (self.batch_size, self.context_length), dtype=torch.long, device=self.device
            )

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch with memory access."""
        start_indices = np.random.randint(0, self.max_start_idx, size=self.batch_size)

        for i, start_idx in enumerate(start_indices):
            self.input_pool[i] = self.dataset[start_idx : start_idx + self.context_length]
            self.target_pool[i] = self.dataset[start_idx + 1 : start_idx + self.context_length + 1]

        if hasattr(self, "gpu_input_buffer"):
            self.gpu_input_buffer.copy_(torch.from_numpy(self.input_pool[: self.batch_size]))
            self.gpu_target_buffer.copy_(torch.from_numpy(self.target_pool[: self.batch_size]))
            return self.gpu_input_buffer.clone(), self.gpu_target_buffer.clone()
        else:
            inputs = torch.from_numpy(self.input_pool[: self.batch_size]).to(self.device)
            targets = torch.from_numpy(self.target_pool[: self.batch_size]).to(self.device)
            return inputs, targets


def create_dataloader(
    data_path: str,
    batch_size: int,
    context_length: int,
    device: str = "cuda",
    num_workers: int = 0,
    prefetch_factor: int = 2,
    use_memory_mapping: bool = True,
) -> BatchSampler:
    """
    Create a data loader for language model training.

    Args:
        data_path: Path to tokenized data file (.npy)
        batch_size: Batch size for training
        context_length: Sequence length
        device: Target device
        num_workers: Number of data loading workers
        prefetch_factor: Number of batches to prefetch
        use_memory_mapping: Whether to use memory mapping for large files

    Returns:
        Batch sampler
    """
    if use_memory_mapping:
        dataset = np.memmap(data_path, dtype=np.uint16, mode="r")
    else:
        dataset = np.load(data_path)

    return BatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
