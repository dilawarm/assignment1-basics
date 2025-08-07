"""Optimized data loading for maximum H100 throughput using pre-tokenized .npy files."""

import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class OptimizedNpyDataset(Dataset):
    """
    Optimized dataset using pre-tokenized .npy files for maximum H100 throughput.

    Key optimizations:
    1. Pre-tokenized data - no tokenization overhead
    2. Memory-mapped numpy arrays for efficient loading
    3. Fixed-length sequences for optimal GPU utilization
    4. Fast random access for efficient shuffling
    """

    def __init__(
        self,
        split: str = "train",
        max_length: int = 1024,
        data_dir: str = "training_data",
        seed: int = 42,
        use_memmap: bool = True,
    ):
        self.split = split
        self.max_length = max_length
        self.data_dir = data_dir
        self.seed = seed
        self.use_memmap = use_memmap

        # Determine file path based on split
        if split == "train":
            self.data_file = os.path.join(data_dir, "owt_train_tokens.npy")
        elif split in ["validation", "val"]:
            self.data_file = os.path.join(data_dir, "owt_valid_tokens.npy")
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"🔧 Loading pre-tokenized data: {self.data_file}")

        # Check if file exists
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        # Load or memory-map the data
        if use_memmap:
            # Memory-mapped loading for very large files
            self.tokens = np.load(self.data_file, mmap_mode="r")
            print(f"✅ Memory-mapped {len(self.tokens):,} tokens")
        else:
            # Load entire file into memory (faster access but uses more RAM)
            self.tokens = np.load(self.data_file)
            print(f"✅ Loaded {len(self.tokens):,} tokens into memory")

        # Calculate number of sequences
        self.num_sequences = len(self.tokens) // max_length
        self.total_tokens = self.num_sequences * max_length

        print(f"📊 Dataset stats:")
        print(f"   Total tokens: {len(self.tokens):,}")
        print(f"   Usable tokens: {self.total_tokens:,}")
        print(f"   Sequences: {self.num_sequences:,}")
        print(f"   Sequence length: {max_length}")

        # Set random seed for reproducible shuffling
        random.seed(seed)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """Get a sequence of tokens starting at position idx * max_length."""
        if idx >= self.num_sequences:
            raise IndexError(f"Index {idx} out of range [0, {self.num_sequences})")

        start_pos = idx * self.max_length
        end_pos = start_pos + self.max_length

        # Extract sequence
        sequence = self.tokens[start_pos:end_pos]

        # Convert to tensor
        input_ids = torch.tensor(sequence, dtype=torch.long)

        return {
            "input_ids": input_ids,
        }


class OptimizedDataCollator:
    """
    Optimized data collator with dynamic batching and memory optimizations.
    """

    def __init__(
        self,
        pad_token_id: int = 50256,
        device: str = "cuda",
        pin_memory: bool = True,
    ):
        self.pad_token_id = pad_token_id
        self.device = device
        self.pin_memory = pin_memory

    def __call__(self, examples):
        """Efficiently collate examples into batches."""
        # Stack input_ids efficiently
        input_ids = torch.stack([example["input_ids"] for example in examples])

        # Labels are the same as inputs for language modeling
        labels = input_ids.clone()

        batch = {
            "input_ids": input_ids,
            "labels": labels,
        }

        # Pin memory for faster GPU transfer (with proper CUDA initialization check)
        if self.pin_memory and torch.cuda.is_available():
            try:
                # Ensure CUDA is properly initialized
                torch.cuda.init()
                for key in batch:
                    batch[key] = batch[key].pin_memory()
            except RuntimeError as e:
                # If CUDA initialization fails, continue without pinned memory
                if "CUDA" in str(e):
                    # Only warn once to avoid spam
                    if not hasattr(self, "_cuda_warning_shown"):
                        print(f"⚠️  CUDA pinning failed: {e}. Continuing without pinned memory.")
                        self._cuda_warning_shown = True
                else:
                    raise e

        return batch


class OptimizedNpyDataModule:
    """
    Optimized data module using pre-tokenized .npy files for H100-specific configurations.
    """

    def __init__(
        self,
        batch_size: int = 16,
        max_length: int = 1024,
        num_workers: int = 8,
        data_dir: str = "training_data",
        seed: int = 42,
        use_memmap: bool = True,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True,
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.seed = seed
        self.use_memmap = use_memmap
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

        print(f"🔧 Creating optimized .npy data module...")
        print(f"   Data directory: {data_dir}")
        print(f"   Batch size: {batch_size}")
        print(f"   Workers: {num_workers}")
        print(f"   Memory-mapped: {use_memmap}")
        print(f"   Prefetch factor: {prefetch_factor}")

        # Create datasets
        self.train_dataset = OptimizedNpyDataset(
            split="train",
            max_length=max_length,
            data_dir=data_dir,
            seed=seed,
            use_memmap=use_memmap,
        )

        self.val_dataset = OptimizedNpyDataset(
            split="validation",
            max_length=max_length,
            data_dir=data_dir,
            seed=seed,
            use_memmap=use_memmap,
        )

        # Data collator
        self.collator = OptimizedDataCollator(pin_memory=pin_memory)

    def train_dataloader(self):
        """Create optimized training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True,  # Important for consistent batch sizes
        )

    def val_dataloader(self):
        """Create optimized validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffle for validation
            num_workers=min(self.num_workers, 4),  # Fewer workers for validation
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=False,
        )


def create_optimized_dataloaders(
    batch_size: int = 16,
    max_length: int = 1024,
    num_workers: int = 8,
    data_dir: str = "training_data",
    seed: int = 42,
    use_memmap: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create optimized train and validation dataloaders using .npy files."""
    data_module = OptimizedNpyDataModule(
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
        data_dir=data_dir,
        seed=seed,
        use_memmap=use_memmap,
        pin_memory=pin_memory,
    )

    return data_module.train_dataloader(), data_module.val_dataloader()


# Backward compatibility function names
def create_npy_dataloaders(*args, **kwargs):
    """Alias for create_optimized_dataloaders for clarity."""
    return create_optimized_dataloaders(*args, **kwargs)
