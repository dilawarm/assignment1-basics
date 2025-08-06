"""Highly optimized data loading for pre-tokenized OpenWebText dataset on H100."""

import os
import random
from functools import lru_cache
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


class LocalTokenizedDataset(IterableDataset):
    """High-performance dataset for pre-tokenized .npy files."""

    def __init__(
        self,
        data_path: str,
        max_length: int = 1024,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.max_length = max_length
        self.seed = seed

        # Load memory-mapped tokenized data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Tokenized data file not found: {data_path}")

        # Use memory mapping for efficient access to large files
        # First, try to load with np.load to detect the actual dtype
        try:
            sample_array = np.load(data_path)
            actual_dtype = sample_array.dtype
            del sample_array  # Free memory
        except:
            actual_dtype = np.uint16  # Default fallback
            
        self.tokens = np.memmap(data_path, dtype=actual_dtype, mode='r')
        self.total_tokens = len(self.tokens)
        
        print(f"📁 Loaded {data_path}: {self.total_tokens:,} tokens ({self.total_tokens/1e9:.2f}B)")
        print(f"📊 Data type: {actual_dtype}, file size: {os.path.getsize(data_path)/1e9:.2f}GB")
        
        # Calculate number of sequences we can create
        self.num_sequences = self.total_tokens // max_length
        print(f"📊 Number of {max_length}-token sequences: {self.num_sequences:,}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over tokenized sequences with efficient random access."""
        worker_info = torch.utils.data.get_worker_info()
        
        # Set up worker-specific data ranges
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            # Each worker gets a portion of the sequences
            sequences_per_worker = self.num_sequences // num_workers
            start_seq = worker_id * sequences_per_worker
            end_seq = (worker_id + 1) * sequences_per_worker if worker_id < num_workers - 1 else self.num_sequences
        else:
            start_seq = 0
            end_seq = self.num_sequences
        
        # Create shuffled sequence indices for this worker
        np.random.seed(self.seed + (worker_info.id if worker_info else 0))
        sequence_indices = np.arange(start_seq, end_seq)
        np.random.shuffle(sequence_indices)
        
        # Yield sequences
        for seq_idx in sequence_indices:
            start_token_idx = seq_idx * self.max_length
            end_token_idx = start_token_idx + self.max_length
            
            # Extract sequence from memory-mapped array
            sequence_tokens = self.tokens[start_token_idx:end_token_idx]
            
            # Convert to torch tensor
            sequence = torch.from_numpy(sequence_tokens.astype(np.int64))
            
            yield {
                "input_ids": sequence,
                "labels": sequence.clone(),
            }


class OptimizedDataCollator:
    """Optimized data collator for maximum throughput."""

    def __init__(self, pad_token_id: int = 50256):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        # Pre-allocate tensors for better performance
        batch_size = len(examples)
        seq_len = examples[0]["input_ids"].shape[0]

        # Stack input_ids efficiently
        input_ids = torch.stack([example["input_ids"] for example in examples])
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class LocalDataModule:
    """Optimized data module for pre-tokenized local files."""

    def __init__(
        self,
        train_data_path: str = "training_data/owt_train_tokens.npy",
        val_data_path: str = "training_data/owt_valid_tokens.npy",
        batch_size: int = 32,  # Increased for H100
        max_length: int = 1024,
        num_workers: int = 8,  # Increased for better throughput
        prefetch_factor: int = 4,  # Increased prefetching
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.seed = seed

        # Create datasets from local files
        self.train_dataset = LocalTokenizedDataset(
            data_path=train_data_path,
            max_length=max_length,
            seed=seed,
        )

        self.val_dataset = LocalTokenizedDataset(
            data_path=val_data_path,
            max_length=max_length,
            seed=seed + 1,  # Different seed for validation
        )

        # Optimized data collator
        self.collator = OptimizedDataCollator()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,  # For consistent batch sizes
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
        )


def create_dataloaders(
    train_data_path: str = "training_data/owt_train_tokens.npy",
    val_data_path: str = "training_data/owt_valid_tokens.npy",
    batch_size: int = 32,
    max_length: int = 1024,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create optimized train and validation dataloaders from local .npy files."""
    data_module = LocalDataModule(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
    )

    return data_module.train_dataloader(), data_module.val_dataloader()


# Alternative: Regular Dataset (non-streaming) for even simpler access
class LocalTokenizedDatasetFixed(Dataset):
    """Fixed-size dataset for pre-tokenized .npy files with indexing."""

    def __init__(self, data_path: str, max_length: int = 1024):
        self.max_length = max_length

        # Load memory-mapped data
        if os.path.exists(data_path):
            # Detect actual dtype from the file
            try:
                sample_array = np.load(data_path)
                actual_dtype = sample_array.dtype
                del sample_array
            except:
                actual_dtype = np.uint16  # Default fallback
                
            self.data = np.memmap(data_path, dtype=actual_dtype, mode="r")
            print(f"📁 Loaded memory-mapped dataset: {len(self.data):,} tokens (dtype: {actual_dtype})")
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Calculate number of sequences
        self.num_sequences = len(self.data) // max_length
        print(f"📊 Fixed dataset sequences: {self.num_sequences:,}")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length

        sequence = torch.from_numpy(self.data[start_idx:end_idx].astype(np.int64))

        return {
            "input_ids": sequence,
            "labels": sequence.clone(),
        }
