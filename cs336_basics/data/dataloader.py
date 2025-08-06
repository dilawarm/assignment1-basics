"""Highly optimized data loading for OpenWebText dataset on H100."""

import os
import random
from functools import lru_cache
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import GPT2TokenizerFast


class StreamingOpenWebTextDataset(IterableDataset):
    """Streaming OpenWebText dataset for maximum throughput."""

    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "gpt2",
        max_length: int = 1024,
        buffer_size: int = 10000,
        seed: int = 42,
    ):
        self.split = split
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.seed = seed

        # Initialize tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load streaming dataset
        self.dataset = load_dataset(
            "openwebtext",
            split="train",
            streaming=True,
        )

        # Create train/val split
        if split == "validation":
            self.dataset = self.dataset.take(int(1e6))  # Take 1M examples for validation
        else:
            self.dataset = self.dataset.skip(int(1e6))  # Skip 1M examples for train

        # Shuffle dataset
        self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over tokenized sequences."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # Multi-worker setup: split data among workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Skip examples for this worker
            dataset = self.dataset.skip(worker_id).take_every(num_workers)
        else:
            dataset = self.dataset

        # Buffer for concatenation
        buffer = []
        buffer_size = 0

        for example in dataset:
            # Tokenize text
            tokens = self.tokenizer(
                example["text"],
                return_tensors="pt",
                truncation=False,
                padding=False,
            )["input_ids"].squeeze(0)

            # Add to buffer
            buffer.append(tokens)
            buffer_size += len(tokens)

            # Yield sequences when buffer is large enough
            while buffer_size >= self.max_length:
                # Concatenate buffer
                concat_tokens = torch.cat(buffer)

                # Extract sequence
                sequence = concat_tokens[: self.max_length]
                remainder = concat_tokens[self.max_length :]

                # Update buffer
                if len(remainder) > 0:
                    buffer = [remainder]
                    buffer_size = len(remainder)
                else:
                    buffer = []
                    buffer_size = 0

                # Yield sequence
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


class OpenWebTextDataModule:
    """Optimized data module for maximum H100 throughput."""

    def __init__(
        self,
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

        # Create datasets
        self.train_dataset = StreamingOpenWebTextDataset(
            split="train",
            max_length=max_length,
            seed=seed,
        )

        self.val_dataset = StreamingOpenWebTextDataset(
            split="validation",
            max_length=max_length,
            seed=seed,
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
    batch_size: int = 32,
    max_length: int = 1024,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create optimized train and validation dataloaders."""
    data_module = OpenWebTextDataModule(
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
    )

    return data_module.train_dataloader(), data_module.val_dataloader()


# Alternative: Memory-mapped dataset for even better performance
class MemoryMappedDataset(Dataset):
    """Memory-mapped dataset for maximum I/O performance."""

    def __init__(self, data_path: str, max_length: int = 1024):
        self.max_length = max_length

        # Load memory-mapped data
        if os.path.exists(data_path):
            self.data = np.memmap(data_path, dtype=np.int32, mode="r")
            print(f"Loaded memory-mapped dataset: {len(self.data):,} tokens")
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Calculate number of sequences
        self.num_sequences = len(self.data) // max_length

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
