"""Efficient data loading for OpenWebText dataset."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class OpenWebTextDataset(Dataset):
    """OpenWebText dataset loading from preprocessed numpy files."""

    def __init__(
        self,
        data_path: str,
        max_length: int = 1024,
        memmap: bool = True,
    ):
        """
        Initialize OpenWebText dataset from local numpy files.

        Args:
            data_path: Path to the .npy file (e.g., 'data/encoded/owt_train_tokens.npy')
            max_length: Maximum sequence length
            memmap: Whether to use memory mapping for efficient loading
        """
        self.data_path = data_path
        self.max_length = max_length

        # Load data as memory-mapped array for efficiency
        print(f"Loading data from {data_path}...")
        if memmap:
            self.tokens = np.load(data_path, mmap_mode="r")
        else:
            self.tokens = np.load(data_path)

        # Calculate number of sequences
        self.total_tokens = len(self.tokens)
        self.num_sequences = self.total_tokens // max_length

        print(f"Loaded {self.total_tokens:,} tokens ({self.num_sequences:,} sequences of length {max_length})")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Get sequence at index
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length

        # Extract sequence
        input_ids = torch.from_numpy(self.tokens[start_idx:end_idx].astype(np.int64))

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


class DataCollator:
    """Custom data collator for language modeling."""

    def __init__(self, pad_token_id: int = 50256):  # GPT-2 pad token
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        # Stack input_ids
        input_ids = torch.stack([example["input_ids"] for example in examples])

        # Labels are the same as inputs for language modeling
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class OpenWebTextDataModule:
    """Data module for OpenWebText with train/validation splits."""

    def __init__(
        self,
        train_path: str = "data/encoded/owt_train_tokens.npy",
        val_path: str = "data/encoded/owt_valid_tokens.npy",
        batch_size: int = 8,
        max_length: int = 1024,
        num_workers: int = 4,
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

        # Create datasets from local files
        self.train_dataset = OpenWebTextDataset(
            data_path=train_path,
            max_length=max_length,
            memmap=True,  # Use memory mapping for large files
        )

        self.val_dataset = OpenWebTextDataset(
            data_path=val_path,
            max_length=max_length,
            memmap=True,
        )

        # Data collator
        self.collator = DataCollator()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


def create_dataloaders(
    batch_size: int = 8,
    max_length: int = 1024,
    num_workers: int = 4,
    train_path: str = "data/encoded/owt_train_tokens.npy",
    val_path: str = "data/encoded/owt_valid_tokens.npy",
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from local numpy files."""
    data_module = OpenWebTextDataModule(
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
    )

    return data_module.train_dataloader(), data_module.val_dataloader()
