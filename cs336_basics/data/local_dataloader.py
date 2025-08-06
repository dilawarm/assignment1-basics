"""Local file-based data loading for OpenWebText with GPT2TokenizerFast."""

import mmap
import os
import random
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import GPT2TokenizerFast


class LocalOpenWebTextDataset(IterableDataset):
    """Memory-mapped streaming dataset for local OpenWebText files."""

    def __init__(
        self,
        file_path: str,
        tokenizer_name: str = "gpt2",
        max_length: int = 1024,
        buffer_size: int = 50000,  # Number of characters to read at once
        seed: int = 42,
        skip_lines: int = 0,
        take_lines: Optional[int] = None,
    ):
        self.file_path = file_path
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.seed = seed
        self.skip_lines = skip_lines
        self.take_lines = take_lines

        # Initialize tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Verify file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training data file not found: {file_path}")

        # Get file size for progress tracking
        self.file_size = os.path.getsize(file_path)
        print(f"Loading from {file_path} ({self.file_size / 1e9:.1f}GB)")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over tokenized sequences."""
        worker_info = torch.utils.data.get_worker_info()

        # Handle multi-worker data loading
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Calculate byte ranges for this worker
            bytes_per_worker = self.file_size // num_workers
            start_byte = worker_id * bytes_per_worker
            end_byte = start_byte + bytes_per_worker if worker_id < num_workers - 1 else self.file_size
        else:
            start_byte = 0
            end_byte = self.file_size

        # Buffer for accumulating tokens
        token_buffer = []
        buffer_size = 0

        with open(self.file_path, "r", encoding="utf-8") as f:
            # Seek to start position for this worker
            if start_byte > 0:
                f.seek(start_byte)
                # Skip to next newline to avoid splitting lines
                f.readline()

            lines_processed = 0

            while f.tell() < end_byte:
                # Read a batch of lines
                lines = []
                current_size = 0

                for _ in range(100):  # Read 100 lines at a time
                    if f.tell() >= end_byte:
                        break

                    line = f.readline()
                    if not line:
                        break

                    # Skip empty lines
                    line = line.strip()
                    if not line:
                        continue

                    # Apply skip/take logic
                    if self.skip_lines > 0 and lines_processed < self.skip_lines:
                        lines_processed += 1
                        continue

                    if self.take_lines is not None and lines_processed >= self.skip_lines + self.take_lines:
                        break

                    lines.append(line)
                    current_size += len(line)
                    lines_processed += 1

                    # Process batch when it gets large enough
                    if current_size > self.buffer_size:
                        break

                if not lines:
                    break

                # Tokenize the batch of lines
                batch_text = "\n".join(lines)

                try:
                    tokens = self.tokenizer(
                        batch_text,
                        return_tensors="pt",
                        truncation=False,
                        padding=False,
                        add_special_tokens=False,
                    )["input_ids"].squeeze(0)

                    # Add to buffer
                    token_buffer.append(tokens)
                    buffer_size += len(tokens)

                    # Yield sequences when buffer is large enough
                    while buffer_size >= self.max_length:
                        # Concatenate buffer
                        if len(token_buffer) > 1:
                            concat_tokens = torch.cat(token_buffer)
                        else:
                            concat_tokens = token_buffer[0]

                        # Extract sequence
                        sequence = concat_tokens[: self.max_length]
                        remainder = concat_tokens[self.max_length :]

                        # Update buffer
                        if len(remainder) > 0:
                            token_buffer = [remainder]
                            buffer_size = len(remainder)
                        else:
                            token_buffer = []
                            buffer_size = 0

                        # Yield sequence
                        yield {
                            "input_ids": sequence,
                            "labels": sequence.clone(),
                        }

                except Exception as e:
                    print(f"Warning: Failed to tokenize batch: {e}")
                    continue


class PreprocessedDataset(Dataset):
    """Dataset for pre-tokenized data stored as memory-mapped arrays."""

    def __init__(self, data_file: str, max_length: int = 1024):
        self.max_length = max_length

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Preprocessed data file not found: {data_file}")

        # Load memory-mapped data
        self.data = np.memmap(data_file, dtype=np.int32, mode="r")
        self.num_sequences = len(self.data) // max_length

        print(f"Loaded preprocessed data: {len(self.data):,} tokens, {self.num_sequences:,} sequences")

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


class LocalOpenWebTextDataModule:
    """Data module for local OpenWebText files."""

    def __init__(
        self,
        train_file: str = "training_data/owt_train.txt",
        val_file: str = "training_data/owt_valid.txt",
        batch_size: int = 32,
        max_length: int = 1024,
        num_workers: int = 8,
        prefetch_factor: int = 4,
        seed: int = 42,
        use_preprocessed: bool = False,
        preprocessed_train: Optional[str] = None,
        preprocessed_val: Optional[str] = None,
    ):
        self.train_file = train_file
        self.val_file = val_file
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.seed = seed
        self.use_preprocessed = use_preprocessed

        # Create datasets
        if use_preprocessed and preprocessed_train and preprocessed_val:
            print("Using preprocessed datasets...")
            self.train_dataset = PreprocessedDataset(preprocessed_train, max_length)
            self.val_dataset = PreprocessedDataset(preprocessed_val, max_length)
        else:
            print("Using streaming datasets from text files...")
            self.train_dataset = LocalOpenWebTextDataset(
                file_path=train_file,
                max_length=max_length,
                seed=seed,
            )

            self.val_dataset = LocalOpenWebTextDataset(
                file_path=val_file,
                max_length=max_length,
                seed=seed,
            )

        # Data collator
        self.collator = OptimizedDataCollator()

    def train_dataloader(self):
        # For IterableDataset, don't use shuffle=True
        shuffle = not isinstance(self.train_dataset, IterableDataset)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
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
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
        )


def create_local_dataloaders(
    train_file: str = "training_data/owt_train.txt",
    val_file: str = "training_data/owt_valid.txt",
    batch_size: int = 32,
    max_length: int = 1024,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create optimized dataloaders for local OpenWebText files."""

    data_module = LocalOpenWebTextDataModule(
        train_file=train_file,
        val_file=val_file,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
    )

    return data_module.train_dataloader(), data_module.val_dataloader()


def preprocess_and_save(
    input_file: str,
    output_file: str,
    tokenizer_name: str = "gpt2",
    max_length: int = 1024,
    chunk_size: int = 10000,
) -> None:
    """Preprocess text file and save as memory-mapped array for faster loading."""

    print(f"Preprocessing {input_file} -> {output_file}")

    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)

    # Process in chunks and save
    all_tokens = []

    with open(input_file, "r", encoding="utf-8") as f:
        chunk = []

        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            chunk.append(line)

            # Process chunk when it gets large enough
            if len(chunk) >= chunk_size:
                batch_text = "\n".join(chunk)

                try:
                    tokens = tokenizer(
                        batch_text,
                        return_tensors="pt",
                        truncation=False,
                        padding=False,
                        add_special_tokens=False,
                    )["input_ids"].squeeze(0)

                    all_tokens.extend(tokens.numpy().astype(np.int32))

                except Exception as e:
                    print(f"Warning: Failed to tokenize chunk at line {line_num}: {e}")

                chunk = []

                if line_num % 100000 == 0:
                    print(f"Processed {line_num:,} lines, {len(all_tokens):,} tokens")

        # Process remaining chunk
        if chunk:
            batch_text = "\n".join(chunk)
            try:
                tokens = tokenizer(
                    batch_text,
                    return_tensors="pt",
                    truncation=False,
                    padding=False,
                    add_special_tokens=False,
                )["input_ids"].squeeze(0)

                all_tokens.extend(tokens.numpy().astype(np.int32))
            except Exception as e:
                print(f"Warning: Failed to tokenize final chunk: {e}")

    # Save as memory-mapped array
    tokens_array = np.array(all_tokens, dtype=np.int32)

    # Truncate to multiple of max_length for clean sequences
    num_sequences = len(tokens_array) // max_length
    truncated_length = num_sequences * max_length
    tokens_array = tokens_array[:truncated_length]

    print(f"Saving {len(tokens_array):,} tokens ({num_sequences:,} sequences) to {output_file}")

    # Create memory-mapped array
    mm_array = np.memmap(output_file, dtype=np.int32, mode="w+", shape=tokens_array.shape)
    mm_array[:] = tokens_array[:]
    del mm_array  # Flush to disk

    print(f"Preprocessing complete: {output_file}")


def main():
    """Example usage and preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess OpenWebText data")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess text files")
    parser.add_argument("--train_file", default="training_data/owt_train.txt", help="Training file")
    parser.add_argument("--val_file", default="training_data/owt_valid.txt", help="Validation file")
    parser.add_argument("--output_dir", default="training_data", help="Output directory")
    parser.add_argument("--test", action="store_true", help="Test data loading")

    args = parser.parse_args()

    if args.preprocess:
        # Preprocess files
        preprocess_and_save(
            args.train_file,
            os.path.join(args.output_dir, "owt_train_preprocessed.npy"),
        )
        preprocess_and_save(
            args.val_file,
            os.path.join(args.output_dir, "owt_val_preprocessed.npy"),
        )

    if args.test:
        # Test data loading
        print("Testing data loading...")

        train_loader, val_loader = create_local_dataloaders(
            train_file=args.train_file,
            val_file=args.val_file,
            batch_size=4,
            max_length=1024,
            num_workers=2,
        )

        print("Testing training data loader...")
        for i, batch in enumerate(train_loader):
            print(f"Batch {i}: {batch['input_ids'].shape}")
            if i >= 2:
                break

        print("Testing validation data loader...")
        for i, batch in enumerate(val_loader):
            print(f"Batch {i}: {batch['input_ids'].shape}")
            if i >= 2:
                break

        print("Data loading test complete!")


if __name__ == "__main__":
    main()
