"""Data handling components for long-context model training.

This module contains Lightning data components that can be used independently
of training scripts.
"""

import logging

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from efficient_longctx.training.data import (
    DataCollator,
    setup_tokenizer,
    stream_dataset,
)


class TokenizedDataset(Dataset):
    """Dataset wrapper for tokenized examples."""

    def __init__(self, examples: list[dict]) -> None:
        """Initialize with tokenized examples.

        Args:
            examples: List of tokenized examples with 'input_ids' key
        """
        self.examples = examples

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """Get example by index."""
        return self.examples[idx]


class LongCtxDataModule(LightningDataModule):
    """Data module for long-context model training."""

    def __init__(
        self,
        *,  # Force keyword-only arguments
        dataset_name: str = "openwebtext",
        max_tokens: int = 1000000,
        seq_len: int = 2048,
        batch_size: int = 16,
        val_split: float = 0.01,
        dataset_config: str | None = None,
        tokenizer_name: str = "gpt2",
        num_workers: int = 4,
    ) -> None:
        """Initialize data module.

        Args:
            dataset_name: Name of dataset to load
            max_tokens: Maximum number of tokens to process
            seq_len: Sequence length for packing
            batch_size: Batch size for training
            val_split: Fraction of data to use for validation
            dataset_config: Optional dataset configuration name
            tokenizer_name: Hugging Face model name for tokenizer
            num_workers: Number of workers for data loading
        """
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.max_tokens = max_tokens
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.val_split = val_split
        self.dataset_config = dataset_config
        self.tokenizer_name = tokenizer_name
        self.num_workers = num_workers

        self.tokenizer: AutoTokenizer | None = None
        self.collator: DataCollator | None = None

    def prepare_data(self) -> None:
        """Download and prepare dataset."""
        # Set up tokenizer
        self.tokenizer = setup_tokenizer(self.tokenizer_name)
        self.collator = DataCollator(self.seq_len, self.tokenizer.pad_token_id)

    def setup(self, stage: str) -> None:
        """Set up train and validation datasets."""
        if stage == "fit":
            # Ensure tokenizer and collator are set up
            if self.tokenizer is None:
                self.tokenizer = setup_tokenizer(self.tokenizer_name)
                self.collator = DataCollator(self.seq_len, self.tokenizer.pad_token_id)

            # Load and tokenize dataset
            train_iter, val_iter = stream_dataset(
                self.dataset_name,
                self.max_tokens,
                self.tokenizer,
                self.val_split,
                self.dataset_config,
            )

            # Convert iterators to lists for dataset creation
            self.train_examples = list(train_iter)
            self.val_examples = list(val_iter)

            logging.info(f"Loaded {len(self.train_examples)} train examples")
            logging.info(f"Loaded {len(self.val_examples)} val examples")

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader.

        Returns:
            Training dataloader

        Raises:
            RuntimeError: If setup() hasn't been called
        """
        if not hasattr(self, "train_examples"):
            raise RuntimeError(
                "train_examples not found. Make sure setup('fit') was called before train_dataloader()"
            )
        train_dataset = TokenizedDataset(self.train_examples)
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader.

        Returns:
            Validation dataloader

        Raises:
            RuntimeError: If setup() hasn't been called
        """
        if not hasattr(self, "val_examples"):
            raise RuntimeError(
                "val_examples not found. Make sure setup('fit') was called before val_dataloader()"
            )
        val_dataset = TokenizedDataset(self.val_examples)
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
