"""Data loaders for SlimPajama/OpenWebText subsets.

Stream and tokenize small subsets of public corpora for quick iteration.
"""

import argparse
from collections.abc import Iterator
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


class DataCollator:
    """Collator that packs tokens into sequences with right-shift labels.

    Args:
        seq_len: Maximum sequence length for packing
        pad_token_id: Token ID to use for padding
    """

    def __init__(self, seq_len: int, pad_token_id: int):
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack tokens into sequences and create labels with right-shift.

        Args:
            batch: List of tokenized examples with 'input_ids' key

        Returns:
            Tuple of (input_ids, labels) tensors with shape (batch_size, seq_len)
        """
        input_ids_list = []
        labels_list = []

        for example in batch:
            tokens = example["input_ids"]

            # Split into chunks of seq_len
            for i in range(0, len(tokens), self.seq_len):
                chunk = tokens[i : i + self.seq_len]

                # Pad if necessary
                if len(chunk) < self.seq_len:
                    chunk = chunk + [self.pad_token_id] * (self.seq_len - len(chunk))

                # Create labels with right-shift (next token prediction)
                labels = chunk[1:] + [
                    self.pad_token_id
                ]  # Shift right, pad last position

                input_ids_list.append(chunk)
                labels_list.append(labels)

        return torch.tensor(input_ids_list), torch.tensor(labels_list)


def setup_tokenizer(model_name: str = "gpt2") -> AutoTokenizer:
    """Set up tokenizer with proper padding token.

    Args:
        model_name: Hugging Face model name for tokenizer

    Returns:
        Configured tokenizer with pad_token set to eos_token
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def stream_dataset(
    dataset_name: str,
    max_tokens: int,
    tokenizer: AutoTokenizer,
    val_split: float = 0.001,
    dataset_config: str | None = None,
) -> tuple[Iterator[dict], Iterator[dict]]:
    """Stream dataset and tokenize examples.

    Args:
        dataset_name: Name of dataset to load (e.g., "openwebtext")
        max_tokens: Maximum number of tokens to process
        tokenizer: Tokenizer for processing text
        val_split: Fraction of data to use for validation
        dataset_config: Optional dataset configuration name

    Returns:
        Tuple of (train_iterator, val_iterator) yielding tokenized examples
    """
    # Load streaming dataset
    if dataset_config:
        dataset = load_dataset(
            dataset_name, dataset_config, split="train", streaming=True
        )
    else:
        dataset = load_dataset(dataset_name, split="train", streaming=True)

    train_examples = []
    val_examples = []
    total_tokens = 0

    print(f"Processing {dataset_name} dataset...")

    for example in tqdm(dataset, desc="Tokenizing"):
        if total_tokens >= max_tokens:
            break

        # Tokenize the text
        text = example.get("text", "")
        if not text.strip():
            continue

        tokenized = tokenizer(text, truncation=True, max_length=2048)

        # Count tokens
        num_tokens = len(tokenized["input_ids"])
        total_tokens += num_tokens

        # Split into train/val based on val_split
        if len(train_examples) / (len(train_examples) + len(val_examples) + 1) < (
            1 - val_split
        ):
            train_examples.append(tokenized)
        else:
            val_examples.append(tokenized)

    print(f"Processed {total_tokens:,} tokens")
    print(f"Train examples: {len(train_examples):,}")
    print(f"Val examples: {len(val_examples):,}")

    return iter(train_examples), iter(val_examples)


def dry_run_samples(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    num_samples: int = 5,
    dataset_config: str | None = None,
) -> None:
    """Print several tokenized samples for inspection.

    Args:
        dataset_name: Name of dataset to load
        tokenizer: Tokenizer for processing text
        num_samples: Number of samples to print
        dataset_config: Optional dataset configuration name
    """
    print(f"\n=== Dry run: {num_samples} samples from {dataset_name} ===")

    if dataset_config:
        dataset = load_dataset(
            dataset_name, dataset_config, split="train", streaming=True
        )
    else:
        dataset = load_dataset(dataset_name, split="train", streaming=True)

    for i, example in enumerate(dataset):
        if i >= num_samples:
            break

        text = example.get("text", "")
        if not text.strip():
            continue

        # Tokenize and truncate for display
        tokenized = tokenizer(text, truncation=True, max_length=100)

        print(f"\n--- Sample {i + 1} ---")
        print(f"Original text (first 200 chars): {text[:200]}...")
        print(f"Tokenized length: {len(tokenized['input_ids'])}")
        print(f"First 20 tokens: {tokenized['input_ids'][:20]}")
        print(f"Decoded back: {tokenizer.decode(tokenized['input_ids'][:20])}")


def save_tokenizer(tokenizer: AutoTokenizer, save_path: str) -> None:
    """Save tokenizer to specified path.

    Args:
        tokenizer: Tokenizer to save
        save_path: Directory path to save tokenizer
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(save_path)
    print(f"Tokenizer saved to {save_path}")


def main():
    """CLI entry point for data loading."""
    parser = argparse.ArgumentParser(description="Data loaders for pretraining")
    parser.add_argument("--dataset", default="openwebtext", help="Dataset name")
    parser.add_argument("--dataset_config", help="Dataset configuration name")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument(
        "--max_tokens", type=float, default=1e8, help="Maximum tokens to process"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.001, help="Validation split fraction"
    )
    parser.add_argument("--save_tokenizer", help="Path to save tokenizer")
    parser.add_argument(
        "--dry_run", action="store_true", help="Print tokenized samples"
    )
    parser.add_argument("--model_name", default="gpt2", help="Tokenizer model name")

    args = parser.parse_args()

    # Set up tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = setup_tokenizer(args.model_name)

    if args.dry_run:
        dry_run_samples(args.dataset, tokenizer, dataset_config=args.dataset_config)
        return

    # Process dataset
    train_iter, val_iter = stream_dataset(
        args.dataset,
        int(args.max_tokens),
        tokenizer,
        args.val_split,
        args.dataset_config,
    )

    # Test collator
    print(f"\nTesting collator with seq_len={args.seq_len}")
    collator = DataCollator(args.seq_len, tokenizer.pad_token_id)

    # Get a small batch for testing
    batch = []
    for i, example in enumerate(train_iter):
        if i >= 2:  # Small batch for testing
            break
        batch.append(example)

    input_ids, labels = collator(batch)
    print(f"Collator output shapes: input_ids={input_ids.shape}, labels={labels.shape}")
    print(f"Sample input_ids: {input_ids[0][:20]}")
    print(f"Sample labels: {labels[0][:20]}")

    # Save tokenizer if requested
    if args.save_tokenizer:
        save_tokenizer(tokenizer, args.save_tokenizer)


if __name__ == "__main__":
    main()
