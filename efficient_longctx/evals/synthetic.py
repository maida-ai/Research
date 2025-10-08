"""
Synthetic long-context evaluation tasks.

This module provides toy tasks to validate long-range retention and streaming stability:
- Passkey retrieval: plant a random token early; query at end
- Copy/recall: reproduce a distant subsequence
- Drift curve: track metric vs increasing sequence length
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from efficient_longctx.blocks.blade import BLADEBlock
from efficient_longctx.blocks.dpassm import DPASSMBlock


class SyntheticDataset:
    """Base class for synthetic evaluation datasets."""

    def __init__(self, vocab_size: int = 1000, seed: int = 42):
        self.vocab_size = vocab_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate_batch(
        self, batch_size: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of (input_ids, labels)."""
        raise NotImplementedError


class PasskeyDataset(SyntheticDataset):
    """Passkey retrieval task: plant a random token early; query at end."""

    def __init__(self, vocab_size: int = 1000, seed: int = 42):
        super().__init__(vocab_size, seed)
        # Reserve special tokens
        self.passkey_token = vocab_size - 1  # Use last token as passkey
        self.query_token = vocab_size - 2  # Second to last as query marker
        self.padding_token = vocab_size - 3  # Third to last as padding

    def generate_batch(
        self, batch_size: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate passkey retrieval examples."""
        # Generate random sequences
        input_ids = torch.randint(0, self.vocab_size - 3, (batch_size, seq_len))

        # Plant passkey token at random position in first half
        passkey_pos = torch.randint(0, seq_len // 2, (batch_size,))
        for i in range(batch_size):
            input_ids[i, passkey_pos[i]] = self.passkey_token

        # Add query token at the end
        input_ids[:, -1] = self.query_token

        # Labels: the passkey token should be predicted at the query position
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        labels[:, -1] = self.passkey_token  # Only the query position has a label

        return input_ids, labels


class CopyRecallDataset(SyntheticDataset):
    """Copy/recall task: reproduce a distant subsequence."""

    def __init__(
        self, vocab_size: int = 1000, subsequence_len: int = 5, seed: int = 42
    ):
        super().__init__(vocab_size, seed)
        self.subsequence_len = subsequence_len
        # Reserve special tokens
        self.query_token = vocab_size - 1
        self.padding_token = vocab_size - 2

    def generate_batch(
        self, batch_size: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate copy/recall examples."""
        # Generate random sequences
        input_ids = torch.randint(0, self.vocab_size - 2, (batch_size, seq_len))

        # Generate target subsequence to copy
        target_subseq = torch.randint(
            0, self.vocab_size - 2, (batch_size, self.subsequence_len)
        )

        # Place target subsequence in first quarter
        start_pos = torch.randint(0, seq_len // 4, (batch_size,))
        for i in range(batch_size):
            end_pos = min(start_pos[i] + self.subsequence_len, seq_len)
            actual_len = end_pos - start_pos[i]
            input_ids[i, start_pos[i] : end_pos] = target_subseq[i, :actual_len]

        # Add query token at the end
        input_ids[:, -1] = self.query_token

        # Labels: predict the target subsequence after the query
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        for i in range(batch_size):
            # Label positions after query token with target subsequence
            if seq_len > 1:
                end_label_pos = min(self.subsequence_len + 1, seq_len)
                labels[i, 1:end_label_pos] = target_subseq[i, : end_label_pos - 1]

        return input_ids, labels


class DriftCurveDataset(SyntheticDataset):
    """Drift curve task: track metric vs increasing sequence length."""

    def __init__(self, vocab_size: int = 1000, seed: int = 42):
        super().__init__(vocab_size, seed)
        # Reserve special tokens
        self.target_token = vocab_size - 1
        self.query_token = vocab_size - 2

    def generate_batch(
        self, batch_size: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate drift curve examples."""
        # Generate random sequences
        input_ids = torch.randint(0, self.vocab_size - 2, (batch_size, seq_len))

        # Plant target token at position 0
        input_ids[:, 0] = self.target_token

        # Add query token at the end
        input_ids[:, -1] = self.query_token

        # Labels: predict target token at query position
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        labels[:, -1] = self.target_token

        return input_ids, labels


class SimpleModel(nn.Module):
    """Simple model for synthetic evaluations."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        block_type: str = "dpassm",
        pretrained_model: str | None = None,
        **block_kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_type = block_type
        self.pretrained_model = pretrained_model

        # Load pretrained model if specified
        if pretrained_model:
            self._load_pretrained_model(pretrained_model, d_model)
        else:
            # Embedding layer
            self.embedding = nn.Embedding(vocab_size, d_model)

        # Attention blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            if block_type == "dpassm":
                block = DPASSMBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    window_size=block_kwargs.get("window_size", 128),
                    ssm_state_dim=block_kwargs.get("ssm_state_dim", 64),
                )
            elif block_type == "blade":
                block = BLADEBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    chunk_size=block_kwargs.get("chunk_size", 128),
                    state_dim=block_kwargs.get("state_dim", 64),
                )
            else:
                raise ValueError(f"Unknown block type: {block_type}")
            self.blocks.append(block)

        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _load_pretrained_model(self, model_name: str, d_model: int) -> None:
        """Load pretrained model and adapt to our architecture."""
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name
            )  # pragma: no cover
            pretrained_model = AutoModel.from_pretrained(model_name)  # pragma: no cover

            # Use pretrained embeddings if available
            if hasattr(pretrained_model, "embeddings"):  # pragma: no cover
                self.embedding = (
                    pretrained_model.embeddings.word_embeddings
                )  # pragma: no cover
                # Resize if needed
                if (
                    self.embedding.weight.shape[0] != self.vocab_size
                ):  # pragma: no cover
                    self.embedding = nn.Embedding(
                        self.vocab_size, d_model
                    )  # pragma: no cover
            else:  # pragma: no cover
                self.embedding = nn.Embedding(
                    self.vocab_size, d_model
                )  # pragma: no cover

            print(f"Loaded pretrained model: {model_name}")  # pragma: no cover
        except Exception as e:
            print(f"Failed to load pretrained model {model_name}: {e}")
            print("Falling back to random initialization")
            self.embedding = nn.Embedding(self.vocab_size, d_model)

    def forward(
        self, input_ids: torch.Tensor, state: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass."""
        x = self.embedding(input_ids)

        # Apply blocks
        for block in self.blocks:
            if self.block_type == "dpassm":
                x, state = block(x, state)
            else:  # blade
                x, state = block(x, state)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def pretrain_model(
    model: nn.Module,
    dataset: SyntheticDataset,
    seq_len: int,
    num_steps: int = 100,
    batch_size: int = 8,
    device: str = "cpu",
    learning_rate: float = 1e-3,
) -> dict[str, float]:
    """Lightweight pretraining on synthetic dataset."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    print(f"Pretraining for {num_steps} steps...")
    for _step in tqdm(range(num_steps), desc="Pretraining"):
        optimizer.zero_grad()

        # Generate batch
        input_ids, labels = dataset.generate_batch(batch_size, seq_len)
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(input_ids)

        # Compute loss
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()

        # Compute accuracy
        mask = labels != -100
        if mask.any():
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / num_steps
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    print(f"Pretraining completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "total_correct": total_correct,
        "total_tokens": total_tokens,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: SyntheticDataset,
    seq_len: int,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate model on synthetic dataset."""
    model.eval()
    total_correct = 0
    total_tokens = 0

    for _ in range(10):  # Evaluate on 10 batches
        input_ids, labels = dataset.generate_batch(batch_size, seq_len)
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids)

        # Compute accuracy only on non-ignored positions
        mask = labels != -100
        if mask.any():
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    return {
        "accuracy": accuracy,
        "total_correct": total_correct,
        "total_tokens": total_tokens,
    }


def run_passkey_task(
    seq_len: int,
    max_len: int,
    model_path: str,
    block_type: str,
    device: str = "cpu",
    pretrain_steps: int = 0,
    pretrained_model: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run passkey retrieval task."""
    print(f"Running passkey task with seq_len={seq_len}, max_len={max_len}")

    # Create dataset and model
    dataset = PasskeyDataset()
    model = SimpleModel(
        vocab_size=1000,
        d_model=kwargs.get("d_model", 256),
        n_heads=kwargs.get("n_heads", 8),
        n_layers=kwargs.get("n_layers", 2),
        block_type=block_type,
        pretrained_model=pretrained_model,
        window_size=kwargs.get("window_size", 128),
        ssm_state_dim=kwargs.get("ssm_state_dim", 64),
        chunk_size=kwargs.get("chunk_size", 128),
        state_dim=kwargs.get("state_dim", 64),
    ).to(device)

    # Pretrain if requested
    if pretrain_steps > 0:  # pragma: no cover
        pretrain_model(
            model, dataset, seq_len, pretrain_steps, device=device
        )  # pragma: no cover

    # Evaluate at different sequence lengths
    results = {}
    lengths = (
        [seq_len]
        if seq_len == max_len
        else np.logspace(np.log10(seq_len), np.log10(max_len), 5).astype(int)
    )

    for length in lengths:
        print(f"Evaluating at length {length}")
        metrics = evaluate_model(model, dataset, length, device=device)
        results[length] = metrics

    return results


def run_copy_recall_task(
    seq_len: int,
    max_len: int,
    model_path: str,
    block_type: str,
    device: str = "cpu",
    pretrain_steps: int = 0,
    pretrained_model: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run copy/recall task."""
    print(f"Running copy/recall task with seq_len={seq_len}, max_len={max_len}")

    # Create dataset and model
    dataset = CopyRecallDataset(subsequence_len=kwargs.get("subsequence_len", 5))
    model = SimpleModel(
        vocab_size=1000,
        d_model=kwargs.get("d_model", 256),
        n_heads=kwargs.get("n_heads", 8),
        n_layers=kwargs.get("n_layers", 2),
        block_type=block_type,
        pretrained_model=pretrained_model,
        window_size=kwargs.get("window_size", 128),
        ssm_state_dim=kwargs.get("ssm_state_dim", 64),
        chunk_size=kwargs.get("chunk_size", 128),
        state_dim=kwargs.get("state_dim", 64),
    ).to(device)

    # Pretrain if requested
    if pretrain_steps > 0:  # pragma: no cover
        pretrain_model(
            model, dataset, seq_len, pretrain_steps, device=device
        )  # pragma: no cover

    # Evaluate at different sequence lengths
    results = {}
    lengths = (
        [seq_len]
        if seq_len == max_len
        else np.logspace(np.log10(seq_len), np.log10(max_len), 5).astype(int)
    )

    for length in lengths:
        print(f"Evaluating at length {length}")
        metrics = evaluate_model(model, dataset, length, device=device)
        results[length] = metrics

    return results


def run_drift_curve_task(
    seq_len: int,
    max_len: int,
    model_path: str,
    block_type: str,
    device: str = "cpu",
    pretrain_steps: int = 0,
    pretrained_model: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run drift curve task."""
    print(f"Running drift curve task with seq_len={seq_len}, max_len={max_len}")

    # Create dataset and model
    dataset = DriftCurveDataset()
    model = SimpleModel(
        vocab_size=1000,
        d_model=kwargs.get("d_model", 256),
        n_heads=kwargs.get("n_heads", 8),
        n_layers=kwargs.get("n_layers", 2),
        block_type=block_type,
        pretrained_model=pretrained_model,
        window_size=kwargs.get("window_size", 128),
        ssm_state_dim=kwargs.get("ssm_state_dim", 64),
        chunk_size=kwargs.get("chunk_size", 128),
        state_dim=kwargs.get("state_dim", 64),
    ).to(device)

    # Pretrain if requested
    if pretrain_steps > 0:  # pragma: no cover
        pretrain_model(
            model, dataset, seq_len, pretrain_steps, device=device
        )  # pragma: no cover

    # Evaluate at many sequence lengths for smooth curve
    results = {}
    lengths = np.logspace(np.log10(seq_len), np.log10(max_len), 10).astype(int)

    for length in tqdm(lengths, desc="Evaluating drift curve"):
        metrics = evaluate_model(model, dataset, length, device=device)
        results[length] = metrics

    return results


def plot_drift_curve(results: dict[str, Any], task_name: str, output_dir: Path):
    """Plot drift curve and save to file."""
    lengths = sorted(results.keys())
    accuracies = [results[length]["accuracy"] for length in lengths]

    plt.figure(figsize=(10, 6))
    plt.plot(lengths, accuracies, "b-o", linewidth=2, markersize=6)
    plt.xlabel("Sequence Length")
    plt.ylabel("Accuracy")
    plt.title(f"{task_name} - Performance vs Sequence Length")
    plt.grid(True, alpha=0.3)
    plt.xscale("log")

    # Save plot
    output_file = output_dir / f"{task_name.lower().replace(' ', '_')}_drift_curve.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved drift curve plot to {output_file}")


def main():  # pragma: no cover
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Synthetic long-context evaluation tasks"
    )
    parser.add_argument(
        "--task",
        choices=["passkey", "copy_recall", "drift_curve"],
        required=True,
        help="Task to run",
    )
    parser.add_argument(
        "--seq_len", type=int, default=32768, help="Minimum sequence length"
    )
    parser.add_argument(
        "--max_len", type=int, default=131072, help="Maximum sequence length"
    )
    parser.add_argument(
        "--model", type=str, default="", help="Model path or HF ID (not used yet)"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="HuggingFace model ID to load pretrained embeddings (e.g., 'gpt2')",
    )
    parser.add_argument(
        "--pretrain_steps",
        type=int,
        default=0,
        help="Number of pretraining steps on synthetic task (0 = no pretraining)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for pretraining",
    )
    parser.add_argument(
        "--block",
        choices=["dpassm", "blade"],
        default="dpassm",
        help="Attention block type",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument(
        "--cpu_mode", action="store_true", help="CPU-friendly mode for short sequences"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/synthetic",
        help="Output directory for plots",
    )

    # Model parameters
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument(
        "--n_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")

    # Block-specific parameters
    parser.add_argument(
        "--window_size", type=int, default=128, help="Window size for DP-ASSM"
    )
    parser.add_argument(
        "--ssm_state_dim", type=int, default=64, help="SSM state dimension"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=128, help="Chunk size for BLADE"
    )
    parser.add_argument(
        "--state_dim", type=int, default=64, help="State dimension for BLADE"
    )
    parser.add_argument(
        "--subsequence_len",
        type=int,
        default=5,
        help="Subsequence length for copy/recall",
    )

    args = parser.parse_args()

    # Adjust for CPU mode
    if args.cpu_mode:
        args.seq_len = min(args.seq_len, 1024)
        args.max_len = min(args.max_len, 1024)
        args.device = "cpu"
        print("CPU mode: limiting sequence lengths to 1024")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare kwargs for model
    kwargs = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "window_size": args.window_size,
        "ssm_state_dim": args.ssm_state_dim,
        "chunk_size": args.chunk_size,
        "state_dim": args.state_dim,
        "subsequence_len": args.subsequence_len,
    }

    # Run task
    if args.task == "passkey":
        results = run_passkey_task(
            args.seq_len,
            args.max_len,
            args.model,
            args.block,
            args.device,
            args.pretrain_steps,
            args.pretrained_model,
            **kwargs,
        )
    elif args.task == "copy_recall":
        results = run_copy_recall_task(
            args.seq_len,
            args.max_len,
            args.model,
            args.block,
            args.device,
            args.pretrain_steps,
            args.pretrained_model,
            **kwargs,
        )
    elif args.task == "drift_curve":
        results = run_drift_curve_task(
            args.seq_len,
            args.max_len,
            args.model,
            args.block,
            args.device,
            args.pretrain_steps,
            args.pretrained_model,
            **kwargs,
        )

    # Print results
    print(f"\nResults for {args.task}:")
    for length, metrics in results.items():
        print(f"Length {length}: Accuracy = {metrics['accuracy']:.4f}")

    # Save results to JSON (convert numpy int64 keys to Python ints)
    results_file = output_dir / f"{args.task}_results.json"
    results_json = {int(k): v for k, v in results.items()}
    with open(results_file, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved results to {results_file}")

    # Plot drift curve if applicable
    if args.task == "drift_curve":
        plot_drift_curve(results, args.task, output_dir)


if __name__ == "__main__":
    main()
