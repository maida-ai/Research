#!/usr/bin/env python3
"""Quick benchmark script for rapid testing."""

import argparse
import sys

import torch

from benchmark.benchmark.base import BenchmarkConfig
from benchmark.benchmark.blocks import run_comprehensive_benchmark


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quick benchmark for attention blocks")

    parser.add_argument(
        "--blocks", type=str, nargs="+", help="Blocks to test (BLADE, DPASSM, etc.)"
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cuda, cpu)"
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Create config for quick benchmark
    config = BenchmarkConfig(
        sequence_lengths=[args.seq_len],
        batch_size=args.batch_size,
        device=device,
        warmup_runs=2,
        benchmark_runs=args.runs,
        verbose=True,
        save_results=False,
    )

    print(f"Quick benchmark: {args.seq_len} tokens, {args.batch_size} batch, {device}")
    print("=" * 60)

    # Run benchmark
    try:
        results = run_comprehensive_benchmark(config=config, blocks_to_test=args.blocks)

        # Print quick summary
        print("\nQuick Results:")
        print("-" * 30)
        for block_name, block_results in results.items():
            for result in block_results:
                print(
                    f"{block_name}: {result.forward_time_mean * 1000:.2f}ms "
                    f"({result.throughput_tokens_per_sec:.0f} tokens/s)"
                )

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
