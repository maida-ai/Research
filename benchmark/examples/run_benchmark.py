#!/usr/bin/env python3
"""Main benchmark script for running comprehensive benchmarks."""

import argparse
import json
import sys

import torch

from benchmark.benchmark.base import BenchmarkConfig
from benchmark.benchmark.blocks import (
    compare_blocks_performance,
    run_comprehensive_benchmark,
)
from benchmark.benchmark.lightning_benchmark import (
    create_benchmark_config_for_lightning,
)
from benchmark.benchmark.utils import get_system_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark efficient long-context attention blocks"
    )

    # Model configuration
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument(
        "--n-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")

    # Sequence lengths
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
        help="Sequence lengths to benchmark",
    )

    # Benchmarking configuration
    parser.add_argument(
        "--warmup-runs", type=int, default=3, help="Number of warmup runs"
    )
    parser.add_argument(
        "--benchmark-runs", type=int, default=10, help="Number of benchmark runs"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)"
    )

    # PyTorch optimizations
    parser.add_argument(
        "--no-torch-benchmark", action="store_true", help="Disable torch benchmark"
    )
    parser.add_argument(
        "--no-cudnn-benchmark", action="store_true", help="Disable cuDNN benchmark"
    )
    parser.add_argument(
        "--cudnn-deterministic", action="store_true", help="Enable cuDNN deterministic"
    )

    # Memory profiling
    parser.add_argument(
        "--no-memory-profile", action="store_true", help="Disable memory profiling"
    )

    # Output configuration
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./benchmark_results",
        help="Results directory",
    )
    parser.add_argument("--output-file", type=str, help="Output file name")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")

    # Block selection
    parser.add_argument(
        "--blocks",
        type=str,
        nargs="+",
        choices=["BLADE", "DPASSM", "Longformer", "BigBird"],
        help="Specific blocks to benchmark",
    )

    # Lightning mode
    parser.add_argument(
        "--lightning-mode", action="store_true", help="Use Lightning-optimized settings"
    )

    return parser.parse_args()


def create_config_from_args(args) -> BenchmarkConfig:
    """Create benchmark config from command line arguments."""

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Create config
    if args.lightning_mode:
        config = create_benchmark_config_for_lightning(
            d_model=args.d_model,
            n_heads=args.n_heads,
            batch_size=args.batch_size,
            sequence_lengths=args.sequence_lengths,
            device=device,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
            torch_benchmark=not args.no_torch_benchmark,
            cudnn_benchmark=not args.no_cudnn_benchmark,
            cudnn_deterministic=args.cudnn_deterministic,
            profile_memory=not args.no_memory_profile,
            verbose=not args.quiet,
            results_dir=args.results_dir,
        )
    else:
        config = BenchmarkConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            batch_size=args.batch_size,
            sequence_lengths=args.sequence_lengths,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
            device=device,
            torch_benchmark=not args.no_torch_benchmark,
            cudnn_benchmark=not args.no_cudnn_benchmark,
            cudnn_deterministic=args.cudnn_deterministic,
            profile_memory=not args.no_memory_profile,
            verbose=not args.quiet,
            results_dir=args.results_dir,
        )

    return config


def print_system_info():
    """Print system information."""
    info = get_system_info()
    print("System Information:")
    print("-" * 50)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("-" * 50)


def main():
    """Main function."""
    args = parse_args()

    # Print system info
    if not args.quiet:
        print_system_info()

    # Create config
    config = create_config_from_args(args)

    # Run benchmark
    try:
        results = run_comprehensive_benchmark(config=config, blocks_to_test=args.blocks)

        # Save results with custom filename if specified
        if args.output_file:
            import os

            os.makedirs(config.results_dir, exist_ok=True)
            filepath = os.path.join(config.results_dir, args.output_file)

            results_data = []
            for block_results in results.values():
                results_data.extend([result.to_dict() for result in block_results])

            with open(filepath, "w") as f:
                json.dump(results_data, f, indent=2)

            if not args.quiet:
                print(f"\nResults saved to {filepath}")

        # Print comparison
        if not args.quiet and len(results) > 1:
            print("\nPerformance Comparison (Forward Time):")
            print("-" * 50)
            comparison = compare_blocks_performance(results, "forward_time_mean")

            for seq_len in config.sequence_lengths:
                print(f"\nSequence Length {seq_len}:")
                block_times = {}
                for block_name, times in comparison.items():
                    if seq_len in times:
                        block_times[block_name] = times[seq_len]

                # Sort by performance
                sorted_blocks = sorted(block_times.items(), key=lambda x: x[1])
                for i, (block_name, time_ms) in enumerate(sorted_blocks):
                    print(f"  {i + 1}. {block_name}: {time_ms * 1000:.2f}ms")

        return 0

    except Exception as e:
        print(f"Error during benchmarking: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
